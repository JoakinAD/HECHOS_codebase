# webScrapping/crawlers/ElPlural.py
import json
import re
import time
import uuid
import html as html_lib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from crawlers.Crawler import Crawler

class ElPlural(Crawler):
    SECTION_URLS = (
        "https://www.elplural.com/politica",
        "https://www.elplural.com/politica/espana",
        "https://www.elplural.com/politica/internacional",
        "https://www.elplural.com/autonomias",
        "https://www.elplural.com/economia",
        "https://www.elplural.com/sociedad",
        "https://www.elplural.com/sociedad/tribunales",
        "https://www.elplural.com/autonomias/andalucia",
        "https://www.elplural.com/autonomias/cataluna",
        "https://www.elplural.com/autonomias/madrid",
        "https://www.elplural.com/autonomias/galicia",
        "https://www.elplural.com/autonomias/castilla-leon",
        "https://www.elplural.com/autonomias/castilla-la-mancha",
        "https://www.elplural.com/autonomias/comunidad-valenciana",
        "https://www.elplural.com/autonomias/asturias",
        "https://www.elplural.com/memoria.html",
    )
    PAYWALL_CTA_RE = re.compile(
    r"(?is)\b("
    r"súmate a|apoya nuestro trabajo|navega sin publicidad|entra a todos los contenidos|hazte socio|hazte\s+socia|"
    r"hazte miembro|hazte\s+miembr[oa]|apóyanos|apoyanos"
    r")\b.*$"
)
    # Ej: https://www.elplural.com/politica/espana/..._380703102
    ARTICLE_RE = re.compile(r"_\d{6,}$")

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "EL PLURAL"

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0 Safari/537.36"
                ),
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            }
        )

    # ---------------------------
    # URL / HTTP
    # ---------------------------
    def _add_page_param(self, base_url: str, page: int) -> str:
        """
        ElPlural pagina con ?_page=2
        """
        p = urlparse(base_url)
        q = parse_qs(p.query)
        q["_page"] = [str(page)]
        new_query = urlencode(q, doseq=True)
        return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))

    def _get_soup(self, url: str, timeout: int = 20) -> BeautifulSoup | None:
        try:
            r = self._session.get(url, timeout=timeout)
            if r.status_code != 200:
                return None
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            return None

    def _is_article_url(self, href: str) -> bool:
        if not href:
            return False
        href = href.strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            return False

        abs_url = urljoin(self.url, href)
        p = urlparse(abs_url)
        if "elplural.com" not in p.netloc:
            return False

        path = p.path.rstrip("/")

        # evita páginas de sección/paginación/etiquetas
        if any(x in path.lower() for x in ("/tags/", "/tag/", "/autor/", "/search", "/busc")):
            return False

        return bool(self.ARTICLE_RE.search(path))

    def _extract_section_links(self, soup: BeautifulSoup) -> list[str]:
        """
        En listados, los links suelen estar en:
        div.item a[href]
        """
        urls: list[str] = []
        seen: set[str] = set()

        for a in soup.select("div.list-items div.item a[href], article a[href], h2 a[href], h3 a[href]"):
            href = (a.get("href") or "").strip()
            if not self._is_article_url(href):
                continue
            u = urljoin(self.url, href)
            if u not in seen:
                seen.add(u)
                urls.append(u)

        return urls

    # ---------------------------
    # Text helpers
    # ---------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        text = html_lib.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1:
            t = h1.get_text(" ", strip=True)
            if t:
                return t

        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return og["content"].strip()

        t = soup.find("title")
        return t.get_text(strip=True) if t else ""

    def _body_from_jsonld(self, soup: BeautifulSoup) -> str:
        for s in soup.select('script[type="application/ld+json"]'):
            raw = (s.string or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            candidates = data if isinstance(data, list) else [data]
            for obj in list(candidates):
                if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
                    candidates.extend([x for x in obj["@graph"] if isinstance(x, dict)])

            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                typ = obj.get("@type") or obj.get("type")
                if isinstance(typ, list):
                    typ = typ[0] if typ else None
                if typ in ("NewsArticle", "Article", "ReportageNewsArticle"):
                    body = obj.get("articleBody")
                    if isinstance(body, str) and body.strip():
                        return self._clean_text(body)

        return ""

    def _body_from_dom(self, soup: BeautifulSoup) -> str:
        # intenta pillar el contenedor típico de cuerpo; si no, usa <article>
        root = (
            soup.select_one("div.news-body-complete")
            or soup.select_one("div.news-body")
            or soup.select_one("div.article-body")
            or soup.select_one("div.article__body")
            or soup.find("article")
            or soup
        )

        if not root:
            return ""

        # limpia basura común
        for tag in root.select(
            "script,style,noscript,header,footer,nav,form,aside,figure,iframe,"
            ".share,.social,.related,.tags,.breadcrumbs,.newsletter,.banner,.ads,.advertising"
        ):
            tag.decompose()

        parts: list[str] = []
        for p in root.select("p"):
            txt = p.get_text(" ", strip=True)
            if not txt:
                continue
            if len(txt) < 35:
                continue
            low = txt.lower()
            # filtros básicos anti-basura
            if "suscríbete" in low or "suscrib" in low or "newsletter" in low:
                continue
            parts.append(txt)

        return self._clean_text("\n\n".join(parts))

    def _extract_body(self, soup: BeautifulSoup) -> str:
        body = self._body_from_jsonld(soup)
        if not body or len(body) < 250:
            body = self._body_from_dom(soup)

        body = self._strip_cta_tail(body)
        return body


    def _strip_cta_tail(self, text: str) -> str:
        if not text:
            return text
        t = text.strip()

        # Recorta desde el primer CTA (si aparece)
        t2 = self.PAYWALL_CTA_RE.sub("", t).strip()

        # Limpieza extra por si queda comilla/coma suelta
        t2 = re.sub(r'[\s,"\']+$', "", t2).strip()
        return t2

    # ---------------------------
    # Date helpers (solo HOY)
    # ---------------------------
    @staticmethod
    def _normalize_dt(dt: str) -> str:
        dt2 = dt.strip().replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(dt2)
        except ValueError:
            return ""

        if d.tzinfo is None:
            # si viene sin tz, lo devolvemos como naive ISO
            return d.isoformat(timespec="seconds")

        d_utc = d.astimezone(timezone.utc)
        return d_utc.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _extract_date_iso(self, soup: BeautifulSoup, link: str) -> str:
        # 1) JSON-LD datePublished/dateModified
        for s in soup.select('script[type="application/ld+json"]'):
            raw = (s.string or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            candidates = data if isinstance(data, list) else [data]
            for obj in list(candidates):
                if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
                    candidates.extend([x for x in obj["@graph"] if isinstance(x, dict)])

            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                typ = obj.get("@type") or obj.get("type")
                if isinstance(typ, list):
                    typ = typ[0] if typ else None
                if typ in ("NewsArticle", "Article", "ReportageNewsArticle"):
                    dt = obj.get("dateModified") or obj.get("datePublished")
                    if isinstance(dt, str) and dt.strip():
                        iso = self._normalize_dt(dt)
                        if iso:
                            return iso

        # 2) meta article:published_time
        m = soup.find("meta", attrs={"property": "article:published_time"})
        if m and m.get("content"):
            iso = self._normalize_dt(m["content"])
            if iso:
                return iso

        # 3) fallback: intentar dd/mm/yyyy en texto visible (si hubiera)
        txt = soup.get_text("\n", strip=True)
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})", txt)
        if m:
            dd, mm, yyyy, hh, mins = m.groups()
            d_local = datetime(
                int(yyyy), int(mm), int(dd), int(hh), int(mins),
                tzinfo=ZoneInfo("Europe/Madrid")
            )
            d_utc = d_local.astimezone(timezone.utc)
            return d_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

        return ""

    @staticmethod
    def _is_today(dt_iso: str) -> bool:
        try:
            d = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        except Exception:
            return False

        madrid = ZoneInfo("Europe/Madrid")
        today_madrid = datetime.now(madrid).date() - timedelta(days = 1)

        if d.tzinfo is None:
            d = d.replace(tzinfo=madrid)
        else:
            d = d.astimezone(madrid)

        return d.date() == today_madrid

    @staticmethod
    def _iso_to_ddmmyyyy(dt_iso: str) -> str:
        try:
            d = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
            return d.strftime("%d-%m-%Y")
        except Exception:
            return ""

    # ---------------------------
    # Main crawl
    # ---------------------------
    def crawl(self, max_news: int = 300, sleep_s: float = 0.05) -> list[dict]:
        """
        2 páginas por sección:
          page=1 (sin _page) y page=2 (?_page=2)
        Solo artículos de HOY (Europe/Madrid).
        """
        urls: list[str] = []
        seen: set[str] = set()

        for sec in self.SECTION_URLS:
            # página 1
            s1 = self._get_soup(sec)
            if s1:
                for u in self._extract_section_links(s1):
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)

            # página 2
            sec2 = self._add_page_param(sec, 2)
            s2 = self._get_soup(sec2)
            if s2:
                for u in self._extract_section_links(s2):
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)

        if not urls:
            return []

        data: list[dict] = []
        for link in urls[:max_news]:
            time.sleep(sleep_s)
            soup = self._get_soup(link)
            if not soup:
                continue

            dt_iso = self._extract_date_iso(soup, link)
            if not dt_iso or not self._is_today(dt_iso):
                continue

            headline = self._extract_title(soup)
            body = self._extract_body(soup)

            if not headline or not body or len(body) < 300:
                continue

            data.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link.strip())),
                    "headline": headline,
                    "body": body,
                    "link": link,
                    "date": self._iso_to_ddmmyyyy(dt_iso),  # dd-mm-aaaa
                    "bias": "N",
                    "newspaper": self.newspaper,
                }
            )

        return data

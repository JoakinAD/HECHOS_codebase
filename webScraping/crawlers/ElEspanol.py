# webScrapping/crawlers/ElEspanol.py
import json
import re
import time
import uuid
import html as html_lib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from crawlers.Crawler import Crawler


class ElEspanol(Crawler):
    SECTION_URLS = (
        "https://www.elespanol.com/ultimas/",
        "https://www.elespanol.com/espana/",
        "https://www.elespanol.com/espana/politica/",
        "https://www.elespanol.com/espana/tribunales/",
        "https://www.elespanol.com/mundo/",
        "https://www.elespanol.com/invertia/",
        "https://www.elespanol.com/sociedad/",
        "https://www.elespanol.com/sociedad/sucesos/",
        "https://www.elespanol.com/sociedad/educacion/",
        "https://www.elespanol.com/reportajes/",
        "https://www.elespanol.com/ciencia/",
        "https://www.elespanol.com/ciencia/medio-ambiente/",
        "https://www.elespanol.com/ciencia/salud/",
        "https://www.elespanol.com/madrid/",
        "https://www.elespanol.com/castilla-y-leon/",
        "https://www.elespanol.com/espana/andalucia/",
        "https://www.elespanol.com/espana/comunidad-valenciana/",
        "https://www.elespanol.com/espana/galicia/",
    )

    # Links de artículo típicos:
    # /espana/.../20260205/.../1003744118717_0.html
    # https://www.elespanol.com/.../20260205/.../100...html
    ARTICLE_RE = re.compile(r"/\d{8}/[^\"?#]+/\d+_\d+\.html$", re.I)

    # Rutas que queremos evitar (opinion, tribunas, etc.)
    SKIP_PATH_PARTS = (
        "/opinion/",
        "/tribunas/",
        "/reportajes/",  # si quieres incluir reportajes, quita esto
        "/podcast/",
        "/videos/",
        "/album/",
        "/galerias/",
    )

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "EL ESPAÑOL"

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0 Safari/537.36"
                ),
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    # ---------------------------
    # HTTP / URL helpers
    # ---------------------------
    def _get_soup(self, url: str, timeout: int = 20) -> BeautifulSoup | None:
        try:
            r = self._session.get(url, timeout=timeout, allow_redirects=True)
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

        if "elespanol.com" not in p.netloc:
            return False

        path = p.path.lower()

        if any(x in path for x in self.SKIP_PATH_PARTS):
            return False

        return bool(self.ARTICLE_RE.search(path))

    def _extract_section_links(self, soup: BeautifulSoup) -> list[str]:
        """
        En España / Mundo / Sociedad, suelen estar en:
          article.art h2.art__title a[href]
        En Invertia también, pero a veces con href relativo "/invertia/..."
        """
        urls, seen = [], set()

        for a in soup.select("article.art h2.art__title a[href], article.art a[href]"):
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
        """
        Fallback DOM: intenta capturar <p> del artículo.
        Como no tenemos aquí el HTML del artículo completo, lo hacemos defensivo.
        """
        container = (
            soup.select_one("article")
            or soup.select_one("[itemprop='articleBody']")
            or soup.select_one(".article")
            or soup
        )
        if not container:
            return ""

        for tag in container.select(
            "script,style,noscript,header,footer,nav,form,aside,figure,iframe,.adv,.advertising,.rrss"
        ):
            tag.decompose()

        parts: list[str] = []
        for p in container.select("p"):
            txt = p.get_text(" ", strip=True)
            if not txt:
                continue
            if len(txt) < 40:
                continue
            parts.append(txt)

        return self._clean_text("\n\n".join(parts))

    def _extract_body(self, soup: BeautifulSoup) -> str:
        body = self._body_from_jsonld(soup)
        if not body or len(body) < 250:
            body = self._body_from_dom(soup)
        return body

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
            return d.isoformat(timespec="seconds")

        d_utc = d.astimezone(timezone.utc)
        return (
            d_utc.replace(tzinfo=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    def _extract_date_iso(self, soup: BeautifulSoup) -> str:
        # 1) JSON-LD
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

        # 2) Meta
        for prop in ("article:published_time", "article:modified_time"):
            m = soup.find("meta", attrs={"property": prop})
            if m and m.get("content"):
                iso = self._normalize_dt(m["content"])
                if iso:
                    return iso

        # 3) <time datetime="YYYY-MM-DD"> (como en tus secciones)
        t = soup.find("time")
        if t and t.get("datetime"):
            # aquí suele venir "2026-02-05"
            dt = t["datetime"].strip()
            # normalizamos a ISO Z a medianoche si solo viene fecha
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dt):
                return dt + "T00:00:00Z"
            iso = self._normalize_dt(dt)
            if iso:
                return iso

        return ""

    @staticmethod
    def _is_today(dt_iso: str) -> bool:
        try:
            d = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        except Exception:
            return False
        return d.date() == datetime.now().date() - timedelta(days = 1)

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
    def crawl(self, max_news: int = 250, sleep_s: float = 0.05) -> list[dict]:
        urls: list[str] = []
        seen: set[str] = set()

        # 1) recolectar links de portada de secciones
        for sec in self.SECTION_URLS:
            sec_soup = self._get_soup(sec)
            if not sec_soup:
                continue
            for u in self._extract_section_links(sec_soup):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)

        if not urls:
            return []

        data: list[dict] = []

        # 2) visitar cada noticia y filtrar por HOY (con fecha del artículo, no la sección)
        for link in urls[:max_news]:
            time.sleep(sleep_s)
            soup = self._get_soup(link)
            if not soup:
                continue

            dt_iso = self._extract_date_iso(soup)
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

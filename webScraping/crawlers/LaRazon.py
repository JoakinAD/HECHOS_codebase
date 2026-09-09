# webScrapping/crawlers/LaRazon.py
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


class LaRazon(Crawler):
    SECTION_URLS = (
        "https://www.larazon.es/espana/",
        "https://www.larazon.es/internacional/",
        "https://www.larazon.es/economia/",
        "https://www.larazon.es/sociedad/",
    )

    # Ej real:
    # https://www.larazon.es/espana/..._2026020469837c622f00a04688f78f3c.html
    ARTICLE_RE = re.compile(r"_\d{8}[0-9a-z]+\.html$", re.I)

    # ---------------------------
    # Filtrado "no noticias" (servicio / viral / lifestyle)
    # ---------------------------
    LIFESTYLE_TAGS_DENY = {
        "hogar", "decoración", "bricolaje", "cocina", "recetas", "belleza", "moda",
        "salud", "bienestar", "psicología", "curiosidades", "tiktok", "mascotas",
        "horóscopo", "astrología", "viajes", "motor", "tecnología", "compras",
    }

    HARD_NEWS_TAGS_HINT = {
        "pp", "psoe", "vox", "sumar", "podemos", "gobierno", "congreso", "senado",
        "tribunales", "aemet", "ucrania", "israel", "gaza", "rusia",
        "ministerio", "sanidad", "hacienda", "economía", "inflación",
        "elecciones", "ue", "otan",
    }

    HOWTO_HEADLINE_RE = re.compile(
        r"^(¿\s*)?(cómo|qué|por qué|cuál|cuáles)\b|"
        r"\b(según|experto|electricista|psicólogo|en tiktok|trucos|consejos)\b",
        re.IGNORECASE,
    )

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "LA RAZON"

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
    # HTTP / URL helpers
    # ---------------------------
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
        if "larazon.es" not in p.netloc:
            return False

        path = p.path.lower()
        if "/opinion/" in path:
            return False

        return bool(self.ARTICLE_RE.search(path))

    def _extract_section_links(self, soup: BeautifulSoup) -> list[str]:
        urls, seen = [], set()
        for a in soup.select("article a[href], h2 a[href], h3 a[href]"):
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
        article = soup.find("article") or soup
        if not article:
            return ""

        for tag in article.select("script,style,noscript,header,footer,nav,form,aside,figure,iframe"):
            tag.decompose()

        parts = []
        for p in article.select("p"):
            txt = p.get_text(" ", strip=True)
            if not txt or len(txt) < 35:
                continue
            low = txt.lower()
            if "suscríbete" in low or "suscrib" in low or "inicia sesión" in low:
                continue
            parts.append(txt)

        return self._clean_text("\n\n".join(parts))

    def _extract_body(self, soup: BeautifulSoup) -> str:
        body = self._body_from_jsonld(soup)
        return body if body and len(body) >= 250 else self._body_from_dom(soup)

    # ---------------------------
    # Filtrado La Razón (evitar "servicio / viral")
    # ---------------------------
    def _extract_archivado_tags(self, soup: BeautifulSoup) -> set[str]:
        """
        Intenta capturar tags cercanos al texto "ARCHIVADO EN".
        No depende de clases específicas.
        """
        tags: set[str] = set()
        marker = soup.find(string=re.compile(r"ARCHIVADO EN", re.IGNORECASE))
        if not marker:
            return tags

        container = marker.parent
        for _ in range(3):
            if container and container.parent:
                container = container.parent

        for a in container.find_all("a", href=True):
            t = a.get_text(" ", strip=True).lower()
            if t:
                tags.add(t)

        return tags

    def _should_skip_article(self, link: str, headline: str, soup: BeautifulSoup) -> bool:
        """
        True => descartar (lifestyle/servicio)
        False => mantener (noticia)
        """
        low_link = (link or "").lower()

        # 1) Hard reject por patrón observado: -p7m_ / _p7m_
        if "-p7m_" in low_link or "_p7m_" in low_link:
            return True

        # 2) Tags (si existen)
        tags = self._extract_archivado_tags(soup)

        # deny por tags claros de lifestyle
        if any(t in self.LIFESTYLE_TAGS_DENY for t in tags):
            return True

        # allow por tags de hard-news
        if any(t in self.HARD_NEWS_TAGS_HINT for t in tags):
            return False

        # 3) Fallback: heurística por titular (cuando tags no ayudan)
        if headline and self.HOWTO_HEADLINE_RE.search(headline):
            return True

        return False

    # ---------------------------
    # Date helpers (solo HOY)
    # ---------------------------
    @staticmethod
    def _normalize_dt(dt: str) -> str:
        """
        Normaliza a ISO 8601. Si trae tz -> UTC con 'Z'.
        """
        dt2 = dt.strip().replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(dt2)
        except ValueError:
            return ""

        if d.tzinfo is None:
            return d.isoformat(timespec="seconds")

        d_utc = d.astimezone(timezone.utc)
        return d_utc.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _extract_date_iso(self, soup: BeautifulSoup, link: str) -> str:
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

        # 2) Texto visible: "Creada: 04.02.2026 18:06"
        txt = soup.get_text("\n", strip=True)
        m = re.search(r"Creada:\s*(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})", txt)
        if m:
            dd, mm, yyyy, hh, mins = m.groups()
            d_local = datetime(
                int(yyyy), int(mm), int(dd), int(hh), int(mins),
                tzinfo=ZoneInfo("Europe/Madrid")
            )
            d_utc = d_local.astimezone(timezone.utc)
            return d_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

        # 3) Fallback desde URL: _YYYYMMDD....
        m = re.search(r"_(\d{8})[0-9a-z]+\.html$", urlparse(link).path, re.I)
        if m:
            ymd = m.group(1)
            return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}T00:00:00Z"

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
        urls, seen = [], set()
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
        for link in urls[:max_news]:
            time.sleep(sleep_s)
            soup = self._get_soup(link)
            if not soup:
                continue

            dt_iso = self._extract_date_iso(soup, link)
            if not dt_iso or not self._is_today(dt_iso):
                continue

            headline = self._extract_title(soup)
            if not headline:
                continue

            # Filtrado para no scrapear "servicio/viral"
            if self._should_skip_article(link, headline, soup):
                continue

            body = self._extract_body(soup)
            if not body or len(body) < 300:
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

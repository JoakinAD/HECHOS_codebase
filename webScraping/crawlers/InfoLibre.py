# webScrapping/crawlers/InfoLibre.py
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


class InfoLibre(Crawler):
    SECTION_URLS = (
        "https://www.infolibre.es/politica/",
        "https://www.infolibre.es/economia/",
        "https://www.infolibre.es/igualdad/",
        "https://www.infolibre.es/internacional/",
        "https://www.infolibre.es/medioambiente/",
        "https://www.infolibre.es/union-europea/",
        "https://www.infolibre.es/investigaciones/",
    )

    # Ej:
    # https://www.infolibre.es/internacional/..._1_2140160.html
    ARTICLE_RE = re.compile(r"_1_\d+\.html$", re.I)

    # Recorta colas típicas de registro/suscripción/CTA
    CTA_TAIL_RE = re.compile(
        r"(?is)\b("
        r"hazte\s+soci[oa]|suscr[ií]bete|suscripci[oó]n|inicia\s+sesi[oó]n|reg[ií]strate|"
        r"accede\s+para\s+seguir|para\s+seguir\s+leyendo|contenido\s+para\s+socios|"
        r"apoya\s+nuestro\s+trabajo|ap[óo]yanos|navega\s+sin\s+publicidad|"
        r"entra\s+a\s+todos\s+los\s+contenidos|hazte\s+miembr[oa]"
        r")\b.*$"
    )

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "INFOLIBRE"

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

    def _get_text(self, url: str, timeout: int = 20) -> str:
        try:
            r = self._session.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code != 200:
                return ""
            return r.text or ""
        except requests.RequestException:
            return ""

    def _is_article_url(self, href: str) -> bool:
        if not href:
            return False
        href = href.strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            return False

        abs_url = urljoin(self.url, href)
        p = urlparse(abs_url)

        if "infolibre.es" not in p.netloc:
            return False

        path = p.path.lower()

        # Evitar cosas claramente no-noticia
        if "/opinion/" in path or "/blog/" in path or "/humor/" in path:
            return False

        return bool(self.ARTICLE_RE.search(path))

    def _extract_section_links(self, soup: BeautifulSoup) -> list[str]:
        """
        En las secciones, los links suelen estar en:
        - h2.ni-title a[href]
        - figure.ni-figure a[href]
        - y genéricamente dentro de módulos md__new
        """
        urls, seen = [], set()

        selectors = [
            "h2.ni-title a[href]",
            "figure.ni-figure a[href]",
            ".md__new a[href]",
            "article a[href]",
        ]

        for sel in selectors:
            for a in soup.select(sel):
                href = (a.get("href") or "").strip()
                if not self._is_article_url(href):
                    continue
                u = urljoin(self.url, href)
                if u not in seen:
                    seen.add(u)
                    urls.append(u)

        return urls

    def _extract_view_more_links(self, section_soup: BeautifulSoup) -> list[str]:
        """
        Algunas secciones (economía/igualdad/internacional) cargan más contenidos con un botón:
          <button class="view-more-button" data-href="https://www.infolibre.es/webapi/more-contents/...">
        Ese endpoint devuelve HTML con más titulares/enlaces.
        """
        urls: list[str] = []
        btn = section_soup.select_one("button.view-more-button[data-href]")
        if not btn:
            return urls

        more_url = (btn.get("data-href") or "").strip()
        if not more_url:
            return urls

        # puede venir relativo (poco común), pero por si acaso:
        more_url = urljoin(self.url, more_url)

        html = self._get_text(more_url)
        if not html:
            return urls

        more_soup = BeautifulSoup(html, "html.parser")
        return self._extract_section_links(more_soup)

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

    def _strip_cta_tail(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        t = self.CTA_TAIL_RE.sub("", t).strip()
        t = re.sub(r'[\s,"\']+$', "", t).strip()
        return t

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
                        return self._strip_cta_tail(self._clean_text(body))

        return ""

    def _body_from_dom(self, soup: BeautifulSoup) -> str:
        # Intentos de contenedor típico
        container = (
            soup.select_one("article")
            or soup.select_one(".article")
            or soup.select_one(".news")
            or soup
        )
        if not container:
            return ""

        # Quitar basura evidente
        for tag in container.select(
            "script,style,noscript,header,footer,nav,form,aside,figure,iframe,.advertising,.edi-advertising"
        ):
            tag.decompose()

        parts: list[str] = []
        for p in container.select("p"):
            txt = p.get_text(" ", strip=True)
            if not txt:
                continue

            # saltar párrafos muy cortos (firmas, entradillas raras, etc.)
            if len(txt) < 35:
                continue

            low = txt.lower()
            if any(
                k in low
                for k in (
                    "suscríbete",
                    "suscrib",
                    "hazte socio",
                    "hazte socia",
                    "inicia sesión",
                    "regístrate",
                    "apoya nuestro trabajo",
                    "navega sin publicidad",
                    "entra a todos los contenidos",
                )
            ):
                continue

            parts.append(txt)

        body = self._clean_text("\n\n".join(parts))
        body = self._strip_cta_tail(body)
        return body

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
                    dt = obj.get("datePublished")
                    if isinstance(dt, str) and dt.strip():
                        iso = self._normalize_dt(dt)
                        if iso:
                            return iso

        # 2) Meta OpenGraph / article
        for prop in ("article:published_time",):
            m = soup.find("meta", attrs={"property": prop})
            if m and m.get("content"):
                iso = self._normalize_dt(m["content"])
                if iso:
                    return iso

        # 3) <time datetime="...">
        t = soup.find("time")
        if t and t.get("datetime"):
            iso = self._normalize_dt(t["datetime"])
            if iso:
                return iso

        # 4) Fallback: no inventamos fecha (mejor descartar)
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
    def crawl(self, max_news: int = 250, sleep_s: float = 0.05) -> list[dict]:
        urls: list[str] = []
        seen: set[str] = set()

        for sec in self.SECTION_URLS:
            sec_soup = self._get_soup(sec)
            if not sec_soup:
                continue

            # 1) links visibles en la sección
            for u in self._extract_section_links(sec_soup):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)

            # 2) soporte para secciones tipo "Ver siguientes" (webapi/more-contents)
            for u in self._extract_view_more_links(sec_soup):
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
            if not self._accept_publication_date(dt_iso):
                continue

            headline = self._extract_title(soup)
            body = self._extract_body(soup)

            if not headline or not body or len(body) < 300:
                continue

            data.append(
                {
                    # UUID determinista por URL (dedupe fácil)
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link.strip())),
                    "headline": headline,
                    "body": body,
                    "link": link,
                    "date": self._format_publication_date(dt_iso),
                    "bias": "N",
                    "newspaper": self.newspaper,
                }
            )

        return data

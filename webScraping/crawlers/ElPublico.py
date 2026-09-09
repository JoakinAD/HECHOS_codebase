# webScrapping/crawlers/ElPublico.py
import json
import re
import time
import uuid
import html as html_lib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawlers.Crawler import Crawler


class ElPublico(Crawler):
    SECTION_URLS = (
        "https://www.infolibre.es/politica/",
        "https://www.infolibre.es/economia/",
        "https://www.infolibre.es/igualdad/",
        "https://www.infolibre.es/internacional/",
        "https://www.infolibre.es/medioambiente/",
        "https://www.infolibre.es/union-europea/",
        "https://www.infolibre.es/investigaciones/",
    )

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "EL PUBLICO"

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
    # HTTP / URL
    # ---------------------------
    def _get_soup(self, url: str, timeout: int = 20) -> BeautifulSoup | None:
        try:
            r = self._session.get(url, timeout=timeout)
            if r.status_code != 200:
                return None
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException:
            return None

    def _is_valid_article_url(self, href: str) -> bool:
        """
        Público (según tu HTML):
          /politica/slug.html
          /politica/tribunales/slug.html
        Evitar:
          /politica (sección)
          /politica/2 (paginación)
          /politica/300 (paginación)
        """
        if not href:
            return False
        href = href.strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            return False

        abs_url = urljoin(self.url, href)
        p = urlparse(abs_url)

        if "publico.es" not in p.netloc:
            return False

        path = p.path.rstrip("/")
        low = path.lower()

        # descartar secciones base y paginaciones numéricas
        # (ej: /politica/2)
        last = low.split("/")[-1]
        if last.isdigit():
            return False

        # debe acabar en .html (las noticias del listado lo hacen)
        if not low.endswith(".html"):
            return False

        # ruido típico
        deny = (
            "/autor/",
            "/autores/",
            "/opinion/",
            "/blogs/",
            "/tags/",
            "/tag/",
            "/video/",
            "/videos/",
            "/podcast/",
            "/podcasts/",
            "/newsletter",
            "/suscripcion",
            "/suscrib",
        )
        if any(d in low for d in deny):
            return False

        return True

    def _extract_section_links(self, soup: BeautifulSoup) -> list[str]:
        """
        En tu HTML el link está en: h2.title a.page-link[href]
        """
        urls, seen = [], set()

        for a in soup.select("h2.title a.page-link[href], article a.page-link[href]"):
            href = (a.get("href") or "").strip()
            if not self._is_valid_article_url(href):
                continue
            u = urljoin(self.url, href)
            if u not in seen:
                seen.add(u)
                urls.append(u)

        return urls

    # ---------------------------
    # Text
    # ---------------------------
    def _clean_text(self, text: str) -> str:
        text = html_lib.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _strip_paywall_cuts(self, text: str) -> str:
        # por si meten token suelto " ... " o " … " (sin cargarte puntos suspensivos normales)
        if not text:
            return text
        t = re.sub(r"\s+\.\.\.\s+", " ", text)
        t = re.sub(r"\s+…\s+", " ", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        return t.strip()

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
                        return self._strip_paywall_cuts(self._clean_text(body))

        return ""

    def _body_from_dom(self, soup: BeautifulSoup) -> str:
        root = (
            soup.select_one('[itemprop="articleBody"]')
            or soup.select_one("div.article-body")
            or soup.select_one("div.content")
            or soup.find("article")
        )
        if not root:
            return ""

        for tag in root.select("script,style,noscript,header,footer,nav,form,aside,figure,iframe"):
            tag.decompose()

        parts: list[str] = []
        for node in root.select("h2, h3, p, blockquote, li"):
            txt = node.get_text(" ", strip=True)
            if not txt or len(txt) < 35:
                continue
            low = txt.lower()
            if "suscríbete" in low or "inicia sesión" in low:
                continue
            parts.append(txt)

        return self._strip_paywall_cuts(self._clean_text("\n\n".join(parts)))

    def _extract_body(self, soup: BeautifulSoup) -> str:
        body = self._body_from_jsonld(soup)
        if body and len(body) >= 300:
            return body
        return self._body_from_dom(soup)

    # ---------------------------
    # Date
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
        return d_utc.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _extract_date_iso(self, soup: BeautifulSoup, link: str) -> str:
        # 1) JSON-LD (preferido)
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

        # 2) DOM
        t = soup.select_one("time[datetime]")
        if t and t.get("datetime"):
            iso = self._normalize_dt(t["datetime"])
            if iso:
                return iso

        # 3) fallback: por imágenes /uploads/YYYY/MM/DD/... (sale en tu HTML)
        m = re.search(r"/uploads/(\d{4})/(\d{2})/(\d{2})/", soup.decode())
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            return f"{y}-{mo}-{d}T00:00:00Z"

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
    # Main
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
            art_soup = self._get_soup(link)
            if not art_soup:
                continue

            dt_iso = self._extract_date_iso(art_soup, link)
            if not dt_iso or not self._is_today(dt_iso):
                continue

            headline = self._extract_title(art_soup)
            body = self._extract_body(art_soup)

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

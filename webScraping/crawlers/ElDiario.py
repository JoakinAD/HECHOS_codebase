# webScrapping/crawlers/ElDiario.py
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


class ElDiario(Crawler):
    SECTION_URLS = (
        "https://www.eldiario.es/ultimas-noticias/",
        "https://www.eldiario.es/politica/",
        "https://www.eldiario.es/internacional/",
        "https://www.eldiario.es/economia/",
        "https://www.eldiario.es/sociedad/",
        "https://www.eldiario.es/tecnologia/",
        "https://www.eldiario.es/desalambre/",
        "https://www.eldiario.es/focos/igualdad/",
        "https://www.eldiario.es/focos/crisis-climatica/",
        "https://www.eldiario.es/focos/derechos_y_libertades/",
        "https://www.eldiario.es/focos/salud-publica/",
        "https://www.eldiario.es/focos/educacion/",
        "https://www.eldiario.es/focos/vivienda",
        "https://www.eldiario.es/andalucia/",
        "https://www.eldiario.es/catalunya/",
        "https://www.eldiario.es/comunitat-valenciana/",
        "https://www.eldiario.es/euskadi/",
        "https://www.eldiario.es/galicia/",
        "https://www.eldiario.es/madrid/",
    )

    # eldiario suele tener urls tipo /politica/..._1_1234567.html o similares
    ARTICLE_RE = re.compile(r"\.html($|\?)", re.I)

    # por si alguna URL trae fecha /2026/02/04/
    DATE_PATH_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "EL DIARIO"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        })

    def _soup(self, url: str, timeout: int = 20) -> BeautifulSoup | None:
        try:
            r = self.session.get(url, timeout=timeout)
            return BeautifulSoup(r.text, "html.parser") if r.status_code == 200 else None
        except requests.RequestException:
            return None

    @staticmethod
    def _clean(text: str) -> str:
        text = html_lib.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _is_article_url(self, href: str) -> bool:
        if not href:
            return False
        href = href.strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            return False

        abs_url = urljoin(self.url, href)
        p = urlparse(abs_url)
        if "eldiario.es" not in p.netloc:
            return False

        path = p.path.lower()

        # ruido típico (mínimo, puedes ampliar si te hace falta)
        deny = ("/autor/", "/autores/", "/newsletter", "/suscripcion", "/contacto", "/cookies", "/opinion")
        if any(d in path for d in deny):
            return False

        # debe parecer noticia (.html)
        return bool(self.ARTICLE_RE.search(path))

    def _extract_links_from_section(self, soup: BeautifulSoup) -> list[str]:
        urls, seen = [], set()

        # eldiario suele tener titulares en h2/h3 con <a>
        for a in soup.select("article a[href], h2 a[href], h3 a[href]"):
            href = (a.get("href") or "").strip()
            if not self._is_article_url(href):
                continue
            u = urljoin(self.url, href)
            if u not in seen:
                seen.add(u)
                urls.append(u)

        return urls

    @staticmethod
    def _normalize_iso(dt: str) -> str:
        # "Z" -> "+00:00" para fromisoformat
        dt2 = dt.strip().replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(dt2)
        except ValueError:
            return ""
        # a UTC ISO con Z
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

            objs = data if isinstance(data, list) else [data]
            for o in list(objs):
                if isinstance(o, dict) and isinstance(o.get("@graph"), list):
                    objs.extend([x for x in o["@graph"] if isinstance(x, dict)])

            for o in objs:
                if not isinstance(o, dict):
                    continue
                typ = o.get("@type")
                if isinstance(typ, list):
                    typ = typ[0] if typ else None
                if typ in ("NewsArticle", "Article", "ReportageNewsArticle"):
                    dt = o.get("dateModified") or o.get("datePublished")
                    if isinstance(dt, str) and dt.strip():
                        return self._normalize_iso(dt)

        # 2) time[datetime]
        t = soup.select_one("time[datetime]")
        if t and t.get("datetime"):
            return self._normalize_iso(t["datetime"])

        # 3) fallback URL /YYYY/MM/DD/
        m = self.DATE_PATH_RE.search(link)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            return f"{y}-{mo}-{d}T00:00:00Z"

        return ""

    @staticmethod
    def _iso_to_ddmmyyyy(dt_iso: str) -> str:
        try:
            dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
            return dt.strftime("%d-%m-%Y")
        except Exception:
            return ""

    @staticmethod
    def _is_today(dt_iso: str) -> bool:
        try:
            dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        except Exception:
            return False
        return dt.date() == datetime.now().date() - timedelta(days = 1)

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

    def _extract_body(self, soup: BeautifulSoup) -> str:
        # 1) JSON-LD articleBody
        for s in soup.select('script[type="application/ld+json"]'):
            raw = (s.string or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            objs = data if isinstance(data, list) else [data]
            for o in list(objs):
                if isinstance(o, dict) and isinstance(o.get("@graph"), list):
                    objs.extend([x for x in o["@graph"] if isinstance(x, dict)])

            for o in objs:
                if isinstance(o, dict) and (o.get("@type") in ("NewsArticle", "Article", "ReportageNewsArticle")):
                    b = o.get("articleBody")
                    if isinstance(b, str) and b.strip():
                        return self._clean(b)

        # 2) DOM fallback
        article = soup.select_one("article") or soup
        for tag in article.select("script,style,noscript,header,footer,nav,form,aside,figure,iframe"):
            tag.decompose()

        parts = []
        # tu selector original + fallback
        for p in article.select("p.article-text, .article-text p, article p"):
            txt = p.get_text(" ", strip=True)
            if txt and len(txt) >= 30:
                parts.append(txt)

        return self._clean("\n\n".join(parts))

    def crawl(self, max_news: int = 300, sleep_s: float = 0.05) -> list[dict]:
        # 1) links desde secciones
        urls, seen = [], set()
        for sec in self.SECTION_URLS:
            s = self._soup(sec)
            if not s:
                continue
            for u in self._extract_links_from_section(s):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)

        if not urls:
            return []

        # 2) scrap + filtrar solo HOY
        data = []
        for link in urls[:max_news]:
            time.sleep(sleep_s)
            soup = self._soup(link)
            if not soup:
                continue

            dt_iso = self._extract_date_iso(soup, link)
            if not dt_iso or not self._is_today(dt_iso):
                continue

            headline = self._extract_title(soup)
            body = self._extract_body(soup)
            if not headline or not body or len(body) < 300:
                continue

            data.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link.strip())),
                "headline": headline,
                "body": body,
                "link": link,
                "date": self._iso_to_ddmmyyyy(dt_iso),  # dd-mm-aaaa
                "bias": "N",
                "newspaper": self.newspaper,
            })

        return data

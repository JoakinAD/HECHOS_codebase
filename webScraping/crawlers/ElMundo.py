# webScrapping/crawlers/ElMundo.py
import json
import re
import time
import uuid
import html
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawlers.Crawler import Crawler


class ElMundo(Crawler):
    SECTION_URLS = (
        "https://www.elmundo.es/ultimas-noticias.html",
        "https://www.elmundo.es/espana.html",
        "https://www.elmundo.es/internacional.html",
        "https://www.elmundo.es/economia.html",
        "https://www.elmundo.es/elecciones.html",
        "https://www.elmundo.es/salud.html",
        "https://www.elmundo.es/ciencia.html",
        "https://www.elmundo.es/tecnologia.html",
        "https://www.elmundo.es/madrid.html",
        "https://www.elmundo.es/andalucia.html",
        "https://www.elmundo.es/cataluna.html",
        "https://www.elmundo.es/comunidad-valenciana.html",
        "https://www.elmundo.es/galicia.html",
        "https://www.elmundo.es/pais-vasco.html",
    )

    ARTICLE_RE = re.compile(r"/\d{4}/\d{2}/\d{2}/[0-9a-f]{24}\.html$", re.I)
    DENY = ("/opinion/", "/yodona/", "/autor/", "/autores/", "/suscripcion", "/newsletter", "/tags/", "/tag/, #ancla_comentarios")

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "ELMUNDO"
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

    def _is_article_url(self, href: str) -> bool:
        if not href:
            return False
        href = href.strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            return False

        abs_url = urljoin(self.url, href)
        p = urlparse(abs_url)
        if "elmundo.es" not in p.netloc:
            return False

        path = p.path.lower()
        if any(d in path for d in self.DENY):
            return False

        return bool(self.ARTICLE_RE.search(path))

    @staticmethod
    def _clean(text: str) -> str:
        text = html.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _section_links(self, soup: BeautifulSoup) -> list[str]:
        urls, seen = [], set()
        for art in soup.select("article.ue-c-cover-content"):
            if "ue-c-cover-content--is-opinion" in " ".join(art.get("class", [])):
                continue
            a = art.select_one("a.ue-c-cover-content__link[href]") or art.select_one(
                "a.ue-c-cover-content__link-whole-content[href]"
            )
            if not a:
                continue
            href = a.get("href", "").strip()
            if not self._is_article_url(href):
                continue
            u = urljoin(self.url, href)
            if u not in seen:
                seen.add(u)
                urls.append(u)
        return urls

    def _title(self, soup: BeautifulSoup) -> str:
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

    def _body(self, soup: BeautifulSoup) -> str:
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
                if isinstance(o, dict) and (o.get("@type") in ("NewsArticle", "Article", "ReportageNewsArticle")):
                    b = o.get("articleBody")
                    if isinstance(b, str) and b.strip():
                        return self._clean(b)

        # 2) DOM fallback
        root = soup.select_one("div.ue-c-article__body") or soup.select_one("article")
        if not root:
            return ""
        for tag in root.select("script,style,noscript,header,footer,nav,form,aside,figure,iframe"):
            tag.decompose()

        parts = []
        for node in root.select("h2, h3, p, blockquote"):
            txt = node.get_text(" ", strip=True)
            if txt and len(txt) >= 35:
                low = txt.lower()
                if "suscríbete" in low or "hazte suscriptor" in low or "inicia sesión" in low:
                    continue
                parts.append(txt)

        return self._clean("\n\n".join(parts))

    def crawl(self, max_news: int = 300, sleep_s: float = 0.05) -> list[dict]:
        urls, seen = [], set()
        for sec in self.SECTION_URLS:
            s = self._soup(sec)
            if not s:
                continue
            for u in self._section_links(s):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)

        out = []
        for link in urls[:max_news]:
            time.sleep(sleep_s)
            s = self._soup(link)
            if not s:
                continue

            dt_iso = self._extract_publication_date_iso(s)
            if not self._accept_publication_date(dt_iso):
                continue

            title = self._title(s)
            body = self._body(s)
            if not title or not body or len(body) < 300:
                continue

            out.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link.strip())),
                "headline": title,
                "body": body,
                "link": link,
                "date": self._format_publication_date(dt_iso),
                "bias": "N",
                "newspaper": self.newspaper,
            })

        return out

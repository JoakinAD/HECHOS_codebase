# webScrapping/crawlers/ElPais.py
import json
import re
import time
import uuid
import html as html_lib
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawlers.Crawler import Crawler


class ElPais(Crawler):
    SECTION_URLS = (
        "https://elpais.com/ultimas-noticias/",
        "https://elpais.com/actualidad/",
        "https://elpais.com/espana/",
        "https://elpais.com/espana/politica/",
        "https://elpais.com/espana/andalucia/",
        "https://elpais.com/espana/catalunya/",
        "https://elpais.com/espana/comunidad-valenciana/",
        "https://elpais.com/espana/galicia/",
        "https://elpais.com/espana/madrid/",
        "https://elpais.com/espana/pais-vasco/",
        "https://elpais.com/internacional/",
        "https://elpais.com/internacional/europa/",
        "https://elpais.com/economia/",
        "https://elpais.com/economia/vivienda/",
        "https://elpais.com/economia/negocios/",
        "https://elpais.com/noticias/empleo/",
        "https://elpais.com/noticias/empresas/",
        "https://elpais.com/sociedad/",
        "https://elpais.com/educacion/",
        "https://elpais.com/salud-y-bienestar/",
        "https://elpais.com/noticias/igualdad-oportunidades/",
        "https://elpais.com/noticias/sanidad/",
        "https://elpais.com/noticias/sucesos/",
        "https://elpais.com/clima-y-medio-ambiente/",
        "https://elpais.com/ciencia/",
        "https://elpais.com/tecnologia/",
        "https://elpais.com/planeta-futuro/",
        "https://elpais.com/america-futura/",
    )

    # /internacional/2026-02-03/slug.html
    ARTICLE_RE = re.compile(r"/\d{4}-\d{2}-\d{2}/.+\.html$", re.I)
    DATE_IN_URL_RE = re.compile(r"/(\d{4})-(\d{2})-(\d{2})/")

    DENY = ("/opinion/", "/negocios/", "/branded/", "/blogs/", "/autor/", "/autores/", "/tag/", "/tags/")

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "EL PAIS"
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
        if "elpais.com" not in p.netloc:
            return False

        path = p.path.lower()
        if any(d in path for d in self.DENY):
            return False

        return bool(self.ARTICLE_RE.search(path))

    def _date_ddmmyyyy_if_today_from_url(self, url: str) -> str:
        """
        Si la URL contiene /YYYY-MM-DD/ y es HOY (hora local del sistema),
        devuelve dd-mm-aaaa. Si no, "".
        """
        m = self.DATE_IN_URL_RE.search(url)
        if not m:
            return ""
        y, mo, d = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime(int(y), int(mo), int(d)).date()
        except ValueError:
            return ""
        return f"{d}-{mo}-{y}" if dt == datetime.now().date() - timedelta(days = 1) else ""

    def _section_links_today(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        """
        Devuelve lista de (url, fecha_dd-mm-aaaa) SOLO de HOY.
        Además evita:
          - bloques branded/patrocinados (sociedad)
          - bloque Opinión incrustado (aunque no lleve /opinion/)
        """
        # secciones branded/patrocinadas
        excluded_secs = set()
        for sec in soup.select("section.b-bra, section[data-dtm-region^='portada_branded']"):
            excluded_secs.add(id(sec))

        def inside_excluded(node) -> bool:
            cur = node
            while cur is not None:
                if id(cur) in excluded_secs:
                    return True
                cur = getattr(cur, "parent", None)
            return False

        # URLs del bloque "Opinión" detectado por cabecera
        opinion_urls = set()
        for h in soup.select("header.header"):
            if "opinión" in h.get_text(" ", strip=True).lower():
                block = h.find_next_sibling()
                if not block:
                    continue
                for a in block.select("article a[href]"):
                    opinion_urls.add(urljoin(self.url, (a.get("href") or "").strip()))

        out, seen = [], set()
        for art in soup.select("article"):
            if inside_excluded(art):
                continue
            if "c-o" in (art.get("class") or []):  # opinión en listados
                continue

            a = art.select_one("a[href]")
            if not a:
                continue

            href = (a.get("href") or "").strip()
            if not self._is_article_url(href):
                continue

            u = urljoin(self.url, href)
            if u in opinion_urls:
                continue

            fecha = self._date_ddmmyyyy_if_today_from_url(u)
            if not fecha:
                continue

            if u not in seen:
                seen.add(u)
                out.append((u, fecha))

        return out

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
        # JSON-LD primero
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

        # DOM fallback
        root = (
            soup.select_one("div[data-dtm-region='articulo_cuerpo']")
            or soup.select_one("div.a_c")
            or soup.select_one("article")
        )
        if not root:
            return ""

        for tag in root.select("script,style,noscript,header,footer,nav,form,aside,figure,iframe"):
            tag.decompose()

        parts = []
        for node in root.select("h2, h3, p, blockquote"):
            txt = node.get_text(" ", strip=True)
            if txt and len(txt) >= 35:
                low = txt.lower()
                if "suscríbete" in low or "suscrib" in low or "inicia sesión" in low:
                    continue
                parts.append(txt)

        return self._clean("\n\n".join(parts))

    def crawl(self, max_news: int = 250, sleep_s: float = 0.05) -> list[dict]:
        # 1) links SOLO de HOY desde secciones
        pairs, seen = [], set()
        for sec in self.SECTION_URLS:
            s = self._soup(sec)
            if not s:
                continue
            for u, fecha in self._section_links_today(s):
                if u not in seen:
                    seen.add(u)
                    pairs.append((u, fecha))

        if not pairs:
            return []

        # 2) scrap de cada artículo
        data = []
        for link, fecha in pairs[:max_news]:
            time.sleep(sleep_s)
            soup = self._soup(link)
            if not soup:
                continue

            headline = self._title(soup)
            body = self._body(soup)

            if not headline or not body or len(body) < 300:
                continue

            data.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, link.strip())),
                "headline": headline,
                "body": body,
                "link": link,
                "date": fecha,     # dd-mm-aaaa (solo HOY)
                "bias": "N",
                "newspaper": self.newspaper,
            })

        return data

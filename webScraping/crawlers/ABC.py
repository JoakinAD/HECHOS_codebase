# webScrapping/crawlers/ABC.py
import re
import time
import uuid
import json
import html as html_lib
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawlers.Crawler import Crawler


class ABC(Crawler):
    SECTION_URLS = (
        "https://www.abc.es/ultimas-noticias/",
        "https://www.abc.es/espana/",
        "https://www.abc.es/internacional/",
        "https://www.abc.es/economia/",
        "https://www.abc.es/sociedad/",
        "https://www.abc.es/salud/",
        "https://www.abc.es/natural/",
        "https://www.abc.es/ciencia/",
        "https://www.abc.es/tecnologia/",
    )

    ARTICLE_RE = re.compile(r"-\d{14}-(nt|nts|di)\.html$", re.I)

    def __init__(self, url: str):
        super().__init__(url)
        self.newspaper = "ABC"

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
    # HTTP / URLs
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
        if not href:
            return False
        href = href.strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            return False

        abs_url = urljoin(self.url, href)
        parsed = urlparse(abs_url)
        if "abc.es" not in parsed.netloc:
            return False

        p = parsed.path.lower()
        if "/opinion" in p:
            return False

        return bool(self.ARTICLE_RE.search(parsed.path))

    def _extract_section_links(self, soup: BeautifulSoup) -> list[str]:
        urls, seen = [], set()
        for a in soup.select("article a[href], h2 a[href], h3 a[href]"):
            href = (a.get("href") or "").strip()
            if not self._is_valid_article_url(href):
                continue
            u = urljoin(self.url, href)
            if u not in seen:
                seen.add(u)
                urls.append(u)
        return urls

    # ---------------------------
    # Text cleaning
    # ---------------------------
    def _clean_text(self, text: str) -> str:
        text = html_lib.unescape(text).replace("\xa0", " ")
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _strip_paywall_cuts(self, text: str) -> str:
        """
        Quita cortes típicos de paywall en ABC cuando aparecen como token aislado:
          - ' ... ' (con espacios alrededor)
          - ' … ' (unicode)
        No elimina 'algo...' (pegado a palabra), para no romper puntuación normal.
        """
        if not text:
            return text

        t = text
        t = re.sub(r"\s+\.\.\.\s+", " ", t)
        t = re.sub(r"\s+…\s+", " ", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        return t.strip()

    # ---------------------------
    # Extractors
    # ---------------------------
    def _extract_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(" ", strip=True)

        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return og["content"].strip()

        t = soup.find("title")
        return t.get_text(strip=True) if t else ""

    def _extract_body(self, soup: BeautifulSoup) -> str:
        article = soup.select_one("article.voc-d__article") or soup.find("article")
        if not article:
            return ""

        # Limpieza básica de elementos muy ruidosos (lista compacta)
        for tag in article.select(
            "script,style,noscript,header,footer,nav,form,figure,"
            ".voc-advertising,.voc-d-c-related-news,.voc-topics,.voc-most-read,"
            ".voc-ints,.voc-list,.voc-cope-mod,.voc-ob-wrapper,"
            "ev-engagement,ev-em-product-selection,[data-voc-vam],"
            ".voc-save-news,.voc-author__social"
        ):
            tag.decompose()

        parts: list[str] = []
        ps = article.select("p.voc-p") or article.find_all("p")
        for p in ps:
            txt = p.get_text(" ", strip=True)
            if not txt or len(txt) < 35:
                continue
            low = txt.lower()
            if "suscríbete" in low or "súmate" in low or "esta funcionalidad es sólo" in low:
                continue
            parts.append(txt)

        body = self._clean_text("\n\n".join(parts))
        return self._strip_paywall_cuts(body)

    def _is_probably_paywalled(self, body: str) -> bool:
        return len(body) < 300

    # ---------------------------
    # Date helpers
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

        # 2) DOM: time[datetime]
        t = soup.select_one("time[datetime]")
        if t and t.get("datetime"):
            iso = self._normalize_dt(t["datetime"])
            if iso:
                return iso

        # 3) fallback: desde la URL -YYYYMMDDHHMMSS-
        m = re.search(r"-(\d{14})-(nt|nts|di)\.html$", urlparse(link).path, re.I)
        if m:
            ymd = m.group(1)[:8]  # YYYYMMDD
            return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}T00:00:00Z"

        return ""

    @staticmethod
    def _is_today(dt_iso: str) -> bool:
        try:
            d = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        except Exception:
            return False
        return d.date() == datetime.now(timezone.utc).date() - timedelta(days = 1)

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
            art_soup = self._get_soup(link)
            if not art_soup:
                continue

            dt_iso = self._extract_date_iso(art_soup, link)
            if not self._accept_publication_date(dt_iso):
                continue

            headline = self._extract_title(art_soup)
            body = self._extract_body(art_soup)

            # Fallback si el body sale corto (pero sin descartar)
            if self._is_probably_paywalled(body):
                article = art_soup.find("article")
                if article:
                    for tag in article.select("script,style,noscript,aside,figure,footer,header,form,nav"):
                        tag.decompose()
                    ps = [p.get_text(" ", strip=True) for p in article.find_all("p")]
                    ps = [t for t in ps if t and len(t) > 30]
                    alt = self._clean_text("\n\n".join(ps))
                    alt = self._strip_paywall_cuts(alt)
                    if len(alt) > len(body):
                        body = alt

            if not headline or not body or len(body) < 300:
                continue

            data.append(
                {
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

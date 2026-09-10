# webScrapping/crawlers/Crawler.py
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

class Crawler():
    MADRID_TZ = ZoneInfo("Europe/Madrid")

    def __init__(self, url):
        self.url = url
        self.fecha = datetime.now().strftime("%d/%m/%y")
        self.date_stats = Counter()

    @classmethod
    def _parse_publication_date(cls, value):
        """Return an aware Madrid datetime; date-only values are Madrid dates."""
        if not isinstance(value, str) or not (value := value.strip()):
            return None

        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            dt = None

        if dt is None:
            for fmt in ("%d/%m/%Y %H:%M", "%d.%m.%Y %H:%M", "%d/%m/%Y", "%d.%m.%Y"):
                try:
                    dt = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    pass

        if dt is None:
            match = re.fullmatch(r"hace\s+(\d+)\s+hora(?:s)?", value.lower())
            if match:
                return datetime.now(cls.MADRID_TZ) - timedelta(hours=int(match.group(1)))
            try:
                dt = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None

        if dt is None:
            return None

        return dt.replace(tzinfo=cls.MADRID_TZ) if dt.tzinfo is None else dt.astimezone(cls.MADRID_TZ)

    @classmethod
    def _is_today_or_yesterday(cls, value):
        dt = cls._parse_publication_date(value)
        if dt is None:
            return False
        today = datetime.now(cls.MADRID_TZ).date()
        return dt.date() in (today, today - timedelta(days=1))

    def _accept_publication_date(self, value):
        dt = self._parse_publication_date(value)
        if dt is None:
            self.date_stats["undetermined"] += 1
            return False
        today = datetime.now(self.MADRID_TZ).date()
        if dt.date() not in (today, today - timedelta(days=1)):
            self.date_stats["outside_window"] += 1
            return False
        self.date_stats["today" if dt.date() == today else "yesterday"] += 1
        return True

    def _format_publication_date(self, value):
        dt = self._parse_publication_date(value)
        return dt.strftime("%d-%m-%Y") if dt else ""

    def _extract_publication_date_iso(self, soup):
        """Prefer original publication metadata; never substitute modified time."""
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads((script.string or "").strip())
            except (json.JSONDecodeError, TypeError):
                continue
            candidates = data if isinstance(data, list) else [data]
            for obj in list(candidates):
                if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
                    candidates.extend(obj["@graph"])
            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                article_type = obj.get("@type") or obj.get("type")
                if isinstance(article_type, list):
                    article_type = article_type[0] if article_type else None
                if article_type in ("NewsArticle", "Article", "ReportageNewsArticle"):
                    value = obj.get("datePublished")
                    if self._parse_publication_date(value):
                        return value

        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta and self._parse_publication_date(meta.get("content")):
            return meta["content"]
        time_tag = soup.select_one("time[datetime]")
        if time_tag and self._parse_publication_date(time_tag.get("datetime")):
            return time_tag["datetime"]
        timestamp = soup.select_one("time[data-timestamp]")
        if timestamp:
            try:
                seconds = int(timestamp["data-timestamp"])
                if seconds > 10_000_000_000:
                    seconds //= 1000
                return datetime.fromtimestamp(seconds, self.MADRID_TZ).isoformat()
            except (KeyError, TypeError, ValueError, OSError):
                pass
        return ""

    @staticmethod
    def crawl():
        # to be implemented by the child class
        pass

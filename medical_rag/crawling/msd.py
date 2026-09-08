"""MSD Manuals crawler (professional edition).

Discovers topic URLs from the official sitemap (cached locally) and fetches
individual topic articles. The article body lives inside ``<main>`` and is
server-rendered, so plain requests is sufficient.  We honour the site's
``Crawl-delay: 5`` directive.
"""
from __future__ import annotations

import gzip  # noqa: F401  (kept in case MSD serves .gz again)
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDoc
from ..cleaners import clean_text
from ..config import ROOT, CrawlConfig

log = logging.getLogger(__name__)

BASE = "https://www.msdmanuals.com"
SITEMAP_URL = f"{BASE}/sitemaps/professional-topic.xml.gz"
SITEMAP_CACHE = ROOT / "data" / "processed" / "msd_prof_topics_sitemap.xml"


class MSDCrawler(BaseCrawler):
    key = "msd"
    name = "MSD Manuals (Professional)"
    content_kind = "html"
    lang = "en"
    respect_robots = True

    def __init__(self, cfg: CrawlConfig):
        super().__init__(cfg)
        # Honor robots.txt Crawl-delay for this site.
        self.request_delay = max(cfg.delay, 5.0)

    # ------------------------------------------------------------------ data
    def _load_sitemap_urls(self) -> list[str]:
        if SITEMAP_CACHE.exists():
            data = SITEMAP_CACHE.read_text(encoding="utf-8-sig", errors="ignore")
        else:
            try:
                resp = self._get(SITEMAP_URL)
            except Exception as exc:  # noqa: BLE001
                log.warning("MSD sitemap fetch failed: %s", exc)
                return []
            raw = resp.content
            # Served historically as gzip or plain XML with a BOM.
            if raw.startswith(b"\x1f\x8b"):
                data = gzip.decompress(raw).decode("utf-8-sig", errors="ignore")
            else:
                data = raw.decode("utf-8-sig", errors="ignore")
            SITEMAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SITEMAP_CACHE.write_text(data, encoding="utf-8")
        urls = re.findall(r"<loc>([^<]+)</loc>", data)
        topics = [u for u in urls if "/professional/" in u and u.count("/") >= 4]
        return topics

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 8
        topics = self._load_sitemap_urls()
        if not topics:
            return
        # Spread across sections for topical diversity.
        if len(topics) > cap:
            step = len(topics) / cap
            topics = [topics[int(i * step)] for i in range(cap)]

        for url in topics:
            if cap <= 0:
                return
            try:
                html = self._get_html(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", url, exc)
                continue
            title, body, meta = self._extract(html, url)
            if not body or len(body) < 400:
                continue
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata=meta,
                raw=body,
                content_kind="text",
            )

    def _extract(self, html: str, url: str) -> tuple[Optional[str], str, dict]:
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        title = clean_text(h1.get_text(" ", strip=True)) if h1 else None
        main = soup.find("main") or soup.select_one("div.Topic_topic__yIQy4")
        if main is None:
            return title, "", {}
        for junk in main.find_all(["script", "style", "noscript"]):
            junk.decompose()
        for nav in main.find_all("nav"):
            nav.decompose()
        text = main.get_text("\n", strip=True)
        text = re.sub(r"^Full Review:\s*[^\n]*", "", text)
        text = re.sub(r"^View Patient Education\s*", "", text)
        body = clean_text(text)
        path = url.replace(BASE, "").strip("/").split("/")
        meta = {
            "section": path[1] if len(path) > 1 else "",
            "subsection": path[2] if len(path) > 2 else "",
            "topic_slug": path[-1] if path else "",
            "publisher": "Merck Sharp & Dohme",
        }
        return title, body, meta

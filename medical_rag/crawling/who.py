"""WHO (World Health Organization) fact-sheet crawler.

Walks the fact-sheets listing and fetches individual fact sheets, extracting
the main ``#content`` region so we keep the prose and drop global nav.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator, Optional

from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDoc
from ..cleaners import clean_text
from ..config import CrawlConfig

log = logging.getLogger(__name__)

BASE = "https://www.who.int"
INDEX_URL = f"{BASE}/news-room/fact-sheets"

# Fallback slugs in case the listing page yields no links (e.g. JS shell).
FALLBACK_SLUGS = [
    "diabetes",
    "hypertension",
    "cardiovascular-diseases",
    "cancer",
    "asthma",
    "obesity",
    "physical-activity",
    "tobacco",
    "hepatitis",
    "alcohol",
    "influenza",
    "tuberculosis",
]


class WHOCrawler(BaseCrawler):
    key = "who"
    name = "WHO Fact Sheets"
    content_kind = "html"
    lang = "en"

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 12
        slugs: list[str] = []
        try:
            html = self._get_html(INDEX_URL)
            slugs = re.findall(
                r'href="(/news-room/fact-sheets/detail/[^"]+)"', html
            )
            slugs = [s.rsplit("/", 1)[-1] for s in slugs]
            # Deduplicate, preserve order.
            slugs = list(dict.fromkeys(slugs))
        except Exception as exc:  # noqa: BLE001
            log.warning("fact-sheets index failed (%s); using fallback slugs", exc)
        if not slugs:
            slugs = FALLBACK_SLUGS
        if len(slugs) > cap:
            # Take a spread across the list rather than just the head.
            step = len(slugs) / cap
            slugs = [slugs[int(i * step)] for i in range(cap)]

        for slug in slugs:
            if cap <= 0:
                return
            url = f"{BASE}/news-room/fact-sheets/detail/{slug}"
            try:
                page_html = self._get_html(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", url, exc)
                continue
            title, body = self._extract(page_html)
            if not body or len(body) < 300:
                continue
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata={"fact_sheet_slug": slug, "organization": "World Health Organization"},
                raw=body,
                content_kind="text",
            )

    def _extract(self, html: str) -> tuple[Optional[str], str]:
        soup = BeautifulSoup(html, "lxml")
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else None
        main = soup.select_one("#content") or soup.find(attrs={"role": "main"}) or soup.find("main")
        if main is None:
            return title, ""
        for junk in main.find_all(["script", "style", "noscript"]):
            junk.decompose()
        text = main.get_text("\n", strip=True)
        # Drop breadcrumb / credit lines at the top.
        text = re.sub(r"^Home\s*/\s*Newsroom\s*/\s*Fact sheets\s*/\s*Detail\s*/\s*[^\n]*", "", text)
        text = re.sub(r"^WHO/[^\n]*\n?©[^\n]*\n?Credits\s*", "", text)
        return title, clean_text(text)

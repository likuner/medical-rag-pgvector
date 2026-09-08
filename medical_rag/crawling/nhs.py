"""NHS crawler (Health A-Z conditions).

Each condition is represented by a landing page plus a few sub-section pages
(symptoms / diagnosis / treatment / causes / complications / prevention).  We
fetch the landing page and follow the one-level sub-pages, then combine the
prose into a single document so vectorised chunks are substantial.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDoc
from ..cleaners import clean_text
from ..config import CrawlConfig

log = logging.getLogger(__name__)

BASE = "https://www.nhs.uk"
CONDITIONS_URL = f"{BASE}/conditions/"

SUBSECTIONS = [
    "symptoms",
    "treatment",
    "causes",
    "diagnosis",
    "complications",
    "prevention",
]


class NHSCrawler(BaseCrawler):
    key = "nhs"
    name = "NHS Health A-Z"
    content_kind = "html"
    lang = "en"

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 10      # number of conditions
        index_html = self._get_html(CONDITIONS_URL)
        slugs = re.findall(r'href="(/conditions/[^"]+/)"', index_html)
        slugs = list(dict.fromkeys(slugs))
        if not slugs:
            return
        # Spread across the alphabetised list instead of taking only A-conditions.
        if len(slugs) > cap:
            step = len(slugs) / cap
            slugs = [slugs[int(i * step)] for i in range(cap)]

        for base_path in slugs:
            if cap <= 0:
                return
            landing_url = urljoin(BASE, base_path)
            try:
                landing_html = self._get_html(landing_url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", landing_url, exc)
                continue

            title = self._title(landing_html)
            sections: list[str] = []
            main_text, main_url = self._extract_page(landing_html, landing_url)
            if main_text:
                sections.append(main_text)
            sub_pages: list[str] = []
            for sub in SUBSECTIONS:
                sub_url = urljoin(landing_url + "/", sub + "/")
                if sub in main_text.lower() or sub_url in landing_html:
                    try:
                        sub_html = self._get_html(sub_url)
                    except Exception as exc:  # noqa: BLE001
                        continue
                    sub_text, _ = self._extract_page(sub_html, sub_url, heading_only=True)
                    sub_text_full, _ = self._extract_page(sub_html, sub_url)
                    if sub_text_full:
                        sections.append(sub_text_full)
                        sub_pages.append(sub_url)
            body = "\n\n".join(s for s in sections if s).strip()
            if not body or len(body) < 400:
                continue
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=landing_url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata={
                    "condition_slug": base_path.strip("/").split("/")[0],
                    "sub_pages": sub_pages,
                    "organization": "National Health Service (UK)",
                },
                raw=body,
                content_kind="text",
            )

    def _title(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        return clean_text(h1.get_text(" ", strip=True)) if h1 else None

    def _extract_page(
        self, html: str, url: str, *, heading_only: bool = False
    ) -> tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")
        main = soup.select_one("#maincontent") or soup.find("main") or soup.find("article")
        if main is None:
            return "", url
        h1 = soup.find("h1")
        head = clean_text(h1.get_text(" ", strip=True)) if h1 else ""
        for junk in main.find_all(["script", "style", "noscript"]):
            junk.decompose()
        # Drop the breadcrumb that says "Health A-Z ... / your condition".
        for nav in main.find_all("nav"):
            nav.decompose()
        text = main.get_text("\n", strip=True)
        body = clean_text(text)
        if heading_only:
            return (head + "\n" + body[:400]) if body else head, url
        return body, url

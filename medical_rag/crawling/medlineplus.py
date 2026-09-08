"""MedlinePlus crawler (Medical Encyclopedia).

Crawls the alphabetical Medical Encyclopedia and extracts the prose body from
each entry (``#d-article``). This yields substantive disease/condition
overviews (causes, symptoms, treatment) rather than link-hub topic pages.
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

BASE = "https://medlineplus.gov"
INDEX_URL = f"{BASE}/encyclopedia.html"


class MedlinePlusCrawler(BaseCrawler):
    key = "medlineplus"
    name = "MedlinePlus"
    content_kind = "html"
    lang = "en"

    # Letters to walk; keeps the crawl bounded while spanning the alphabet.
    LETTERS = ["A", "C", "D", "H", "S"]

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 12

        # Collect candidate article URLs from the letter pages we selected.
        article_urls: list[tuple[str, str]] = []  # (url, letter)
        for letter in self.LETTERS:
            if len(article_urls) >= cap * 2:
                break
            try:
                html = self._get_html(f"{BASE}/ency/encyclopedia_{letter}.htm")
            except Exception as exc:  # noqa: BLE001
                log.warning("encyclopedia_%s page failed: %s", letter, exc)
                continue
            rels = re.findall(r'href="(article/[^"]+\.htm)"', html)
            for rel in rels:
                article_urls.append((f"{BASE}/ency/{rel}", letter))
                if len(article_urls) >= cap * 2:
                    break

        # De-duplicate by URL, keep first.
        seen: set[str] = set()
        article_urls = [u for u in article_urls if not (u[0] in seen or seen.add(u[0]))]

        for url, letter in article_urls:
            if cap <= 0:
                return
            try:
                html = self._get_html(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", url, exc)
                continue
            title, body = self._extract(html)
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
                doc_metadata={
                    "encyclopedia_letter": letter,
                    "entry_id": url.rsplit("/", 1)[-1].replace(".htm", ""),
                },
                raw=body,
                content_kind="text",
            )

    def _extract(self, html: str) -> tuple[Optional[str], str]:
        soup = BeautifulSoup(html, "lxml")
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else None
        body_el = soup.select_one("#d-article") or soup.find("article")
        if body_el is None:
            return title, ""
        for junk in body_el.find_all(["script", "style", "noscript"]):
            junk.decompose()
        text = body_el.get_text("\n", strip=True)
        text = text.split("URL of this page")[0]
        # Remove "To use the sharing features..." line.
        text = re.sub(r"To use the sharing features on this page, please enable JavaScript\s*", "", text)
        return title, clean_text(text)

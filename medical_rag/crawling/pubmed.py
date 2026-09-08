"""PubMed crawler (NCBI E-utilities).

Uses esearch + efetch to retrieve real abstracts and authoritative metadata
(PMID, DOI, journal, authors, publication date). Content is text (no HTML),
so page_number carries the search-results page and provenance is stored in
``doc_metadata``.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional
from xml.etree import ElementTree as ET

from .base import BaseCrawler, RawDoc
from ..config import CrawlConfig

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# A few broadly relevant clinical query topics (kept modest for politeness).
DEFAULT_QUERIES = [
    "type 2 diabetes mellitus management",
    "hypertension treatment guidelines",
    "heart failure diagnosis and treatment",
    "asthma in adults management",
    "chronic kidney disease progression",
    "acute ischemic stroke thrombolysis",
    "breast cancer screening mammography",
    "opioid use disorder treatment",
    "depression in older adults",
    "lower back pain evidence-based care",
]


def _namespace(tag: str) -> str:
    return f"{{http://www.w3.org/2005/Atom}}{tag}"


class PubMedCrawler(BaseCrawler):
    key = "pubmed"
    name = "PubMed"
    content_kind = "text"
    lang = "en"
    respect_robots = False  # Public research API; not subject to host robots.

    def __init__(self, cfg: CrawlConfig, queries: list[str] | None = None):
        super().__init__(cfg)
        self.queries = queries or DEFAULT_QUERIES

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 15       # total articles to fetch
        per_page = 5                              # results per esearch page
        pages = max(1, min(3, (cap + per_page - 1) // per_page))

        for qidx, query in enumerate(self.queries):
            if cap <= 0:
                break
            ids_and_page: list[tuple[str, int, int]] = []
            for page in range(1, pages + 1):
                retstart = (page - 1) * per_page
                params = {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": str(per_page),
                    "retstart": str(retstart),
                    "sort": "relevance",
                }
                data = self._get_json(f"{EUTILS}/esearch.fcgi", params=params)
                res = data.get("esearchresult", {})
                ids = res.get("idlist", [])
                for i, pmid in enumerate(ids):
                    ids_and_page.append((pmid, page, retstart + i + 1))
                if not ids:
                    break
            if not ids_and_page:
                continue

            # Batch-fetch full records.
            pmids = [p for p, _, _ in ids_and_page]
            for chunk_start in range(0, len(pmids), 50):
                chunk = pmids[chunk_start:chunk_start + 50]
                for article in self._fetch_articles(chunk):
                    pmid = article["pmid"]
                    # ids_and_page holds (pmid, search_page, position).
                    search_page = next(
                        (pp for p, pp, _ in ids_and_page if p == pmid), None
                    )
                    page_num = search_page
                    article["doc_metadata"].update(
                        {
                            "search_term": query,
                            "search_page": search_page,
                            "position_in_results": next(
                                (pos for p, _, pos in ids_and_page if p == pmid), None
                            ),
                        }
                    )
                    cap -= 1
                    if cap < 0:
                        return
                    yield RawDoc(
                        source=self.key,
                        source_name=self.name,
                        source_url=article["url"],
                        title=article["title"],
                        page_number=page_num,
                        lang=self.lang,
                        doc_metadata=article["doc_metadata"],
                        raw=article["text"],
                        content_kind="text",
                    )

    def _fetch_articles(self, pmids: list[str]) -> list[dict[str, Any]]:
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        try:
            xml_text = self._get_text(f"{EUTILS}/efetch.fcgi", params=params)
        except Exception as exc:  # noqa: BLE001
            log.warning("efetch failed for %s: %s", pmids, exc)
            return []
        return self._parse_pubmed_xml(xml_text)

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return articles

        for rec in root.findall(".//PubmedArticle"):
            medline = rec.find("MedlineCitation")
            if medline is None:
                continue
            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else None
            article = medline.find("Article")

            title = ""
            if article is not None:
                t = article.find("ArticleTitle")
                title = _join_text(t) if t is not None else ""

            abstract_parts: list[str] = []
            if article is not None:
                ab = article.find("Abstract")
                if ab is not None:
                    for ast in ab.findall("AbstractText"):
                        label = ast.attrib.get("Label")
                        if label:
                            abstract_parts.append(f"{label}: {_join_text(ast)}")
                        else:
                            abstract_parts.append(_join_text(ast))

            # Journal + pub date.
            journal = ""
            pub_date = ""
            if article is not None:
                j = article.find("Journal")
                if j is not None:
                    jt = j.find("Title")
                    journal = jt.text or "" if jt is not None else ""
                    ji = j.find("JournalIssue")
                    if ji is not None:
                        pd = ji.find("PubDate")
                        if pd is not None:
                            pub_date = _join_text(pd)

            authors: list[str] = []
            if article is not None:
                al = article.find("AuthorList")
                if al is not None:
                    for a in al.findall("Author"):
                        ln = a.find("LastName")
                        fn = a.find("ForeName")
                        if ln is not None and ln.text:
                            authors.append(
                                f"{fn.text + ' ' if fn is not None and fn.text else ''}{ln.text}"
                            )

            doi = ""
            pd = rec.find(".//ArticleId[@IdType='doi']")
            if pd is not None:
                doi = pd.text or ""

            text = (title + "\n\n" + "\n".join(abstract_parts)).strip()
            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "text": text,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    "doc_metadata": {
                        "pmid": pmid,
                        "doi": doi,
                        "journal": journal,
                        "publication_date": pub_date,
                        "authors": authors[:8],
                        "num_authors": len(authors),
                    },
                }
            )
        return articles


def _join_text(el: ET.Element) -> str:
    return " ".join("".join(el.itertext()).split())

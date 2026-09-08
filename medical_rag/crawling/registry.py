"""Registry of available site crawlers."""
from __future__ import annotations

import logging
from typing import Type

from ..config import CrawlConfig
from .base import BaseCrawler
from .medlineplus import MedlinePlusCrawler
from .msd import MSDCrawler
from .nhs import NHSCrawler
from .pubmed import PubMedCrawler
from .who import WHOCrawler

log = logging.getLogger(__name__)

CRAWLERS: dict[str, Type[BaseCrawler]] = {
    "pubmed": PubMedCrawler,
    "medlineplus": MedlinePlusCrawler,
    "who": WHOCrawler,
    "nhs": NHSCrawler,
    "msd": MSDCrawler,
}

ALL_SITES = ["pubmed", "medlineplus", "who", "nhs", "msd"]


def get_crawler(key: str, cfg: CrawlConfig) -> BaseCrawler:
    if key not in CRAWLERS:
        raise KeyError(f"Unknown site '{key}'. Available: {list(CRAWLERS)}")
    return CRAWLERS[key](cfg)


def get_crawlers(sites: list[str] | None, cfg: CrawlConfig) -> list[BaseCrawler]:
    keys = sites or ALL_SITES
    return [get_crawler(k, cfg) for k in keys]

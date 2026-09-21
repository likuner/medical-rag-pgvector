"""Central configuration for the medical RAG pipeline.

All tunable values live here so the rest of the code stays declarative.
Read from environment variables with sensible defaults for a local setup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from ROOT/.env into os.environ (no override).

    Keeps secrets such as the GLM API key out of the repository while still
    exposing them as environment variables. Must run before the dataclass
    field defaults below snapshot the environment.
    """
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


_load_dotenv()


def _ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Database (pgvector)
# --------------------------------------------------------------------------- #
@dataclass
class DBConfig:
    host: str = os.getenv("PGVECTOR_HOST", "localhost")
    port: int = int(os.getenv("PGVECTOR_PORT", "5433"))
    dbname: str = os.getenv("PGVECTOR_DB", "medrag")
    user: str = os.getenv("PGVECTOR_USER", "meduser")
    password: str = os.getenv("PGVECTOR_PASSWORD", "medpass")

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingConfig:
    # Backend to use: "sentence-transformers" | "tfidf" | "glm" | "auto".
    # "auto" prefers the GLM API when GLM_API_KEY is set, then tries the
    # sentence-transformers model, and finally falls back to TF-IDF.
    backend: str = os.getenv("EMBED_BACKEND", "auto")
    # The sentence-transformers model id (HuggingFace hub or local path).
    model_name: str = os.getenv(
        "EMBED_MODEL", "BAAI/bge-small-zh-v1.5"
    )
    # Output dimensionality. MUST match the model for sentence-transformers
    # (bge-small-zh-v1.5 -> 512). The tfidf backend always reduces to this,
    # and the glm backend passes it as the API `dimensions` parameter.
    dim: int = int(os.getenv("EMBED_DIM", "512"))
    # ---- GLM (Zhipu AI) embedding API ------------------------------------ #
    # The API key is a secret: keep it in the environment / a git-ignored
    # .env file, never in committed code.
    glm_api_key: str = os.getenv("GLM_API_KEY", "")
    glm_model: str = os.getenv("GLM_EMBED_MODEL", "embedding-3")
    glm_base_url: str = os.getenv(
        "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/embeddings"
    )
    glm_batch_size: int = int(os.getenv("GLM_EMBED_BATCH", "32"))
    # For TF-IDF fallback: number of features and LSA components.
    tfidf_max_features: int = 200_000
    tfidf_lsa_components: int | None = None  # None -> use `dim`
    # bge-style models want a prefix on queries to signal retrieval intent.
    query_instruction: str = os.getenv(
        "EMBED_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
    )
    normalize: bool = True  # L2-normalise vectors (cosine distance in pgvector)
    # "cpu" is the safe default (avoids intermittent MPS hangs); set "mps"/"cuda"
    # via EMBED_DEVICE if you want acceleration.
    device: str | None = os.getenv("EMBED_DEVICE", "cpu")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
@dataclass
class ChunkConfig:
    size: int = int(os.getenv("CHUNK_SIZE", "600"))        # target chars
    overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))  # overlapping chars
    min_chunk: int = 80  # drop chunks shorter than this after cleaning


# --------------------------------------------------------------------------- #
# Crawling
# --------------------------------------------------------------------------- #
@dataclass
class CrawlConfig:
    # Delay between requests (seconds) - be polite to source servers.
    delay: float = float(os.getenv("CRAWL_DELAY", "1.0"))
    timeout: int = int(os.getenv("CRAWL_TIMEOUT", "25"))
    max_retries: int = 2
    # Per-site cap on number of pages to crawl in one run (0 = unlimited).
    max_pages_per_site: int = int(os.getenv("CRAWL_MAX_PAGES", "15"))
    user_agent: str = (
        "MedicalRAG-Crawler/1.0 (+https://example.org/crawler; "
        "educational research) requests/2.31"
    )
    # Respect robots.txt.
    respect_robots: bool = True


@dataclass
class PipelineConfig:
    db: DBConfig = field(default_factory=DBConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    crawl: CrawlConfig = field(default_factory=CrawlConfig)

    def __post_init__(self) -> None:
        _ensure_dirs()


CONFIG = PipelineConfig()

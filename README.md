# 医学 RAG 爬虫 → pgvector

一个端到端的医学数据流水线：**抓取 5 个医学网站 → 清洗 → 分块 → 向量化 →
写入 PostgreSQL / pgvector**。每一条记录都会保存原始信息（来源、URL、页码、
元数据）、分块文本及其向量，可直接用于语义检索（RAG）。

## 数据来源

| key          | 站点                         | 内容                                       | 方式          |
|--------------|------------------------------|-------------------------------------------|---------------|
| `pubmed`     | PubMed / NCBI E-utilities    | 文献摘要 + 元数据（PMID / DOI / 作者 / 期刊） | REST API      |
| `medlineplus`| MedlinePlus（医学百科）       | 疾病 / 症状 / 治疗概述                      | HTML 抓取     |
| `who`        | 世界卫生组织实况报道           | 公共卫生事实页                             | HTML 抓取     |
| `nhs`        | 英国 NHS Health A-Z          | 疾病着陆页 + 子页面                        | HTML 抓取     |
| `msd`        | 默沙东诊疗手册（专业版）       | 临床专题文章                               | HTML 抓取 + sitemap |

所有爬虫都保持克制：遵守 `robots.txt`、带随机抖动的限速；其中 MSD 单独遵循站点
的 `Crawl-delay: 5` 指令。

## 项目结构

```
run_pipeline.py            CLI 入口（ingest / search）
medical_rag/
  config.py                全局配置（数据库、embedding、分块、抓取）
  db.py                    pgvector 建表 + 批量写入辅助函数
  embedder.py              sentence-transformers（默认）或 TF-IDF+LSA 回退
  cleaners.py              HTML→文本、空白/CJK 规范化、语言识别
  chunking.py              中英文句界感知的重叠分块器
  pipeline.py              抓取 → 清洗 → 分块 → 向量化 → 入库；+ search()
  crawling/
    base.py                BaseCrawler、RawDoc、HTTP/robots 工具
    pubmed.py  medlineplus.py  who.py  nhs.py  msd.py
    registry.py            站点 key → 爬虫类的映射
```

**数据库表结构（pgvector）** —— 两张表，因此原始来源信息、分块文本与向量都保留：

- `documents` —— `source`、`source_name`、`source_url`、`title`、`page_number`、
  `lang`、`doc_metadata`（jsonb）、`raw_text`、`created_at`。
- `chunks` —— `document_id`（外键）、`chunk_index`、`chunk_text`、`char_length`、
  `token_count`、`embedding vector(N)`、`created_at`，并为 `embedding` 建了
  `hnsw (vector_cosine_ops)` 索引以支持快速相似度检索。

## 环境搭建

1. **pgvector 数据库。** 项目需要一个带 `vector` 扩展的 PostgreSQL 实例，最简单
   的方式是用容器：

   ```bash
   docker run -d --name med-pgvector -p 5433:5432 \
     -e POSTGRES_USER=meduser -e POSTGRES_PASSWORD=medpass -e POSTGRES_DB=medrag \
     pgvector/pgvector:pg16      # 该镜像已内置 vector 扩展
   ```

2. **Python 环境。**

   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt           # 包含 sentence-transformers（默认）
   # 若无法安装重型 ML 依赖，管道会自动回退到 TF-IDF + LSA 向量化，
   # 仍可生成可用向量并支持检索。
   ```

   > **模型下载（HuggingFace）。** 如果你的网络无法访问 huggingface.co（国内环境
   > 常见），需要在运行前设置 `HF_ENDPOINT` 镜像；代码在未设置时会默认使用该值：
   >
   > ```bash
   > export HF_ENDPOINT=https://hf-mirror.com
   > ```

3. **配置项（环境变量，以下为默认值）：**

   | 变量 | 默认值 | 含义 |
   |------|--------|------|
   | `PGVECTOR_HOST` | `localhost` | 数据库主机 |
   | `PGVECTOR_PORT` | `5433` | 数据库端口 |
   | `PGVECTOR_DB` | `medrag` | 数据库名 |
   | `PGVECTOR_USER` / `PGVECTOR_PASSWORD` | `meduser` / `medpass` | 账号 / 密码 |
   | `EMBED_BACKEND` | `auto` | `sentence-transformers` / `tfidf` / `auto` |
   | `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | HuggingFace 模型 id |
   | `EMBED_DIM` | `512` | 向量维度（TF-IDF 路径使用） |
   | `CHUNK_SIZE` | `600` | 目标分块字符数 |
   | `CHUNK_OVERLAP` | `120` | 相邻分块重叠字符数 |
   | `CRAWL_MAX_PAGES` | `15` | 每个站点最多抓取文档数 |
   | `CRAWL_DELAY` | `1.0` | 两次请求之间的间隔（秒） |

   > 默认模型 `BAAI/bge-small-zh-v1.5`（512 维）对中文友好。若你的语料以英文为主，
   > 可将 `EMBED_MODEL` 设为 `sentence-transformers/all-MiniLM-L6-v2`（384 维），或
   > 使用多语种模型如 `paraphrase-multilingual-MiniLM-L12-v2`。

## 使用方法

```bash
# 抓取全部 5 个站点并入库（默认 embedding 后端）。
python run_pipeline.py ingest

# 只抓取 WHO，减少页面数：
python run_pipeline.py ingest --sites who --max-pages 8

# 强制使用轻量的 TF-IDF 向量化：
python run_pipeline.py ingest --embed-backend tfidf

# 入库前先删除并重建表结构（干净重建）：
python run_pipeline.py ingest --recreate

# 对库内分块做语义检索：
python run_pipeline.py search --query "treatment for type 2 diabetes" --k 5
中文查询同样支持，例如：python run_pipeline.py search --query "2型糖尿病的治疗" --k 5
```

`search` 命令会返回最匹配的分块，包含分块文本、来源 URL、标题以及余弦
**相似度** 得分。

## 各部分如何工作

- **清洗**（`cleaners.py`）：剔除 script/style/nav 等样板内容，优先取
  `<article>` / `<main>` / `#content` 区域，压缩空白、还原 HTML 实体、去除重复行、
  过滤垃圾字节；用一个轻量的 CJK 比例启发式判定 `zh` / `en`。
- **分块**（`chunking.py`）：按句界切分（支持中文 `。！？`），贪心填充到
  `CHUNK_SIZE`，并将上一块的尾部 `CHUNK_OVERLAP` 带入下一块，舍弃过短块后重新编号。
- **向量化**（`embedder.py`）：默认使用 `sentence-transformers`（语义向量）；当
  ML 依赖不可用时自动回退到 `TF-IDF + LSA`（scikit-learn）。两种后端都会做 L2 归一化
  并补齐到配置的宽度，保证 `vector(N)` 列一致。当前生效的后端与拟合好的模型会被持久化
  到 `data/processed/`，使 `search` 复用同一向量空间。
- **入库**（`db.py` + `pipeline.py`）：每个页面/文章对应一条 `documents`（含来源信息），
  对应多条 `chunks`（各自携带向量）。整个运行**幂等** —— 已入库的 `(source, source_url)`
  在下次运行时会被跳过。

## 备注

- 公开 API（PubMed E-utilities）设 `respect_robots = False`；HTML 站点开启。所有站点
  均使用礼貌的抓取间隔。
- PubMed 会把检索结果**页码**写入 `documents.page_number`；其他爬虫会把额外的来源信息
  （PMID、DOI、作者、所属专题分类等）写入 `documents.doc_metadata`。
- 抓取到的原始页面缓存于 `data/raw/`，MSD 的 sitemap 缓存于 `data/processed/`，
  以避免每次运行都重复下载约 10 MB 的数据。

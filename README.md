# 医学 RAG 爬虫 → pgvector（中文数据源）

一个端到端的医学数据流水线：**抓取 5 个中文医学网站 → 清洗 → 分块 → 向量化 →
写入 PostgreSQL / pgvector**。每一条记录都会保存原始信息（来源、URL、元数据）、
分块文本及其向量，可直接用于语义检索（RAG）。

## 数据来源

| key           | 站点                        | 内容                          | 方式                  |
|---------------|-----------------------------|-------------------------------|-----------------------|
| `who_zh`      | WHO 中文实况报道             | 公共卫生事实页（中文）          | HTML 抓取             |
| `msd_cn`      | 默沙东诊疗手册（中文专业版）   | 临床专题文章                   | HTML 抓取 + sitemap   |
| `a_hospital`  | A+医学百科                   | 疾病/医学概念百科词条           | HTML 抓取             |
| `jk39`        | 39健康疾病百科               | 疾病主页 + 病因/症状/预防/就诊子栏目 | HTML 抓取        |
| `xywy`        | 寻医问药疾病库               | 疾病概述（页面为 GBK 编码）      | HTML 抓取             |

所有爬虫都保持克制：遵守 `robots.txt`、带随机抖动的限速；其中默沙东中文站单独
遵循站点的 `Crawl-delay: 5` 指令。

## 项目结构

```
run_pipeline.py            CLI 入口（ingest / search）
medical_rag/
  config.py                全局配置（数据库、embedding、分块、抓取）
  db.py                    pgvector 建表 + 批量写入辅助函数
  embedder.py              GLM API（推荐）/ sentence-transformers / TF-IDF+LSA 回退
  cleaners.py              HTML→文本、空白/CJK 规范化、语言识别
  chunking.py              中英文句界感知的重叠分块器
  pipeline.py              抓取 → 清洗 → 分块 → 向量化 → 入库；+ search()
  crawling/
    base.py                BaseCrawler、RawDoc、HTTP/robots 工具
    who_zh.py  msd_cn.py  a_hospital.py  jk39.py  xywy.py
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
   | `EMBED_BACKEND` | `auto` | `glm` / `sentence-transformers` / `tfidf` / `auto` |
   | `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | HuggingFace 模型 id |
   | `EMBED_DIM` | `1024` | 向量维度（GLM 与 TF-IDF 路径使用；GLM embedding-3 支持 256–2048） |
   | `GLM_API_KEY` | （无） | 智谱 GLM embedding API 密钥（**必填才能用 glm 后端**） |
   | `GLM_EMBED_MODEL` | `embedding-3` | GLM embedding 模型名 |
   | `GLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/embeddings` | GLM API 端点 |
   | `GLM_EMBED_BATCH` | `32` | 每次 API 请求携带的文本条数 |
   | `CHUNK_SIZE` | `512` | 目标分块 token 数（LlamaIndex SentenceSplitter） |
   | `CHUNK_OVERLAP` | `100` | 相邻分块重叠 token 数 |
   | `CRAWL_MAX_PAGES` | `15` | 每个站点最多抓取文档数 |
   | `CRAWL_DELAY` | `1.0` | 两次请求之间的间隔（秒） |

   > 默认模型 `BAAI/bge-small-zh-v1.5`（512 维）对中文友好。若你的语料以英文为主，
   > 可将 `EMBED_MODEL` 设为 `sentence-transformers/all-MiniLM-L6-v2`（384 维），或
   > 使用多语种模型如 `paraphrase-multilingual-MiniLM-L12-v2`。

4. **GLM embedding（推荐）。** `glm` 后端调用智谱 AI 的 embedding API
   （默认 `embedding-3`，512 维），无需本地 ML 栈。API key 属于敏感信息，
   **不要写进代码提交到仓库**——放在环境变量或项目根目录的 `.env` 文件中
   （`.env` 已被 `.gitignore` 忽略，程序启动时会自动加载）：

   ```bash
   cat > .env <<'EOF'
   GLM_API_KEY=你的密钥
   EOF
   # 或者直接导出环境变量：
   export GLM_API_KEY=你的密钥
   ```

   `EMBED_BACKEND=auto`（默认）在检测到 `GLM_API_KEY` 时会优先使用 GLM，
   否则依次回退到 sentence-transformers、TF-IDF。

## 使用方法

```bash
# 抓取全部 5 个中文站点并入库（有 GLM_API_KEY 时默认走 GLM embedding）。
python run_pipeline.py ingest

# 只抓取 WHO 中文，减少页面数：
python run_pipeline.py ingest --sites who_zh --max-pages 8

# 强制使用轻量的 TF-IDF 向量化：
python run_pipeline.py ingest --embed-backend tfidf

# 入库前先删除并重建表结构（干净重建）：
python run_pipeline.py ingest --recreate

# 对库内分块做语义检索：
python run_pipeline.py search --query "2型糖尿病的治疗" --k 5
英文查询同样支持，例如：python run_pipeline.py search --query "treatment for type 2 diabetes" --k 5
```

`search` 命令会返回最匹配的分块，包含分块文本、来源 URL、标题以及余弦
**相似度** 得分。

## 各部分如何工作

- **清洗**（`cleaners.py`）：剔除 script/style/nav 等样板内容，优先取
  `<article>` / `<main>` / `#content` 区域，压缩空白、还原 HTML 实体、去除重复行、
  过滤垃圾字节；用一个轻量的 CJK 比例启发式判定 `zh` / `en`。
- **分块**（`chunking.py`）：基于 **LlamaIndex `SentenceSplitter`**——中文句界感知
  （`。！？；`）、按 token 预算贪心填充、相邻块携带重叠。自定义中文 tokenizer
  （中文 1 字 = 1 token、英文 1 词 = 1 token）避免了 tiktoken 运行时下载，
  国内网络离线可用。`CHUNK_SIZE=512` / `CHUNK_OVERLAP=100` 单位为该 token 计数，
  过短块（< 80 字符）会被丢弃并重新编号。
- **向量化**（`embedder.py`）：默认优先调用 GLM embedding API（远程语义向量，
  需 `GLM_API_KEY`）；无 key 时回退到 `sentence-transformers`（本地语义向量），
  ML 依赖仍不可用时最终回退到 `TF-IDF + LSA`（scikit-learn）。所有后端都会做 L2
  归一化并补齐到配置的宽度，保证 `vector(N)` 列一致。当前生效的后端、模型与维度会
  被持久化到 `data/processed/`，使 `search` 复用同一向量空间。
- **入库**（`db.py` + `pipeline.py`）：每个页面/文章对应一条 `documents`（含来源信息），
  对应多条 `chunks`（各自携带向量）。整个运行**幂等** —— 已入库的 `(source, source_url)`
  在下次运行时会被跳过。

## 备注

- 5 个站点均为 HTML 抓取且遵守 `robots.txt`，全部使用礼貌的抓取间隔；默沙东中文站
  遵循其 `Crawl-delay: 5`。候选站 wiki8.com（医学百科）因域名失效被排除。
- 各爬虫会把额外的来源信息（实况报道 slug、疾病拼音码/ID、所属科室分类等）写入
  `documents.doc_metadata`；jk39 会把疾病四个子栏目页合并为一篇文档。
- 寻医问药（xywy）页面为 **GBK 编码**，爬虫内部已显式按 GBK 解码。
- 抓取到的原始页面缓存于 `data/raw/`，默沙东中文站的 sitemap 缓存于
  `data/processed/`，以避免每次运行都重复下载约 10 MB 的数据。

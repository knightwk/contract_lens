# ContractLens 学习开发计划（6 周 · 42 天）

> ⚠️ **本文件从第 2 周起已降级为「参考资料库」——执行计划见 [docs/plan_agent_track.md](docs/plan_agent_track.md)。**
>
> **原因**：原计划把 LangGraph 排在 Day 26、Agent 化排在 Day 32，
> 而目标岗位是 **AI Agent 开发**（不是 RAG）。按原顺序要等四周以上才能投递，优先级错配。
>
> **本文件仍然有用的部分**（需要时来这里查细节，不要按它的顺序走）：
> - Day 10：Cross-Encoder 精排
> - Day 13：检索三方案对比实验
> - Day 22-24：评测集设计与确定性断言框架
> - Day 36-38：三个故障实验
> - 依赖清单与版本策略（见文件开头）
>
> 配套文档：[ContractLens 项目规划文档](ContractLens_项目规划文档.md)
> **学习方式**：先代码、后原理。主轨写代码，副轨（每天最后 20~30 分钟）补知识点和面试题。
> **时间预算**：工作日 3~4h + 周末 8~10h ≈ 30h/周，6 周约 180h。
> **硬件前提**：无需 GPU。embedding 与 reranker 模型本地 CPU 部署。

---

## 一、环境准备与版本清单

### 1.1 版本选择原则

**兼容性与生态成熟度优先，不盲目追新。** 本项目有三处刻意不用最新版本：

| 包 | 最新版本 | 本项目采用 | 为什么不用最新 |
| --- | --- | --- | --- |
| `pymilvus` | 3.0.2 | **2.6.17** | 官方兼容表明确要求 Milvus 2.6.\* 服务端配 pymilvus **2.6.X**。跨大版本用 SDK 是生产事故的常见来源 |
| `transformers` | 5.18.0 | **4.x 稳定线** | 5.x 是刚发布的大版本，生态适配仍在推进。生产环境优先选择被广泛验证的版本线 |
| `sentence-transformers` | 6.1.0 | **5.7.0** | 6.x 依赖 transformers 5.x；5.7.0 + transformers 4.x 的组合更成熟 |

其余原则：
- **关键兼容性包用 `==` 精确锁定**（pymilvus、mcp、langfuse、langgraph、sentence-transformers）
- **其余用 `>=x,<y` 约束**，避免被意外升级到大版本
- **安装成功后执行 `pip freeze > requirements.lock.txt`** 生成你自己的锁文件——这才是可复现的版本记录

### 1.2 `requirements.txt`

```txt
# ============================================================
# ContractLens 依赖清单
# 版本策略：兼容性与生态成熟度优先，不盲目追最新
# 安装后请执行：pip freeze > requirements.lock.txt
# Python 3.12
# ============================================================

# ---------- Web 与服务 ----------
fastapi>=0.115,<1.0
uvicorn[standard]>=0.34,<1.0
pydantic==2.11.4
pydantic-settings>=2.7,<3.0
python-dotenv>=1.0,<2.0
loguru==0.7.3

# ---------- LLM 调用 ----------
openai==1.99.5
httpx==0.28.1

# ---------- 本地模型推理（CPU）----------
# torch 2.5+ 是 transformers 4.x 的运行时要求
# Windows 下从 PyPI 安装即为 CPU 版；Linux 下需指定 CPU 源（见 1.4）
torch>=2.5,<3.0
transformers>=4.45,<5.0
sentence-transformers==5.7.0

# ---------- 向量库 ----------
# 必须与 Milvus 服务端 2.6.x 匹配
pymilvus==2.6.17

# ---------- Agent 编排 ----------
langgraph==1.2.12

# ---------- MCP ----------
# 必须 2.x：2026-07-28 无状态规范只在 v2 中实现
mcp[cli]==2.2.0

# ---------- 评测 ----------
ragas==0.4.3

# ---------- 可观测性 ----------
# 必须 4.x：v3 SDK 在 v4 服务端上已弃用
langfuse==4.16.0

# ---------- 缓存与幂等 ----------
redis>=5.0,<7.0

# ---------- 文档解析 ----------
pypdf>=5.0,<7.0
python-docx>=1.1,<2.0

# ---------- 工具 ----------
tenacity==9.1.2
tiktoken==0.9.0
numpy==2.2.6

# ---------- UI ----------
streamlit>=1.40,<2.0

# ---------- 开发 ----------
pytest==8.4.1
pytest-asyncio==0.26.0
ruff==0.12.8
```

**从旧清单中移除的两个包**（本项目不使用）：
- `litellm` —— 多模型网关，本项目直接用 `openai` SDK 调 OpenAI 兼容接口
- `lightrag-hku` —— 图检索，本项目不做（见项目规划 §21 决策记录）

### 1.3 版本速查表

| 包 | 锁定版本 | 说明 |
| --- | --- | --- |
| Python | 3.12 | |
| fastapi | ≥0.115 | |
| pydantic | 2.11.4 | |
| **torch** | **2.5 ~ 2.x（CPU）** | 本次核查时最新为 2.14.1 |
| **transformers** | **4.45 ~ 4.x** | 本次核查时最新为 5.18.0，刻意不用 |
| **sentence-transformers** | **5.7.0** | 本次核查时最新为 6.1.0，刻意不用 |
| **pymilvus** | **2.6.17** | 本次核查时最新为 3.0.2，**不可用**（服务端是 2.6.x） |
| **langgraph** | **1.2.12** | 注意：1.2.3 曾因回归被 yank |
| **mcp[cli]** | **2.2.0** | 必须 2.x |
| **langfuse** | **4.16.0** | 必须 4.x |
| ragas | 0.4.3 | |
| openai | 1.99.5 | |
| tenacity | 9.1.2 | |
| tiktoken | 0.9.0 | |
| numpy | 2.2.6 | |
| loguru | 0.7.3 | |
| redis | 5.x ~ 6.x | |
| pypdf / python-docx | 5.x / 1.1+ | |
| streamlit | 1.40+ | 由我实现 |
| pytest / pytest-asyncio / ruff | 8.4.1 / 0.26.0 / 0.12.8 | |

> 上表中的版本号是**本次核查时**各包的状态。执行时如遇解析冲突，以 pip 的解析结果为准，然后 `pip freeze` 锁定。

### 1.4 安装步骤

```bash
# 1. 虚拟环境（你已有 .venv，Python 3.12）
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Windows：直接从 PyPI 装即可，默认是 CPU 版
pip install -r requirements.txt

#    Linux：指定 CPU 源，避免拉入体积巨大的 CUDA 依赖
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt

# 3. 生成锁文件（重要）
pip freeze > requirements.lock.txt
```

### 1.5 本地模型获取

**国内网络先设镜像**：

```bash
# .env
HF_ENDPOINT=https://hf-mirror.com
```

**预下载脚本** `scripts/download_models.py`：

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download

MODELS = [
    "BAAI/bge-base-zh-v1.5",     # embedding  ~400 MB
    "BAAI/bge-reranker-base",    # reranker   ~1.1 GB
]

for repo in MODELS:
    name = repo.split("/")[-1]
    snapshot_download(repo_id=repo, local_dir=f"./models/{name}")
    print(f"done: {name}")
```

```bash
python scripts/download_models.py
```

下载后模型在 `./models/` 下，之后加载走本地路径，**完全离线**：

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
embedder = SentenceTransformer("./models/bge-base-zh-v1.5")
reranker = CrossEncoder("./models/bge-reranker-base")
```

> ⚠️ `models/` 要写进 `.gitignore`（约 1.5 GB），Docker 用挂载方式引入。

### 1.6 CPU 推理的三个关键点（W1 Day 2 就要落地）

这三条是本地部署最容易踩的坑，也是面试可讲的加分点。

| 问题 | 现象 | 解法 |
| --- | --- | --- |
| **推理是阻塞调用** | 在 `async def` 里直接调用会**卡死事件循环**，一个请求慢，所有并发请求一起慢 | `await asyncio.to_thread(...)` |
| **首次请求要加载模型** | 第一次调用几十秒 | FastAPI `lifespan` 里**预热加载** |
| **torch 默认吃满所有核心** | 多请求并发时线程互相抢占，总吞吐反而下降 | `torch.set_num_threads(N)`，按并发留余量 |

精排延迟的额外优化：候选池 Top-20 → **Top-10**、文档截断 `max_length=512 → 256`、精排结果缓存。

---

## 二、编码约定（Day 1 必读，5 条）

这 5 条决定了**以后能不能低成本升级**。现在遵守 = 零成本；以后补 = 重构。

| # | 约定 | 为什么 |
| --- | --- | --- |
| **1** | **能力即函数**：每个能力（解析/风险识别/复核/汇总）写成独立可调用的函数或类，**输入输出都是 Pydantic 模型** | 以后把能力包成 Agent 只是加一层壳；若节点间靠读写 State 临时字段互相通信，就得先拆耦合 |
| **2** | **State 按能力分字段**：`extraction_result` / `retrieval_result` / `risk_result` / `review_result` / `report`，不要一个大的 `answer` | 每个字段天然成为对应 Agent 的输出契约 |
| **3** | **每个能力声明工具白名单**（`agent/tool_policy.py`） | 一份工作两个用途：① 权限控制；② 多 Agent 时每个 Agent 的可用工具集 |
| **4** | **目录按能力分组**（`agent/capabilities/extraction.py` 等） | 拆分时只是"把这几组包成子图"，不用重新组织代码 |
| **5** | **所有引用必须带知识库条目 ID** | "引用可回溯"和"不编造法条"的技术基础，不能事后补 |

**第 6 条（本地模型相关）**：

| # | 约定 | 为什么 |
| --- | --- | --- |
| **6** | **模型访问统一走 `core/model_runtime.py`**，业务代码不直接 `import torch` 或实例化模型 | ① 集中管理预热、线程池、线程数；② 切换本地/API 时只改一处 |

---

## 三、6 周总览

| 周 | 主题 | 里程碑 Tag | 核心产出 |
| --- | --- | --- | --- |
| **W1** | 闭环打通 + 本地模型就位 | `v0.1.0-rag-loop` | 上传合同 → 带引用的回答 + UI + 本地模型跑通 |
| **W2** | 生产级检索 + 延迟优化 | `v0.2.0-retrieval` | 混合检索 + 精排 + 两组对比数据 + 双知识源 |
| **W3** | 领域层 + 边界 | `v0.3.0-domain` | 结构化报告 + 幂等 + 权限 + 边界测试 |
| **W4** | 评测体系 + LangGraph + 模型对比 | `v0.4.0-eval` | 60 条评测集 + 确定性断言 + 状态机 + **本地/API 对比实验** |
| **W5** | MCP + 双 Agent | `v0.5.0-agent-mcp` | MCP Server + 风险识别 Agent + 复核 Agent + HITL |
| **W6** | 可靠性 + 交付 | `v1.0.0` | 三个故障实验 + 一键部署 + README + 简历 |

---

# 第一周｜闭环打通 + 本地模型就位

**本周目标**：不管优雅不优雅，先让"上传合同 → 提问 → 带引用的回答"完整跑起来；本地模型跑通并把三条 CPU 推理规则落地。

## Day 1｜项目初始化 + 依赖安装 + 编码约定

**代码**
- 建目录结构（照项目规划 §9）
- 安装依赖：`pip install -r requirements.txt` → `pip freeze > requirements.lock.txt`
- `config.py`（pydantic-settings，含 `EMBEDDING_PROVIDER` / `RERANK_PROVIDER` 等配置）、`logger.py`、`main.py`、`api/router.py`
- `.env.example`、`.gitignore`（记得忽略 `models/`）
- **把 6 条编码约定写进 `docs/conventions.md`**
- Docker 起 Milvus 2.6.x 与 Langfuse

**知识点**
- 项目整体架构与分层理由（接入/处理/本地模型/检索/领域/编排/工具/防护/评测）
- RAG 是什么：为什么需要检索，而不是把文档全塞进上下文
- FastAPI 的 lifespan：启动时初始化资源，关闭时释放
- **版本锁定策略**：为什么要区分 `requirements.txt` 与 `requirements.lock.txt`

**面试题**
- Q：为什么用 FastAPI？
- A：原生异步，适合 LLM 调用的 IO 密集场景；Pydantic 自动做请求校验；原生生成 OpenAPI 文档，与 LLM 返回的 JSON 结构天然匹配。另外我的本地模型推理是 CPU 阻塞操作，FastAPI 的线程池机制配合 `asyncio.to_thread` 能把它隔离出去，不阻塞事件循环。
- Q：依赖为什么锁版本？
- A：生产环境要可复现。我的 `requirements.txt` 里关键兼容性包用 `==` 锁定，其余用区间约束避免大版本意外升级，安装后 `pip freeze` 生成锁文件作为最终记录。比如 pymilvus 我特意用 2.6.17 而不是最新的 3.0.2，因为服务端是 Milvus 2.6.x，官方兼容表要求 SDK 同大版本。

**自检**
- `pip check` 无冲突；`pip freeze` 生成锁文件
- `uvicorn app.main:app --reload` 启动成功，`/docs` 可打开
- Milvus 与 Langfuse 容器健康

---

## Day 2｜本地模型下载与跑通（本周关键）

**代码**
- `scripts/download_models.py`，下载 `bge-base-zh-v1.5` 和 `bge-reranker-base`
- **`core/model_runtime.py`**（编码约定 6）：
  - 模型加载与缓存（单例）
  - `warmup()`：预热
  - `torch.set_num_threads(N)`
  - `async def encode_async(...)` / `async def rerank_async(...)`：内部用 `asyncio.to_thread`
- `core/embedding.py`：本地实现 + API 实现，同一接口
- `core/reranker.py`：同上
- 写冒烟测试脚本：两句相似文本 vs 无关文本比相似度；一对相关 query-doc 的 rerank 分数明显高于无关对

**知识点**
- Embedding 是把文本映射到高维向量空间的语义表示
- 模型规模与延迟的关系：参数量 → 权重体积 → CPU 前向计算时间
- **Bi-Encoder vs Cross-Encoder**：前者 query 和 doc 分别编码（可预计算、快）；后者拼接后一起前向（无法预计算、慢但准）——**这正是精排成为本地部署瓶颈的原因**
- 相似度度量：余弦相似度 / 内积 / 欧氏距离；归一化的意义
- 为什么 `.to_thread()` 是必须的：Python 的 GIL 与 CPU 密集任务的关系

**面试题**
- Q：为什么选 `bge-base-zh-v1.5` 而不是效果更好的 `bge-m3`？
- A：三个原因。一是 CPU 延迟——`bge-m3` 是 5.7 亿参数，`bge-base` 是 1 亿，前向计算差了 5 倍以上；二是合同场景领域收敛，检索目标是条款和法条，用词规范，不需要 `bge-m3` 的多语言和长文本能力；三是权重体积小 5 倍，内存压力低。**但我把模型做成了配置项**，如果实测效果不够，改一行配置就能切到 `bge-m3` 或者 API。
- Q：本地推理是阻塞的，你怎么处理？
- A：在 `async def` 接口里直接调用模型会卡死事件循环——一个请求在算，其他所有请求都得等。我用 `asyncio.to_thread` 把推理丢到线程池。另外还要设 `torch.set_num_threads`，因为 torch 默认吃满所有核心，多请求并发时线程互相抢占，总吞吐反而下降。

**自检**
- 两个模型都能从 `./models/` 离线加载
- 冒烟测试通过（相似文本分数明显更高）
- 记下**首次加载耗时**和**单次编码耗时**（后面优化要用作基线）

---

## Day 3｜文档解析 + 条款级切分

**代码**
- `core/parser.py`：PDF / Word / Markdown 解析
- `core/chunker.py`：**按合同条款结构切分**
- 记录第一版参数：chunk≈300~500 字，overlap≈50

**知识点**
- 文档解析的真实难点：PDF 版式混乱、跨页表格、扫描件需 OCR
- 切分策略对比：固定长度 / 递归字符 / 语义切分 / **条款级切分**
- `chunk_size` 与 `overlap` 的权衡：太小丢上下文，太大引噪声
- 中文 token 估算（约 1 token ≈ 1.5~2 个汉字）
- **切分粒度与本地模型延迟的关系**：块越少，索引越快；块越大，编码越慢

**面试题**
- Q：chunk_size 为什么这么设？
- A：合同的一个完整条款是审查的最小单位，按条款切能保证"一条风险对应一段完整原文"，引用才可回溯。overlap 保证跨块语义连续。**具体数值是实测调的**：第一版按字数硬切，一段几千字，检索出一堆无关内容；改条款级后召回明显改善。

**自检**
- 一份真实合同能切成 N 块，每块基本是完整条款
- 把切分结果打印出来看过一遍（不能省）

---

## Day 4｜Milvus schema（稠密 + BM25）+ 写入 + 检索

**代码**
- `core/vector_store.py`：建 collection
  - 稠密向量字段（768 维，对应 bge-base）
  - **文本字段 + BM25 函数配置**（Milvus 2.6 内置全文检索）
  - 标量字段（用于元数据过滤和权限，后面要用）
- 索引配置
- 写入（批量编码，注意批大小）+ 基础检索跑通

**知识点**
- 向量库索引类型：IVF_FLAT（需训练、内存省）、HNSW（查询快、内存占用高）
- **Milvus 内置 BM25**：只需文本字段 + 分析器配置，不需要额外模型产出稀疏向量
- 稀疏检索与稠密检索的互补性：前者擅长精确匹配（"第 12.3 条"），后者擅长语义匹配（"甲方"≈"委托方"）
- 批量编码与批大小的权衡（CPU 上一次编码太多会吃满内存）

**面试题**
- Q：稀疏检索为什么用 Milvus 内置 BM25，而不是用模型产出稀疏向量？
- A：因为我的需求是"法条编号精确匹配"，这正好是 BM25 的设计目标。用内置 BM25 有三个好处：少加载一个模型（省内存、省延迟）、少维护一份 tokenizer 和稀疏向量的对齐逻辑、BM25 的 k1/b 参数语义清晰可直接调。用模型产稀疏向量（比如 bge-m3 的 lexical 权重）效果上限可能更高，但对我这个场景是过度设计。

**自检**
- 插入若干段落，稠密检索能返回相关结果并带元数据
- BM25 检索路径能跑通（下一周正式用）

---

## Day 5｜LLM 封装 + 最小 RAG 链路

**代码**
- `core/llm.py`：LLM 调用封装（**流式** + 重试骨架 + tiktoken 计数）
- `core/retriever.py`：最朴素的向量 Top-K
- Prompt 组装：检索结果 + 用户问题 → 要求**带原文引用**

**知识点**
- LLM API 关键参数：`temperature`（0~0.3 适合抽取和审查类任务）、`top_p`、`max_tokens`
- 流式输出（SSE）的原理与价值
- **Prompt 结构五要素**：角色 / 任务 / 约束 / 输出格式 / 示例
- 为什么第一版就要求"必须带引用"

**面试题**
- Q：审查类任务的 temperature 怎么设？
- A：设低（0~0.3）。风险审查和要素抽取出错代价高，需要确定性输出。同一流程里我用统一低温，避免不确定性叠加。

**自检**
- 问"甲方是谁"，能得到**基于原文**且带引用的回答

---

## Day 6｜FastAPI 接口 + 引用返回 + 线程池调度

**代码**
- `api/v1/documents.py`：`POST /documents/upload`
- `api/v1/query.py`：`POST /query`
- 响应模型带引用（`original_text` + `clause_location`）
- `api/v1/health.py`
- **确认所有模型调用都经过 `await ..._async()`**（不在 `async def` 里直接调同步推理）

**知识点**
- 异步接口设计与 `async`/`await` 的正确使用（不要在 async 里跑阻塞 IO 或 CPU 密集任务）
- 请求/响应的 Pydantic 建模
- 引用返回的数据结构设计
- 用 `time.perf_counter()` 给关键路径打点，为后续优化建立基线

**面试题**
- Q：为什么要求答案必须带引用？
- A：法务场景下"答案对不对"没法一眼判断，必须让用户能点开原文核对。**引用是这个场景的信任基础**，也是后面"不编造法条"约束的同一套思路。没有引用的答案在这个场景里等于没有价值。

**自检**
- `curl` 上传合同 + 提问，返回带引用的答案
- 记录端到端耗时，并拆出 embedding / 检索 / 精排 / LLM 各段占比

---

## Day 7｜Streamlit UI + 周复盘

**代码**
- `ui/streamlit_app.py`（**我实现，你负责对接与联调**）
- 页面：上传合同 → 提问 → 显示答案与引用卡片

**知识点**
- 可演示性对面试的价值：面试官更愿意看一个能点的东西
- 前后端联调的基本方法

**本周理论文档**（选读）
- RAG 原始论文：<https://arxiv.org/abs/2005.11401>
- Milvus 快速入门：<https://milvus.io/docs/quickstart.md>
- Cross-Encoder 说明：<https://www.sbert.net/examples/applications/cross-encoder/README.html>
- bge-base-zh-v1.5 模型卡片：<https://huggingface.co/BAAI/bge-base-zh-v1.5>

**本周自检**
- [ ] 依赖安装无冲突，锁文件已生成
- [ ] 两个本地模型能离线加载
- [ ] 6 条编码约定写进 `docs/conventions.md` 并体现在代码里
- [ ] 能上传合同并得到带引用的回答
- [ ] 有各段耗时基线数据
- [ ] 代码已推 GitHub

**里程碑**：`git tag v0.1.0-rag-loop`

---

# 第二周｜生产级检索 + 延迟优化

**本周目标**：把检索做扎实，并把本地精排的延迟压下来。产出两组对比数据——这是简历和面试的素材。

## Day 8｜Milvus BM25 稀疏检索接入

**代码**
- `vector_store` 增加 BM25 检索路径
- `retriever` 支持"稠密 / 稀疏"两种模式切换

**知识点**
- BM25 原理：基于词频（TF）和逆文档频率（IDF）
- **两个核心参数**：
  - `k1`（默认 1.2）：词频饱和调节器。越小越关注"是否出现"，越大越奖励重复出现
  - `b`（默认 0.75）：文档长度归一化。b=0 不归一化，b=1 完全归一化
- **为什么向量检索对"第 12.3 条"召回差**：分词把数字和符号切碎了
- 分析器/分词器配置对中文 BM25 效果的影响

**面试题**
- Q：BM25 的 k1 和 b 是什么？你项目里怎么调的？
- A：k1 控制词频饱和速度，b 控制长度归一化程度。合同场景中法条编号需要精确匹配，我调低了 k1（0.5~1.0）让评分更快饱和、更关注"是否出现"；同时调低 b（0.3~0.5）避免长合同条款因长度被过度惩罚。**这是实测调的，不是照默认值用的**。

**自检**
- "第 12.3 条"这类查询，稀疏检索能召回，稠密检索召回差（**把这个对比记下来**）

---

## Day 9｜RRF 融合

**代码**
- `retriever` 实现 RRF：`score(d) = Σ 1/(k + rank(d))`，k 默认 60
- 支持"稠密 Top-20 + BM25 Top-20 → RRF → Top-20"

**知识点**
- RRF 是**无需调参**的融合排序方法，只依赖排名不依赖分数
- 为什么它比"加权求和"稳健：不同检索器的分数量纲不可比，直接加权需要调归一化，RRF 绕开了这个问题
- k 值（默认 60）的作用：k 越大，排名差异的影响越小

**面试题**
- Q：为什么用 RRF 而不是加权求和？
- A：稠密检索输出余弦相似度，BM25 输出无上界分值，两者量纲不可比。加权求和必须先归一化，而归一化本身又要调参。RRF 只用排名，天然规避量纲问题，效果稳定，适合作为默认融合方案。

**自检**
- 混合检索结果优于任一单路（**记录数据**）

---

## Day 10｜Cross-Encoder 精排 + 延迟优化（本周关键）

**代码**
- `retriever` 集成 `bge-reranker-base`：Top-20 → rerank → Top-5（**宽进严出**）
- **延迟优化四件套**：
  1. 候选池 Top-20 → **Top-10**
  2. 文档截断 `max_length=512 → 256`
  3. 精排结果缓存（key = query 归一化 + 候选集合哈希）
  4. 确认走 `asyncio.to_thread`，并调好 `torch.set_num_threads`
- 记录优化前后的精排延迟（P50/P95）——**这组数据要写进简历**

**知识点**
- Cross-Encoder 无法预计算，所以延迟随候选数线性增长 → **候选裁剪是最有效的优化**
- 精排的阈值设置：低于阈值直接丢弃，而不是硬取 Top-5
- 缓存设计：什么 key 能命中、什么情况会失效
- 为什么"宽进严出"（先大池召回再精排）比"小池精排"效果好
- 并发下的线程数配置：为什么不能吃满所有核心

**面试题**
- Q：为什么需要 Reranker？Embedding 不够吗？
- A：Embedding 是 Bi-Encoder，query 和 doc 独立编码，缺少交互。混合检索扩大了召回池，但 BM25 会引入"词面命中但语义无关"的段落。Reranker 是 Cross-Encoder，能压掉这部分噪声。**实测结论是：只开混合不开精排，通常是负优化**——召回变多了但精度掉了，Faithfulness 反而下降。
- Q：本地精排慢，你怎么优化？
- A：先定位瓶颈——`bge-reranker-base` 是 2.78 亿参数，Cross-Encoder 又要对每个候选做完整前向，20 个候选就是 20 次前向。四步优化：候选池从 20 收到 10（延迟直接减半）、文档截断到 256 token、精排结果做缓存、确保推理走 `asyncio.to_thread` 不阻塞事件循环。另外要限制 `torch` 的线程数，否则并发请求会互相抢占核心，总吞吐反而下降。

**自检**
- Top-5 里的无关段落明显减少（**记录数据**）
- **精排 P95 延迟有优化前后的对比数字**

---

## Day 11｜法条库（知识源 1）

**代码**
- `domain/legal/statutes.py`：法条数据结构（**每条必须有唯一 ID**）
- 建独立 collection，写入 + 检索跑通

**知识点**
- 知识库构建流程：收集 → 清洗 → 结构化 → 切分 → 索引
- **条目 ID 的作用**：这是"引用可回溯"的技术基础。引用必须指向 ID，而不是引用一段文本——引用文本无法验证，引用 ID 可以校验存在性
- 数据来源的权威性要求

**面试题**
- Q：为什么引用要带 ID，而不是直接引用原文？
- A：带 ID 才能**程序化校验**这条引用是否真实存在。如果只引用文本，模型编造一段"看起来很专业"的法条，系统无法判断真伪。有了 ID，检索阶段就能验证，验证失败直接标记"依据不足"——这是杜绝编造法条的关键机制。

**自检**
- 能检索到法条并返回条目 ID

---

## Day 12｜判例库（知识源 2）

**代码**
- `domain/legal/precedents.py`：结构与法条库类似
- 检索结果能区分来源（法条 / 判例）

**知识点**
- 异构知识源的检索与归并：不同来源结构不同，统一成"带来源标注的证据"
- **双知识源的额外价值**：只有一个知识源时，面试官会怀疑引用是硬编码的；跨两个异构源都能回溯，说服力完全不同

**面试题**
- Q：为什么要两个知识源？
- A：法条回答"这违反了什么规定"，判例回答"这类问题实践中怎么判的"。对风险审查来说只有法条依据不够，实务上还需要判例支撑。技术上它也验证了检索与引用机制能处理异构来源，而不是只对一个库特调。

**自检**
- 法条和判例能分别检索并按来源标注

---

## Day 13｜对比实验（本周核心产出）

**代码**
- 跑**检索方案对比**：A 单向量 / B 混合 / C 混合+精排
- 跑**切分策略对比**：固定长度 vs 条款级
- 固定同一批查询（含"法条编号""专有名词""口语化描述"三类），输出对比表

**知识点**
- **消融实验（ablation）设计**：一次只改一个变量，其余保持一致，才能归因
- 检索指标：Recall@k、MRR、Precision@k
- 如何设计公平对比（同一查询集、同一 top-k、同一评判标准）

**面试题**
- Q：你怎么证明混合检索真的更好？
- A：我固定了同一批查询，覆盖三类问法，在同一 top-k 下做消融：只改检索方式，其余完全一致。结果是在法条编号类查询上混合检索召回显著优于单向量，但引入无关段落后 Faithfulness 下降，加入精排后才同时改善——**这也是我得出"混合必须配精排"结论的依据**。

**自检**
- **产出两组对比数据表**（进 README 和简历）

---

## Day 14｜评测脚手架 + 周复盘

**代码**
- `evaluation/business_metrics.py` 骨架：关键条款召回率
- 把 W2 的对比数据整理成 markdown

**知识点**
- 业务指标 vs 学术指标：为什么"关键条款召回率"比 Recall@k 更贴近业务
- 评测脚手架的作用：让后续每次改动都能快速回归

**本周理论文档**（选读）
- BM25 参数调优：<https://github.com/asgard-ai-platform/skills/blob/main/algo-ecom-bm25/references/parameter-tuning.md>
- RRF 原始论文：<https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
- bge-reranker-base 模型卡片：<https://huggingface.co/BAAI/bge-reranker-base>

**本周自检**
- [ ] 两组对比数据表完成
- [ ] 法条库 + 判例库都能检索并返回 ID
- [ ] 精排延迟有优化前后对比数字
- [ ] 能说清"混合必须配精排"的实测依据

**里程碑**：`git tag v0.2.0-retrieval`

---

# 第三周｜领域层 + 边界

**本周目标**：做出结构化报告，并把"不做什么"变成代码。**本周是项目从"能跑"变成"可靠"的转折点。**

## Day 15｜结构化输出 + 要素抽取

**代码**
- `domain/schemas.py`：要素抽取的 Pydantic 模型
- `agent/capabilities/extraction.py`：**独立可调用的抽取函数**（编码约定 1）
- 抽取失败/字段缺失的兜底：缺失输出"未载明"，**不推断**

**知识点**
- 结构化输出（JSON mode / function calling）如何约束模型输出
- Pydantic 做输出校验与重试
- **"不臆测未载明内容"**：抽取任务里最常见的错误是模型为了填满字段而编造

**面试题**
- Q：怎么保证模型输出的格式正确？
- A：两层。第一层用结构化输出能力约束格式；第二层用 Pydantic 校验，失败就把错误信息回给模型重试一次，再失败则降级为部分结果并标记缺失字段。**关键是宁可标"未载明"，也不要让模型猜一个值填进去**——法务场景里编造金额比留空危险得多。

**自检**
- 能抽取主体、金额、期限、管辖、条款定位；缺失字段标"未载明"

---

## Day 16｜风险规则库

**代码**
- `domain/legal/rules.py`：20~30 条合同风险规则
- `domain/legal/risk_types.py`：风险分类体系
- 规则库入库（可被检索）

**知识点**
- **规则库 vs 模型判断**：规则确定、可解释、可审计；模型灵活、有幻觉风险。生产系统通常是"规则兜底 + 模型扩展"
- 规则与 RAG 的配合：规则命中优先，规则未覆盖的靠检索法条 + 模型判断
- 风险分级标准设计

**面试题**
- Q：有了大模型为什么还需要规则库？
- A：规则处理**确定性问题**——比如"违约金比例超过合同金额 30% 必须预警"，这类判断不需要模型，用规则更快、更准、可审计、可解释，而且不会漏。模型负责规则覆盖不到的语义类风险。两者互补：规则保证底线不漏，模型提升覆盖面。而且规则命中是可解释的，出了争议能说清依据。

**自检**
- 规则可被检索与匹配

---

## Day 17｜引用可回溯机制（本周关键）

**代码**
- 每个风险点的 `legal_basis` **必须带知识库条目 ID**
- 写回溯校验函数：给一个引用，验证 ID 是否真实存在
- 校验失败 → `evidence_sufficient = False` → 报告标"依据不足"

**知识点**
- **幻觉抑制的工程手段**：不靠模型"注意别编造"，而是靠**事后程序化验证**
- 引用验证的实现：ID 存在性校验 + 内容一致性校验
- 为什么"证据不足"必须显式输出，而不是静默给结论

**面试题**
- Q：你怎么防止模型编造法条？
- A：不靠提示词要求模型"不要编造"——那不可靠。我用的是**程序化验证**：所有法条引用必须指向知识库里的具体条目 ID，生成后系统校验 ID 是否真实存在。校验不通过就不允许这个风险点带法条依据，并把 `evidence_sufficient` 置为 false，报告里明确标"依据不足，建议人工复核"。**宁可说不知道，也不能给出看起来专业的错误引用。**

**自检**
- **故意让模型编造一个不存在的法条，系统能识别并标记"依据不足"**（写进 `tests/`）

---

## Day 18｜风险报告 schema + 汇总节点

**代码**
- `agent/capabilities/summarize.py`：汇总能力（独立函数，编码约定 1）
- 输出风险报告 JSON + Markdown
- 强制规则校验：`legal_basis` 为空且 `evidence_sufficient` 为 false → 标"依据不足"；`risk_level=高` → 必须进人工确认

**知识点**
- 输出契约设计：下游消费 JSON，人看法务读 Markdown
- 数据校验放在输出层，而不是信任上游
- 报告的可追溯性结构

**面试题**
- Q：为什么同时输出 JSON 和 Markdown？
- A：JSON 给下游系统和评测用，结构稳定可程序化处理；Markdown 给法务看，带原文引用和排版，人能快速核对。同一份数据两种呈现，避免为了展示去改数据结构。

**自检**
- 输出符合 schema；高风险项被标记为待确认

---

## Day 19｜幂等设计

**代码**
- `core/idempotency.py`：幂等键（`合同ID + 内容哈希 + 请求参数`）
- 存储层唯一约束
- 同一请求重试只返回同一份报告

**知识点**
- **幂等模式**：为什么"重试"和"副作用"放在一起会出事
- 幂等键设计原则：能唯一标识"同一个业务意图"
- 数据库唯一约束 vs 应用层判重（**两层都要**，应用层有并发竞态）

**面试题**
- Q：怎么保证重试不产生重复报告？
- A：我用幂等键，由"合同 ID + 内容哈希 + 请求参数"生成。请求进来先查这个键是否已有结果，有就直接返回。但只靠应用层查询有并发竞态——两个请求同时查都说没有，然后都写。所以数据库层加了唯一约束兜底，第二个写入会失败，然后走"查询已有结果"的分支返回。**两层保护，应用层省资源，数据库层保正确。**

**自检**
- 同一请求连续提交多次，只产生一份报告

---

## Day 20｜权限校验

**代码**
- `core/auth.py`：API Key 鉴权 + **资源归属校验**
- `agent/tool_policy.py`：工具白名单（编码约定 3）
- 越权请求返回 403 并记录日志

**知识点**
- **认证（Authentication）vs 授权（Authorization）**：前者"你是谁"，后者"你能看什么"
- **为什么不能只靠前端隐藏**：前端是用户可控的，任何前端限制都能绕过
- 资源归属校验：不只是"登录了没"，还要"这个资源是不是你的"
- 工具白名单的双重用途：权限控制 + 未来多 Agent 的工具集

**面试题**
- Q：权限校验为什么不能只做在前端？
- A：前端跑在用户浏览器里，用户完全可控——隐藏按钮只是体验优化，不是安全措施，攻击者直接构造请求就能绕过。所以授权必须在后端每个接口都做，而且要校验到**资源归属**这一层，不能只校验"登录了没"。我的越权测试就是直接用无权限账号发请求，验证后端会拒绝。

**自检**
- 用账号 A 的 Key 请求账号 B 的合同 → 403

---

## Day 21｜边界测试 + 周复盘

**代码**
- `tests/test_boundaries.py`：越权、重复提交、缺失字段、非法参数
- 把本周所有边界规则整理进 README 的"不做什么"章节

**知识点**
- 边界测试设计：从"明确不做什么"反推测试用例（**边界表就是测试用例的来源**）
- 负向测试的价值：正向功能跑通不代表系统可靠

**本周理论文档**（选读）
- Pydantic 校验器：<https://docs.pydantic.dev/latest/concepts/validators/>

**本周自检**
- [ ] 边界测试全部通过
- [ ] 编造法条能被识别
- [ ] 幂等与越权都有测试覆盖

**里程碑**：`git tag v0.3.0-domain`

---

# 第四周｜评测体系 + LangGraph + 模型对比

**本周目标**：建立让面试官可以追问的评测集，把流程升级为状态机，并完成本地 vs API 的模型对比实验。

## Day 22｜评测集设计：40 条正常样本

**代码**
- `evaluation/datasets/normal_40.json`
- 每条含：`id` / `input` / `expected_text` / 期望引用的条款

**知识点**
- 评测集设计原则：**覆盖问题类型，而不是堆数量**
- 三类问法：事实型、关系型、混合型
- 标注一致性：expected 要能客观判定

**面试题**
- Q：你的评测集怎么设计的？
- A：40 条正常样本覆盖三类问法，每类都标注了期望引用的条款，这样检索和生成能分开评估。另外 20 条风险样本专门测边界——越权、重复提交、工具超时、提示注入。**正常样本测"做得好不好"，风险样本测"会不会乱做事"**，后者是我更看重的。

**自检**
- 40 条覆盖三类问法，每条都有 expected

---

## Day 23｜风险样本集：20 条

**代码**
- `evaluation/datasets/risk_20.json`
- 6 类各覆盖若干：越权访问、重复提交、危险描述、工具超时、信息冲突、**提示注入**

**知识点**
- 风险样本的建模方式：不只标"期望输出文本"，还要标**期望动作**（允许调用哪些工具、最终状态、是否转人工）
- 提示注入的常见形式与检测

**自检**
- 6 类风险都有覆盖

---

## Day 24｜确定性断言框架（本周关键）

**代码**
- `evaluation/assertions.py`：按项目规划 §15.3 的模式实现
- 断言维度：`outcome`、`forbid_tools`、`max_reports`、是否转人工

**知识点**
- **确定性断言 vs LLM 打分**：前者客观可复现，后者有噪声
- **核心理念：文本相似度不是唯一质量标准**，权限和副作用必须用代码直接验收
- 断言设计：从"期望动作"而不是"期望文本"出发

**面试题**
- Q：为什么不能只用 RAGAS 评估？
- A：RAGAS 用 LLM 评判，适合评估"答得好不好"，但对"该不该拒绝""有没有重复写库"这类**权限和副作用**问题是无效的——它只能看文本，看不到数据库状态和工具调用记录。所以我另外做了确定性断言层：越权必须 deny、允许调用的工具集、最终报告数量上限，这些直接用代码断言，不经过任何模型。**能确定性验收的，绝不交给模型打分。**

**自检**
- 断言能捕获越权与重复提交（**故意让系统犯错，看断言是否报警**）

---

## Day 25｜RAGAS 接入

**代码**
- `evaluation/ragas_eval.py`
- 跑四项：Faithfulness / Answer Relevancy / Context Precision / Context Recall

**知识点**
- **四项指标分别在评估什么**：Context Precision 看检索排序质量，Context Recall 看覆盖率，Faithfulness 看是否忠于材料，Answer Relevancy 看是否切题
- 两项看检索、两项看生成——**必须分开看**，否则无法定位问题
- LLM-as-judge 的局限：评分模型自身能力上限、成本、不一致性
- 用本地小模型做评分的可行性（但要注意评分质量）

**面试题**
- Q：RAGAS 四项指标分别在评估什么？怎么定位问题？
- A：两项看检索、两项看生成。Context Recall 低说明该召回没召回，查切分和检索策略；Context Precision 低说明召回了无关内容，查排序和精排阈值；Faithfulness 低说明模型没忠于材料，查 Prompt 和上下文长度；Answer Relevancy 低说明答非所问，查 Prompt 的任务描述。**分开看才能定位，混在一起只有一个总分是没法调的。**

**自检**
- 跑出四项指标，记录在结果表里

---

## Day 26｜LangGraph 状态机

**代码**
- `agent/state.py`：**Pydantic State，按能力分字段**（编码约定 2）
- `agent/graph.py`：StateGraph 连接节点
- 先跑通"解析 → 检索 → 汇总"线性流程

**知识点**
- StateGraph 核心 API：`add_node` / `add_edge` / `add_conditional_edges` / `START` / `END` / `compile`
- State 是节点间共享的"笔记本"，节点返回 state update
- **LangGraph 支持用 Pydantic 模型作为 State**
- LangGraph 与 LangChain 的关系：LangChain 是组件底座，LangGraph 是有状态运行时

**面试题**
- Q：LangGraph 和 LangChain 的区别？
- A：LangChain 是通用的组件底座，提供 LLM、Retriever、Tool 这些原子组件，生态最大、也最适合新手切入；LangGraph 是建立在它之上的**有状态运行时**，用状态机管理组件间的执行流程，支持循环、条件分支、状态持久化和中断，面向长时间运行、需要人工介入的生产级 Agent。

**自检**
- 图能执行"解析 → 检索 → 汇总"并输出结果

---

## Day 27｜条件边 + 循环 + 上限

**代码**
- `verify_evidence` 节点：判断证据是否充分
- 条件边：证据不足 → 回到检索（可带查询改写）
- `iteration_count` 硬上限 3 + 超时中断

**知识点**
- 自适应检索：简单问题走快速通道，复杂问题多轮迭代
- Self-RAG 思路：让模型自己判断"检索到的文档是否足以回答"
- **死循环保护的两个机制**：语义判断（证据是否充分）+ 硬上限（迭代次数）
- 为什么两个都要：只有语义判断会转不出来，只有硬上限会过早停止

**面试题**
- Q：你的 Agent 怎么判断什么时候该停止检索？
- A：两个机制配合。一是 `verify_evidence` 判断证据是否充分——不充分就带着改写后的查询回到检索；二是 `iteration_count` 硬上限 3 次。只有语义判断会有转不出来的风险，只有次数上限又会过早停止导致答案不完整。**这是我在测试时踩过的坑**：第一版只有语义判断，遇到一个模糊问题反复检索了 7 轮。

**自检**
- 简单问题 1 轮返回；证据不足触发重检索；不会死循环

---

## Day 28｜本地 vs API 模型对比实验 + 结果表

**代码**
- `evaluation/model_benchmark.py`
- 跑四组配置（照项目规划 §14.3）：
  - L1：`bge-small-zh-v1.5` + `bge-reranker-base`
  - L2：`bge-base-zh-v1.5` + `bge-reranker-base`（默认）
  - L3：`bge-large-zh-v1.5` + `bge-reranker-v2-m3`
  - A1：API（`bge-m3` + `bge-reranker-v2-m3`）
- 采集：embedding 延迟 P50/P95、精排 20 候选延迟 P50/P95、关键条款召回率、RAGAS Recall/Faithfulness、内存占用、调用成本
- 用完整评测集跑一次，填写结果表

**知识点**
- **性能基准测试方法**：预热、多轮取分位数（不要只看均值）、控制变量
- 延迟与质量的关系：为什么"更大的模型"不一定"更划算"
- 成本换算：本地推理的电费/机器成本 vs API 调用成本（生产环境要算这笔账）
- 如何读结果表定位退化：指标下降先看是检索层还是生成层

**面试题**
- Q：本地小模型效果够吗？你怎么验证的？
- A：我做了 4 组配置的对比实验，采集了延迟、召回、内存三个维度。合同场景有两个特点使得小模型够用：**领域收敛**——检索目标是条款和法条，用词规范，不需要开放域的泛化能力；**候选池窄**——精排只需处理 10~20 个候选，不需要覆盖长尾。实测结果是本地 `bge-base` + `bge-reranker-base` 的关键条款召回率与 API 大模型相差 __ 个百分点，但精排 P95 延迟从 __ms 降到 __ms，且零调用成本、数据不出内网。**所以默认走本地，API 作为配置项保留。**

**自检**
- [ ] 4 组配置的对比数据表完成
- [ ] 精排延迟与质量的关系有量化结论
- [ ] 能说清为什么最终选本地 base 级模型

**本周理论文档**（选读）
- RAGAS 官方文档：<https://docs.ragas.io/>
- LangGraph 概念：<https://langchain-ai.github.io/langgraph/concepts/low_level/>
- Self-RAG 论文：<https://arxiv.org/abs/2310.11511>

**本周自检**
- [ ] 60 条评测集完成
- [ ] 确定性断言能捕获越权与重复
- [ ] RAGAS 四项跑通并有数字
- [ ] LangGraph 状态机含条件重检索
- [ ] 本地/API 对比实验完成

**里程碑**：`git tag v0.4.0-eval`

---

# 第五周｜MCP + 双 Agent

**本周目标**：做出项目最值钱的部分——MCP Server，并把两个真正需要自主循环的能力升级为 Agent。

## Day 29｜MCP SDK v2 跑通

**代码**
- `mcp/server.py`：用官方 SDK v2 的 `MCPServer` 建实例
- 客户端能连上并列出工具

**知识点**
- MCP 是什么：Model Context Protocol，Agent 工具调用的事实标准
- **2026-07-28 规范的核心变化**：移除 `initialize` 握手和 `Mcp-Session-Id`，每请求自包含；新增 `server/discover`；结果必须带 `resultType`
- ⚠️ **官方 SDK v2 已把 `FastMCP` 更名为 `MCPServer`**，旧导入路径 `mcp.server.fastmcp` 已**删除**（不是弃用）
- ⚠️ `stateless_http=True` 与 2026 流量**无关**，只影响旧版客户端的服务方式

**面试题**
- Q：MCP 为什么用无状态版本？
- A：旧版依赖 `Mcp-Session-Id` 和粘性会话，会话绑定到具体服务器实例，横向扩展必须做粘性路由和共享会话存储，Serverless 基本不可行。2026-07-28 移除了握手和会话 ID，每个请求自带协议版本和客户端能力，任何实例都能处理，标准负载均衡和 Serverless 直接可用。**补充一点容易被误解的**：`stateless_http` 这个参数其实和 2026 流量无关，它只影响旧版客户端怎么被服务。

**自检**
- Server 启动成功，Client 能列出工具

---

## Day 30｜5 个 Tools

**代码**
- `mcp/tools.py`：`contract_analyze`、`clause_search`、`risk_detect`、`legal_retrieve`、`cross_reference`
- 每个工具用 Pydantic 做参数校验

**知识点**
- Tool schema 设计要素：**名称、描述、参数、返回值**——描述写得好不好直接决定模型会不会正确调用
- 工具粒度设计：太细导致调用轮次多，太粗导致模型无法灵活组合
- 工具参数校验的位置：**执行前**校验，不信任模型输出

**面试题**
- Q：工具 schema 怎么设计？
- A：三个要点。一是**描述要写清"什么时候用"而不是"这是什么"**——模型靠描述选择工具，写"检索法条"不如写"当需要确认某个条款是否违反法律规定时，检索相关法条"。二是参数尽量窄，能用枚举就不用自由文本。三是执行前必须用 Pydantic 校验参数，不能信任模型输出——校验失败就重试或降级，绝不带着错误参数执行。

**自检**
- 5 个工具都能被调用并返回结构化结果

---

## Day 31｜Resources + 规范合规

**代码**
- `mcp/resources.py`：合同元数据、风险规则库、法条索引
- 实现 `server/discover`
- 确保结果带 `resultType`

**知识点**
- **Tools vs Resources**：Tools 是模型主动调用的动作，Resources 是可读的数据
- 规范里的 MUST / SHOULD：`server/discover` 是 MUST，`clientInfo` 是 SHOULD
- 为什么 `subscriptions/listen` 取代了 HTTP GET 端点和 `resources/subscribe`

**自检**
- `server/discover` 可调用，结果带 `resultType`

---

## Day 32｜风险识别 Agent 化（第 1 个真 Agent）

**代码**
- `agent/capabilities/risk_detect.py`：从节点升级为**带工具循环的 Agent**
- 自主决定：查规则库 / 检索法条 / 检索判例
- 检索不足时**改写查询重试**
- 通过 `tool_policy.py` 限制可用工具

**知识点**
- **节点 vs Agent 的判据**：这一步需不需要"自己决定下一步做什么"
- ReAct 模式：Thought → Action → Observation 循环
- 查询改写策略：同义改写、子问题分解、step-back
- 工具循环的终止条件

**面试题**
- Q：什么时候该用 Agent，什么时候用节点就够？
- A：判据是"这一步需不需要自主决定下一步做什么"。我的风险识别需要——它要自己决定查规则库还是检索法条还是判例，结果不够还要改写查询重来，这是有自主性的，所以做成 Agent。而解析和汇总是一次性的确定性处理，做成普通节点就够。**给不需要自主性的步骤套 Agent 循环，只会白花 token 和增加延迟。**

**自检**
- Agent 能自主决定检索路径；证据不足时改写查询重试

---

## Day 33｜复核 Agent（第 2 个真 Agent）

**代码**
- `agent/capabilities/verify.py`：**独立**复核
- 检查三项：证据是否充分、法条引用能否回溯、有没有漏项
- 复核不通过 → 打回重检索 或 标记"依据不足"

**知识点**
- **自我复核 vs 独立复核**：同一个 Agent 检查自己不可靠（同样的盲区会重复）
- 幻觉检测的实现：引用存在性 + 内容一致性 + 交叉来源验证
- 为什么复核用独立的上下文（不共享原 Agent 的推理过程）

**面试题**
- Q：为什么要单独做一个复核 Agent？模型自己检查自己不行吗？
- A：不行，这是结构性原因——同一个模型带着同一套上下文检查自己的输出，会重复同样的盲区。独立复核 Agent 有自己的上下文和工具权限，从"这份结论有没有依据"的角度重新审一遍，是**独立验证而不是自我确认**。

**自检**
- 复核能标出证据不足的风险项；漏项能被发现

---

## Day 34｜HITL：用 MRTR 实现人工确认

**代码**
- `human_confirm` 中断点：高风险项暂停流程
- 用 **MRTR（多轮往返请求）**：工具返回 `InputRequiredResult`，客户端带答案重试
- SDK 层面用 `Resolve(...)` 注解参数

**知识点**
- ⚠️ **2026-07-28 移除了所有服务端主动发起的请求**：`ctx.elicit()` 在新协议连接上会抛 `NoBackChannelError`
- **MRTR 模式**：把"服务端问客户端"翻转成"工具返回问题 + 客户端重试原调用"
- LangGraph 的中断与恢复（checkpointer + interrupt）
- 为什么这个设计对 Serverless 友好

**面试题**
- Q：Human-in-the-loop 怎么实现的？
- A：用多轮往返请求。旧的 elicitation 是服务端主动向客户端发请求，2026-07-28 把这套机制整个移除了——在 2026 连接上根本没有这个通道，`ctx.elicit()` 会直接抛 `NoBackChannelError`。新做法是翻转过来：工具**返回**一个 `InputRequiredResult`，里面带着要问的问题，客户端拿到答案后重试原调用，服务端这次就能继续。SDK 层面用一个 `Resolve` 注解包住参数，两种协议时代的客户端都能用同一份工具代码。

**自检**
- 高风险项触发确认；确认后才生成最终报告；未经确认不出现结论

---

## Day 35｜Agent 评估 + 周复盘

**代码**
- 统计任务完成率、工具调用准确率
- 对比"只有节点" vs "双 Agent"的指标差异

**知识点**
- Agent 评估维度：任务完成率、步骤效率（轮次）、工具调用准确率
- 如何归因 Agent 的失败：工具选错、参数错、还是终止判断错
- 多 Agent 的代价量化：延迟与成本增加了多少

**本周理论文档**（选读）
- MCP 2026-07-28 规范变更：<https://modelcontextprotocol.io/specification/2026-07-28/changelog>
- MCP Python SDK v2：<https://py.sdk.modelcontextprotocol.io/v2/whats-new/>
- LangGraph 多 Agent：<https://langchain-ai.github.io/langgraph/tutorials/multi_agent/>

**本周自检**
- [ ] MCP Server 可被客户端调用，含 `server/discover` 与 `resultType`
- [ ] 2 个 Agent（风险识别、复核）+ 3 个节点（解析、汇总、人工确认）
- [ ] HITL 用 MRTR 实现并能跑通
- [ ] 有"双 Agent vs 单流程"的指标对比

**里程碑**：`git tag v0.5.0-agent-mcp`

---

# 第六周｜可靠性 + 交付

**本周目标**：把系统"弄坏三次"并修复，然后交付。**这一周决定你的项目是"课程作业"还是"工程项目"。**

## Day 36｜故障实验 1：断网重试与幂等

**代码**
- 在"模型已决定出报告、工具尚未返回"时切断网络
- 观察是否产生两份报告
- 若产生 → 修复 → 写回归用例

**知识点**
- 故障注入的思路：在**副作用发生前半步**打断，最容易暴露幂等缺陷
- 需要保留的证据：输入、代码版本、Trace、数据库终态、回归用例

**自检**
- 修复后重复报告数为 0；`tests/failure_cases/fault_01.md` 记录完整

---

## Day 37｜故障实验 2：超时降级不编造

**代码**
- 让法条检索服务超时
- 观察系统是"静默继续并编造法条"还是"显式降级"
- 若编造 → 修复为显式降级 + 标记"依据不足"

**知识点**
- **降级的设计原则**：明确告知哪一步没完成，而不是让模型用"记忆"补上
- 降级链：主工具 → 备用工具 → 显式标记未完成 → 转人工
- 静默失败为什么最危险：它把工程问题伪装成了模型能力问题

**自检**
- 编造法条数为 0；`tests/failure_cases/fault_02.md` 记录完整

---

## Day 38｜故障实验 3：越权拦截

**代码**
- 用无权限账号请求他人合同
- 验证前端隐藏与后端鉴权**同时**成立
- 只靠前端 → 修复后端

**知识点**
- 纵深防御：前端体验优化 + 后端强制校验，两者目的不同
- 越权测试的构造：直接构造 HTTP 请求绕过前端

**自检**
- 越权请求返回 403；`tests/failure_cases/fault_03.md` 记录完整

---

## Day 39｜Langfuse 全链路埋点

**代码**
- `core/observability.py`：初始化 Langfuse
- 加 `@observe`：`retrieve`、`rerank`、`generate`、`tool_call`、`agent_node`
- ⚠️ **正确导入**：`from langfuse import observe`（`langfuse.decorators` 已不可用）
- 确保 SDK 版本 ≥ 4.7.0（否则数据延迟 10~15 分钟）
- **额外给本地模型阶段单独打点**：embedding 耗时、精排耗时、线程等待时间

**知识点**
- Trace / Span / Observation 的层级关系
- observation-first 架构：评测下沉到操作粒度（v4 中 Trace-level evaluators 已弃用）
- 成本与延迟归因：从 Trace 里定位是哪一步慢/贵
- **本地推理的耗时拆分**：真实计算时间 vs 线程池排队时间

**面试题**
- Q：为什么用 Langfuse 而不是 LangSmith？
- A：LangSmith 与 LangChain 强绑定，Langfuse 框架无关且可自部署，数据不出 VPC，满足合规要求——这对法务场景很实际。v4 是 observation-first，评测能下沉到具体操作粒度。**另外我了解 LangSmith 的对应概念**，因为它在 LangGraph 岗位里出现频率很高。

**自检**
- Langfuse 看板能看到一次完整审查的全链路，含延迟与 Token
- 能看到本地精排阶段在总延迟中的占比

---

## Day 40｜Docker Compose 一键部署

**代码**
- `Dockerfile`：多阶段构建
- `docker-compose.yml`：app + Milvus + Redis + Langfuse
- **模型用 volume 挂载**（不打进镜像）
- `.env.example` 补齐全部环境变量

**知识点**
- 多阶段构建：build 阶段装依赖，runtime 阶段只拷必要文件
- Compose 的服务依赖与健康检查（`depends_on` + `healthcheck`，避免 app 比 Milvus 先起来）
- 环境变量注入，避免硬编码密钥
- **镜像与模型分离**：模型约 1.5 GB，打进镜像会让构建和分发都变重
- 容器内 CPU 线程数与宿主机核心数的关系

**面试题**
- Q：你的项目怎么部署？
- A：Docker Compose 一键起 app、Milvus、Redis、Langfuse 四个服务。Dockerfile 用多阶段构建减小镜像体积，环境变量用 `.env` 注入。**模型权重用 volume 挂载而不是打进镜像**——两个模型约 1.5 GB，打进镜像会让构建和分发都变重。**踩过的坑是服务启动顺序**：app 比 Milvus 先起来会连接失败，所以用 healthcheck 加 `depends_on` 的 condition 控制。另一个坑是容器里 CPU 线程数——如果不限制，容器会看到宿主机所有核心并吃满，影响同机器上的其他服务。

**自检**
- `docker compose up -d` 后 Swagger、UI、Langfuse 都可访问
- 镜像体积在合理范围（未包含模型权重）

---

## Day 41｜README 六问 + 结果表 + Failure Analysis

**代码/文档**
- README 回答六个问题（项目规划 §18.1）
- 填写结果表（**没做的写"待验证"，绝不伪造**）
- Failure Analysis：三个故障实验 + 开发过程中的问题

**知识点**
- 六个问题的设计意图：**让别人能复现、能审查、能追问、能相信它失败时不会乱做事**
- 诚实边界比漂亮数字更专业
- Failure Analysis 的写法：现象 → 定位 → 解决 → 结果

**自检**
- 一个没参与项目的人能按 README 跑起来（含模型下载步骤）

---

## Day 42｜简历 + 项目故事 + 模拟面试

**文档**
- 简历：按证据链写法（项目规划 §19.1），**只用实测数字**
- 1 分钟版 + 5 分钟版项目介绍
- 按项目规划 §19.2 的六个故事和 §19.3/§19.4 的问题清单自测
- 模拟面试：录音讲一遍

**知识点**
- 面试叙事结构：讲原理 → 讲场景选择 → 讲工程落地 → 讲可观测与评估
- STAR 法则
- **保险知识点的答法**（项目规划 §19.4）：RAG vs 微调、vLLM、Attention/KV Cache、量化

**自检**
- [ ] 能流畅讲 1 分钟和 5 分钟版本
- [ ] 六个故事每个都能讲 2 分钟，带具体参数
- [ ] 三个故障实验能各讲 1 分钟
- [ ] 本地模型性能优化那段能讲清瓶颈定位与四步优化
- [ ] 四道保险题能答

**里程碑**：`git tag v1.0.0`

---

## 四、知识点总清单（自查用）

打勾前先自问"能不能用自己的话讲 2 分钟"。

### RAG 与检索
- [ ] RAG 原理与适用场景；与长上下文窗口的取舍
- [ ] 文档解析难点（PDF 版式、表格、扫描件）
- [ ] 切分策略：固定 / 递归 / 语义 / **条款级**
- [ ] chunk_size 与 overlap 的权衡（有实测数据）
- [ ] Embedding 原理；稠密 vs 稀疏
- [ ] 相似度度量：余弦 / 内积 / 欧氏；归一化
- [ ] 向量库索引：HNSW vs IVF_FLAT
- [ ] BM25：k1（词频饱和）、b（长度归一化）
- [ ] **RRF 公式**：`score(d) = Σ 1/(k + rank(d))`，k=60，为什么无需调参
- [ ] Bi-Encoder vs Cross-Encoder
- [ ] **宽进严出**；为什么"混合必须配精排"

### 本地模型部署（本项目重点）
- [ ] 模型参数量 → 权重体积 → CPU 推理延迟的关系
- [ ] **Cross-Encoder 为什么无法预计算，因而成为延迟瓶颈**
- [ ] **`asyncio.to_thread` 为什么必需**（GIL、阻塞、事件循环）
- [ ] `torch.set_num_threads` 与并发吞吐的关系
- [ ] lifespan 预热为什么重要
- [ ] 精排延迟优化：候选裁剪 / 文档截断 / 缓存 / 线程隔离
- [ ] 小参数模型在收敛领域够用的原因
- [ ] 本地 vs API 的取舍维度：质量 / 延迟 / 成本 / 数据合规
- [ ] 模型选型做成配置项的意义
- [ ] HuggingFace 模型下载与镜像、离线加载
- [ ] Docker 中模型与镜像分离的做法

### Prompt 与模型调用
- [ ] Prompt 五要素：角色 / 任务 / 约束 / 格式 / 示例
- [ ] 结构化输出（JSON mode / function calling）
- [ ] Pydantic 输出校验与重试
- [ ] **Prompt 注入的 2 种攻击方式与防御**
- [ ] temperature / top_p 的取值逻辑

### Agent 编排
- [ ] ReAct：Thought → Action → Observation
- [ ] StateGraph API：`add_node` / `add_edge` / `add_conditional_edges` / `START` / `END` / `compile`
- [ ] State 设计；**LangGraph 支持 Pydantic 作 State**
- [ ] 条件路由与循环终止
- [ ] **死循环保护的两个机制**
- [ ] Self-RAG 思路
- [ ] **节点 vs Agent 的判据**
- [ ] 自我复核 vs 独立复核
- [ ] Supervisor 模式；subgraph 与 handoff
- [ ] LangGraph vs LangChain

### MCP
- [ ] MCP 解决的问题；Tools vs Resources
- [ ] **2026-07-28 无状态**：移除握手与会话 ID
- [ ] `server/discover`（MUST）与 `resultType`（必填）
- [ ] `Mcp-Method` / `Mcp-Name` 头
- [ ] **MRTR**；为什么 elicitation 不可用
- [ ] `MCPServer` 更名（旧 `FastMCP` 路径已删除）
- [ ] `stateless_http` 与 2026 流量无关

### 生产工程
- [ ] 幂等键设计；应用层 + 数据库两层保护
- [ ] 认证 vs 授权；为什么授权必须在后端
- [ ] 重试（tenacity 指数退避）与**显式降级**
- [ ] 静默失败为什么最危险
- [ ] 并发控制（asyncio.Semaphore）
- [ ] tiktoken 成本核算
- [ ] 缓存策略（查询缓存 / 精排缓存）
- [ ] 多阶段 Docker 构建；Compose 健康检查与启动顺序
- [ ] 依赖版本锁定策略

### 评测与可观测
- [ ] RAGAS 四项各评估什么；**两项看检索、两项看生成**
- [ ] 业务指标：关键条款召回率、漏报率、误报率
- [ ] 评测集设计：覆盖类型而非堆数量
- [ ] **确定性断言 vs LLM 打分**
- [ ] 消融实验设计；性能基准测试方法（预热、分位数）
- [ ] Trace / Span / Observation 层级
- [ ] Langfuse v4 的 observation-first；`from langfuse import observe`

### 保险知识点（不做，但要能答）
- [ ] **RAG vs 微调的选择标准**
- [ ] LoRA / QLoRA 为什么省显存
- [ ] **生产为什么用 vLLM 而不是 Ollama**（PagedAttention、Continuous Batching）
- [ ] **量化怎么选**（GPTQ / AWQ / GGUF 的适用场景）
- [ ] **Attention 计算流程**（Q/K/V → 缩放 → Softmax → 加权）
- [ ] 为什么 Transformer 能并行训练
- [ ] **KV Cache 为什么降低推理延迟**，代价是什么
- [ ] RoPE 解决什么问题

---

## 五、面试题库

见 [项目规划文档 §19](ContractLens_项目规划文档.md#十九简历与面试准备)，共 15 道高频题 + 4 道保险题。**建议做法**：每天副轨时间挑一题，用自己的话录音讲 2 分钟，回放听哪里卡壳。

---

## 六、里程碑 Tag 一览

| Tag | 时间 | 含义 |
| --- | --- | --- |
| `v0.1.0-rag-loop` | Day 7 | 端到端闭环 + UI + 本地模型就位 |
| `v0.2.0-retrieval` | Day 14 | 混合检索 + 精排 + 延迟优化 + 两组对比数据 |
| `v0.3.0-domain` | Day 21 | 结构化报告 + 幂等 + 权限 + 边界测试 |
| `v0.4.0-eval` | Day 28 | 评测集 + 确定性断言 + LangGraph + 模型对比 |
| `v0.5.0-agent-mcp` | Day 35 | MCP Server + 双 Agent + HITL |
| `v1.0.0` | Day 42 | 故障实验 + 一键部署 + README + 简历 |

---

## 七、落后了怎么砍

**从下往上砍。**

| 优先级 | 内容 | 砍掉的影响 |
| --- | --- | --- |
| 🔴 **必保** | RAG 闭环、混合检索+精排对比数据、本地模型跑通、边界（幂等/权限/引用可回溯）、评测集+确定性断言、MCP Server | 砍了就没法投简历 |
| 🟠 尽量保 | 双 Agent 复核、LangGraph 状态机、本地/API 对比实验、三个故障实验、Langfuse | 复核 Agent 可退化为"复核节点"；故障实验至少保 2 个 |
| 🟡 可延后 | 判例库（可先只做法条）、Agent 评估深化、L3 大模型本地对比 | 投完简历再补 |
| 🟢 可砍 | 规则库扩张到 30 条、限流、脱敏 | 影响很小 |

**如果 W3 结束还卡在 W2**：直接跳到"必保"清单顺序推进，先做出一个**能讲的最小版本**去投简历，剩下的边投边补。

---

## 八、每日自查清单（贴墙）

- [ ] 今天的代码能运行吗？
- [ ] 今天有没有留下**可对比的数据**？（面试素材）
- [ ] 今天遇到的问题记录下来了吗？（Failure Analysis 原料）
- [ ] 推到 GitHub 了吗？
- [ ] 副轨的面试题能用自己的话讲 2 分钟吗？
- [ ] 距离"别人能复现"还差什么？

---

## 附录：版本核查来源

以下版本号于本次核查时从 PyPI 官方页面取得（2026 年 9–10 月）：

| 包 | 核查到的版本 | 链接 |
| --- | --- | --- |
| langfuse | 4.16.0 | <https://pypi.org/project/langfuse/> |
| mcp | 2.2.0 | <https://pypi.org/project/mcp/> |
| langgraph | 1.2.12 | <https://pypi.org/project/langgraph/> |
| ragas | 0.4.3 | <https://pypi.org/project/ragas/> |
| sentence-transformers | 6.1.0（本项目用 5.7.0） | <https://pypi.org/project/sentence-transformers/> |
| transformers | 5.18.0（本项目用 4.x 线） | <https://pypi.org/project/transformers/> |
| torch | 2.14.1 | <https://pypi.org/project/torch/> |
| pymilvus | 3.0.2（本项目用 2.6.17） | <https://pypi.org/project/pymilvus/> |

**pymilvus 与 Milvus 的兼容关系**来自 pymilvus 官方文档的兼容表：

| Milvus 版本 | 推荐 pymilvus 版本 |
| --- | --- |
| 2.4.\* | 2.4.X |
| 2.5.\* | 2.5.X |
| **2.6.\*** | **2.6.X** |

---

*本计划配套 [ContractLens 项目规划文档](ContractLens_项目规划文档.md)。*

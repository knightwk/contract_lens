# ContractLens

受监管文档（合同 / 法条 / 判例）智能审查 Agent 平台。

上传合同 → 条款级混合检索 → 风险识别 → 复核 → **带可回溯引用的风险报告**。

> 面向法律、金融等受监管场景：**每条结论都必须能回答"你凭什么这么说"。**

---

## 当前状态

**Day 2 / 42**（W1 — 闭环打通 + 本地模型就位）

已完成：

- [x] 目录结构与分层
- [x] 配置系统（pydantic-settings，含本地 / API 模型双 Provider 开关）
- [x] 统一日志（控制台 + 文件，多线程安全）
- [x] FastAPI 入口 + lifespan + 健康检查
- [x] 依赖版本锁定
- [x] **本地模型就位**：`bge-base-zh-v1.5` + `bge-reranker-base`，CPU 推理
- [x] `model_runtime`：单例加载 / lifespan 预热 / 线程数控制 / 并发闸门
- [x] embedding 与 rerank 的"本地 + API"双实现及 Provider 切换

进行中：

- [ ] Milvus 部署（需 Docker，Day 3）

> 冷启动与延迟基线见 [`docs/experiments.md`](docs/experiments.md)（EXP-002）。
> **结果数据待 W4 评测跑完后填写。** 本 README 不放未经验证的数字。

---

## 技术栈

| 层 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Python 3.12 | |
| Web | FastAPI + Uvicorn | 原生异步，适配 LLM 的 IO 密集场景 |
| 配置 | pydantic-settings | 环境变量 + `.env`，启动时校验 |
| 向量库 | Milvus 2.6.x | 稠密向量 + **内置 BM25** 全文检索 |
| Embedding | `BAAI/bge-base-zh-v1.5` | **本地 CPU 推理**，768 维 |
| Rerank | `BAAI/bge-reranker-base` | **本地 CPU 推理**，Cross-Encoder 精排 |
| LLM | OpenAI 兼容接口 | 可指向任意兼容服务 |
| 编排 | LangGraph | 状态机 + 条件边 + 循环上限 |
| 工具协议 | MCP 2.x | 2026-07-28 无状态规范 |
| 评测 | RAGAS + 确定性断言 | |
| 可观测 | Langfuse v4 | |
| UI | Streamlit | |

**本地模型优先**：embedding 与 rerank 默认跑在本地 CPU 上，离线、免费、数据不出机器。LLM 因参数量原因走 API。

---

## 快速开始

### 1. 环境要求

| 依赖 | 版本 | 必需 |
| --- | --- | --- |
| Python | 3.12 | 是 |
| git | 任意 | 是 |
| Docker Desktop | 最新 | W1 Day 3 起（Milvus） |
| 磁盘 | ≥ 3 GB | 模型约 1.5 GB + 依赖 |
| 内存 | ≥ 8 GB | 建议（本地模型常驻约 1.5 GB） |

### 2. 创建虚拟环境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> PowerShell 若报「禁止运行脚本」：
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 3. 安装依赖

```powershell
pip install -r requirements.txt
```

依赖较重（含 PyTorch），首次安装需要几分钟。分步安装便于定位冲突：

```powershell
pip install fastapi "uvicorn[standard]" pydantic==2.12.0 pydantic-settings python-dotenv loguru==0.7.3
pip install torch "transformers>=4.45,<5.0" sentence-transformers==5.7.0
pip install pymilvus==2.6.17 "openai>=3.3,<4.0" httpx==0.28.1 tenacity==9.1.2 tiktoken==0.9.0 numpy==2.2.6 pypdf python-docx streamlit
pip install langgraph==1.2.12 "mcp[cli]==2.2.0" ragas==0.4.3 langfuse==4.16.0 "redis>=5.0,<9.0"
pip install pytest==8.4.1 pytest-asyncio==0.26.0 ruff==0.12.8
```

> ⚠️ `openai` 必须是 **3.x**：`ragas` 依赖的 `langchain-openai` 要求 `openai>=2.45.0,<4.0.0`，
> 装 1.x 会导致解析失败。详见「版本策略」。

### 4. 配置环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少确认这几项：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEBUG` | `true` | 生产环境必须为 `false`（否则 `/docs` 对外暴露） |
| `EMBEDDING_PROVIDER` | `local` | `local` 或 `api` |
| `RERANK_PROVIDER` | `local` | `local` 或 `api` |
| `TORCH_NUM_THREADS` | `4` | torch 内部线程数。建议 `× MAX_CONCURRENT_INFERENCE` ≈ CPU 物理核数 |
| `LLM_API_KEY` | 空 | Day 3 起需要 |

> `.env` 必须存为 **UTF-8**——`EMBEDDING_QUERY_INSTRUCTION` 含中文，存成 GBK 会读成乱码且不报错。

### 5. 下载本地模型（约 1.5 GB）

**这一步不能跳过。** `models/` 不进 Git，而服务启动时会加载模型——缺失则启动直接失败。

```powershell
python scripts/download_models.py
```

下载到 `./models/bge-base-zh-v1.5`（102M 参数）与 `./models/bge-reranker-base`（278M 参数）。
脚本内置国内镜像 `hf-mirror.com`；换源改 `scripts/download_models.py`。

下载完跑一次冒烟测试，确认模型可用并记录**本机延迟基线**：

```powershell
python scripts/smoke_test_models.py
```

离线自包含验证（确认模型目录不依赖网络）：

```powershell
$env:HF_HUB_OFFLINE="1"; python scripts/smoke_test_models.py; Remove-Item Env:\HF_HUB_OFFLINE
```

### 6. 启动

```powershell
uvicorn app.main:app --reload
```

启动日志里应出现两行冷启动计时（预热在 `lifespan` 中完成，避免第一个请求承担冷启动）：

```
加载 embedding 模型: ...\models\bge-base-zh-v1.5
embedding 就绪 | 冷启动 0.90s
加载 rerank 模型: ...\models\bge-reranker-base
rerank 就绪 | 冷启动 1.78s
ContractLens 启动完成，准备接收请求
```

> 模型预热失败会导致**服务拒绝启动**（fail fast）——模型是本服务的核心能力，
> 与其带着坏掉的检索对外服务，不如启动就报错。

### 7. 验证

```powershell
curl.exe http://127.0.0.1:8000/health
```

期望返回：

```json
{
  "status": "ok",
  "app": "ContractLens",
  "debug": true,
  "providers": { "embedding": "local", "rerank": "local" },
  "models": { "embedding": "bge-base-zh-v1.5", "rerank": "bge-reranker-base", "dim": 768 },
  "models_ready": true,
  "model_runtime": {
    "embedder_loaded": true,
    "embedder_warmed": true,
    "reranker_loaded": true,
    "reranker_warmed": true,
    "torch_threads": 4
  }
}
```

三个字段值得留意：

| 字段 | 说明 |
| --- | --- |
| `app` | 必须等于 `.env` 里的 `APP_NAME`——验证配置真正生效。不符说明 `.env` 没被读到 |
| `models_ready` | 本地模型是否就绪。`provider=api` 时不检查本地模型 |
| `model_runtime.torch_threads` | 必须等于 `TORCH_NUM_THREADS`。**如果等于 CPU 核数，说明线程数没设上** |

交互式 API 文档（仅 `DEBUG=true`）：<http://127.0.0.1:8000/docs>

---

## 项目结构

```
contract_lens/
├── app/
│   ├── main.py                 # FastAPI 入口 + lifespan（模型预热）
│   ├── config.py               # 配置中心（pydantic-settings）
│   ├── api/
│   │   ├── health.py           # 基础设施端点（不版本化）
│   │   ├── router.py           # 业务路由汇总（挂 /api/v1）
│   │   └── v1/                 # documents / review / reports
│   ├── core/                   # 通用能力
│   │   ├── model_runtime.py    # ★ 本地模型加载/预热/线程调度/并发闸门
│   │   ├── embedding.py        # 本地 + API 双实现（含 Provider 工厂）
│   │   ├── reranker.py         # 本地 + API 双实现（含 Provider 工厂）
│   │   ├── parser.py           # 文档解析
│   │   ├── chunker.py          # 条款级切分
│   │   ├── vector_store.py     # Milvus schema（稠密 + BM25）
│   │   ├── retriever.py        # 混合检索 + RRF
│   │   ├── llm.py              # LLM 封装（流式/重试/成本）
│   │   ├── cache.py            # 查询与精排缓存
│   │   ├── idempotency.py      # 幂等键
│   │   ├── auth.py             # API Key + 资源归属校验
│   │   ├── observability.py    # Langfuse
│   │   └── logger.py
│   ├── domain/                 # ★ 可替换领域层
│   │   ├── base.py             # 领域规则抽象接口
│   │   ├── schemas.py          # 风险报告模型
│   │   └── legal/              # 规则库 / 法条库 / 判例库
│   ├── agent/                  # Agent 编排
│   │   ├── state.py            # 按能力分字段
│   │   ├── graph.py            # StateGraph
│   │   ├── nodes.py            # 适配层（薄）
│   │   ├── capabilities/       # ★ 能力本体
│   │   ├── tool_policy.py      # ★ 工具白名单
│   │   └── prompts.py
│   ├── mcp/                    # MCP Server（Tools + Resources）
│   └── evaluation/             # 评测集 + 断言 + 模型对比
├── ui/streamlit_app.py
├── scripts/
│   ├── download_models.py      # 模型预下载（约 1.5 GB）
│   └── smoke_test_models.py    # 模型冒烟测试 + 延迟基线
├── tests/                      # 含 failure_cases/ 故障实验记录
├── models/                     # 模型权重（不进 Git，约 1.5 GB）
├── docs/
│   ├── conventions.md          # ★ 编码约定（动手前必读）
│   ├── experiments.md          # ★ 实验记录（延迟基线 / 对比实验）
│   ├── architecture.md
│   ├── evaluation_report.md    # 结果表
│   └── interview_qa.md
└── requirements.txt            # 意图声明（含版本策略）
```

**分层的一个原则**：`app/core/` 放可复用能力，`app/domain/` 放业务规则，`app/agent/` 只负责编排。

领域层是可替换的——`domain/legal/` 换成 `domain/fintech/` 应该不需要改 `core/` 和 `agent/`。这是"受监管文档智能审查平台"而不是"合同审查工具"的架构依据。

---

## 版本策略

**不盲目追最新版本。** 选型依据是**官方兼容表 + 生态成熟度**，不是版本号大小。

几个刻意的"不用最新版"决策：

| 包 | 最新版 | 本项目 | 原因 |
| --- | --- | --- | --- |
| `pymilvus` | 3.0.2 | **2.6.17** | 官方兼容表要求与服务端 Milvus 2.6.x **同大版本**，跨大版本会出问题 |
| `transformers` | 5.18.0 | **4.x 稳定线** | 5.x 是新大版本，生态适配仍在推进 |
| `sentence-transformers` | 6.1.0 | **5.7.0** | 6.x 依赖 transformers 5.x；5.7.0 + transformers 4.x 组合更成熟 |

> ⚠️ `openai` 锁在 3.x **不属于**上表的"主动选旧"——它是**被依赖约束**的结果：
> `ragas` → `langchain-openai` 要求 `openai>=2.45.0,<4.0.0`，1.x 无法共存。
>
> 这类**被动锁定**比主动选旧更值得警惕：一个只在 W4 评测期用到的依赖，
> 替你决定了全项目的 SDK 大版本。发现它靠的是**查依赖约束**，不是靠读版本号。

版本约束写法：

- **`==` 精确锁定**：跨大版本会破坏 API 兼容的包（`pymilvus`、`langgraph`、`mcp`、`langfuse`、`pydantic` 等）
- **`>=x,<y` 区间**：小版本升级只是修 bug 的包（`fastapi`、`uvicorn`、`pypdf` 等）

`requirements.txt` 是**意图声明**，`requirements.lock.txt` 是**实际环境记录**（`pip freeze` 产物，必须提交）。

```powershell
pip freeze | Out-File -Encoding utf8 requirements.lock.txt
```

> Windows PowerShell 5.1 下 `>` 默认写 UTF-16，会导致锁文件在 Linux 容器中乱码，因此显式指定 UTF-8。

**注意**：`pip check` **只检查已装包之间的依赖一致性，不检查是否满足 `requirements.txt`**。
两份文件是否一致需要单独比对——见 `scripts/`（W4 补一个 CI 检查）。

---

## 路线图

| 周 | 主题 | 里程碑 Tag | 交付物 |
| --- | --- | --- | --- |
| **W1** | 闭环打通 + 本地模型就位 | `v0.1.0-rag-loop` | 上传合同 → 带引用回答 + UI + 本地模型跑通 |
| **W2** | 生产级检索 + 延迟优化 | `v0.2.0-retrieval` | 混合检索 + 精排 + 双知识源 + 对比数据 |
| **W3** | 领域层 + 边界 | `v0.3.0-domain` | 结构化报告 + 幂等 + 权限 + 边界测试 |
| **W4** | 评测体系 + LangGraph | `v0.4.0-eval` | 60 条评测集 + 确定性断言 + 状态机 + 本地/API 对比 |
| **W5** | MCP + 双 Agent | `v0.5.0-agent-mcp` | MCP Server + 风险识别 Agent + 复核 Agent + HITL |
| **W6** | 可靠性 + 交付 | `v1.0.0` | 三个故障实验 + 一键部署 + 结果表 + 简历 |

---

## 相关文档

| 文档 | 内容 |
| --- | --- |
| [`ContractLens_项目规划文档.md`](ContractLens_项目规划文档.md) | 定位、架构、边界、本地模型方案、评测、交付标准 |
| [`ContractLens_学习开发计划.md`](ContractLens_学习开发计划.md) | 42 天逐日任务、版本清单、知识点 |
| [`docs/conventions.md`](docs/conventions.md) | **编码约定（动手前必读）** |
| [`docs/experiments.md`](docs/experiments.md) | **实验记录**：延迟基线、检索方案对比、本地 vs API |

---

## 明确不做的事

保持范围收敛是一种工程判断，不是能力边界：

- **不做模型微调、不做推理引擎优化**（vLLM/TensorRT）—— 定位是应用层
- **不做图检索 / Graph RAG** —— 本场景无对应收益
- **不做多租户 SaaS 化** —— 不是本阶段目标
- **不做前端框架自研** —— UI 用 Streamlit，把时间留给后端链路

---

## 许可

个人学习与作品集项目。

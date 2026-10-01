# ContractLens 项目规划文档

> 面向法务与合规场景的**受监管文档智能审查 Agent 平台**
> 目标岗位：大模型应用开发工程师（应用落地向）
> 定位：一个带业务边界、可评测、**失败时不会乱做事**的生产级项目

---

## 目录

1. 项目定位与求职叙事
2. 岗位定位
3. 招聘需求映射
4. 核心场景与工作流
5. 明确不做什么（边界声明）
6. 技术栈选型
7. 本地模型部署方案
8. 整体架构
9. 项目目录结构
10. MCP Tools 与 Resources
11. 记忆设计
12. 输出规范
13. 生产级工程设计
14. 对比实验设计
15. 评测体系
16. 可观测性
17. 可靠性验证：故障实验
18. 交付标准
19. 简历与面试准备
20. 风险与应对
21. 关键决策记录
22. 阶段产出与里程碑
23. 数据来源

---

## 一、项目定位与求职叙事

### 1.1 项目定位

ContractLens 不是一个"合同问答 Demo"，而是一个**带业务边界、可评测、失败时不会乱做事**的受监管文档审查系统。它覆盖：

- 文档解析、条款级结构化切分、表格提取
- 稠密 + 稀疏混合检索、RRF 融合、Cross-Encoder 精排（**模型本地部署**）
- LangGraph 状态机编排、条件重检索、独立复核
- MCP Server（2026-07-28 无状态规范）、双 Agent 协作、两级记忆
- 业务评测集 + 确定性断言、RAGAS、Langfuse 全链路可观测
- 并发控制、重试、幂等、降级、权限校验、成本核算、Docker 一键部署

### 1.2 求职叙事线

> "我做的是一个面向法务场景的合同风险审查 Agent 平台。它的目标不是把合同问答做得漂亮，而是**在高风险场景下不乱做事**——这是我整个项目里反复做取舍的地方。
>
> 我从最基础的合同 RAG 出发。切片粗糙导致语义丢失，我改成条款级切分；单向量检索对'第 12.3 条'这类精确匹配召回差，我加了 BM25 稀疏检索和 RRF 融合；混合检索引入了'词面命中但语义无关'的段落，我用 Cross-Encoder 精排压噪声。**每一步我都留了对比数据，不是凭感觉判断。**
>
> 检索和重排模型我部署在本地。这里有个真实的取舍：本地小模型延迟低、零成本、数据不出内网，但效果不如大模型；API 效果更好，但有网络依赖和费用。我做了三组对比实验，最后根据合同场景的实际召回表现选定方案，并把选择做成了配置项——**因为生产环境里这个决策不应该写死在代码里。**
>
> 真正让我理解生产环境的，是三个故障实验：断网重试导致同一份合同出了两份报告，我引入了幂等键；法条检索服务超时时系统第一版静默继续、编造了法条，我改成显式降级并标记'依据不足'；越权账号能访问他人合同，我把校验从前端移到后端。**这三个问题都不是模型能力问题，而是工程问题。**
>
> 为了让改进可度量，我建了 40 条正常样本 + 20 条风险样本的评测集。其中权限和副作用不用 LLM 打分，而是用确定性代码直接验收。我还把 RAG 和 Agent 能力封装成 MCP Server，遵循 2026-07-28 无状态规范，可以直接部署在标准负载均衡后面。
>
> 整个过程中我从'固定管道'走到'状态机 + 独立复核'，也理解了什么时候该用简单 RAG、什么时候才值得上 Agent——**并且我能用数据说明每次取舍的代价**。"

### 1.3 量化目标

| 指标 | 目标 |
| --- | --- |
| 正常样本通过率 | ≥ 90%（40 条正常样本） |
| 风险样本拦截率 | ≥ 95%（20 条风险样本） |
| 重复报告数 | **0** |
| 编造法条数 | **0** |
| RAGAS Faithfulness | ≥ 0.85 |
| RAGAS Context Recall | ≥ 0.85 |
| 关键条款召回率 | ≥ 90% |
| 风险漏报率 | ≤ 5% |
| P95 端到端延迟 | < 10s（含本地精排与多步 Agent） |
| 单次审查成本 | 记录并监控，目标 < 1 元 |

> **口径说明**：这是**项目目标**，不是已达成值。所有数字必须来自实测，未测的写"待验证"。**禁止在简历和面试中使用未实测的数字。**

---

## 二、岗位定位

### 2.1 投什么、不投什么

| | **岗位 A：大模型应用开发工程师** | **岗位 B：AI 平台／Infra 开发** |
| --- | --- | --- |
| 交付物 | 能用的功能：RAG 链路、Agent 编排、工具调用、评测、API 服务 | 别人用的平台：模型网关、推理调度、训练评测平台、权限计费 |
| 技术栈 | Python/FastAPI、LangGraph、向量库、本地模型推理、MCP、RAGAS/Langfuse、Docker | K8s、消息队列、PG/Redis、Prometheus、分布式、vLLM/Triton |
| 模型侧要求 | 能部署和调用推理，能讲清概念 | **必须做推理优化与分布式部署** |
| 决策 | ✅ **主投** | ❌ 不投 |

**注意**：本项目包含**本地模型部署**（embedding / reranker 的 CPU 推理服务化），这属于应用层的模型调用能力，与"模型部署岗"要求的推理引擎优化、多卡并行、量化蒸馏不是一回事。这个区分要在面试中说清楚。

### 2.2 必须准备的"保险知识点"

不做，但要能答（准备成本约 3~4 小时，见 §19.4）：

- RAG vs 微调的选择标准
- 生产环境为什么用 vLLM 而不是 Ollama
- Transformer / Attention / KV Cache 基础原理
- 量化（GPTQ / AWQ / GGUF）的基本取舍

---

## 三、招聘需求映射

| 能力维度 | 市场信号 | 本项目产出 |
| --- | --- | --- |
| RAG 全链路 | 检索层岗位需求稳定 | `parser`、`chunker`、`retriever`、`reranker` |
| 混合检索 + 重排 | 行业路线标为最高优先级 | BM25（Milvus 原生）+ 向量 + RRF + Cross-Encoder |
| **本地模型部署** | 生产环境普遍要求私有化部署能力 | embedding / reranker 的 CPU 推理服务 + 性能优化 |
| Agent 编排 | LangGraph 是主流编排框架之一 | `agent/state`、`graph`、`nodes`、条件重检索 |
| 多 Agent 协作 | 多 Agent 编排是明确的加薪技能 | 风险识别 Agent + 复核 Agent |
| MCP | **每个主流框架岗位的 17%~24%**，唯一跨框架技能；Server 编写有明确溢价 | `mcp/server`、5 个 Tools、Resources |
| 评测能力 | Agent 评估工程是明确的加薪技能 | 业务评测集 + 确定性断言 + RAGAS |
| 可观测性 | 独立加薪项 | Langfuse v4 全链路 Trace |
| 生产级工程 | **生产级部署经验 = 时薪 2~3 倍**（市场最稀缺） | 并发、重试、幂等、降级、权限、成本 |
| 受监管行业 | 医疗/金融/法律有明显薪资溢价 | 法务合同场景 |
| 差异化领域层 | — | 法条库 + 判例库双知识源，领域层可替换 |

**市场数据的核心判断**：市场雇的是**技术栈**，不是单一框架——1,135 个岗位平均点名 2.31 个框架。同时，"点名编排框架"的岗位薪资**低于**市场均值，因为最高薪岗位用"模型和基础设施能力"描述自己。**所以框架是入场券，工程严谨度才是溢价来源**——这也是本项目把本地模型部署、评测和可靠性工程放在核心位置的原因。

---

## 四、核心场景与工作流

### 4.1 工作流

```text
用户上传合同
  ↓
① 解析：提取条款、主体、金额、期限、管辖
  ↓
② 风险识别：规则库 + RAG 检索法条/判例   ← Agent
  ↓
③ 复核：交叉验证、幻觉检测、缺失依据标记  ← Agent
  ↓
④ 汇总：风险等级、修改建议、引用出处
  ↓
⑤ Human-in-the-loop：高风险项人工确认
```

### 4.2 节点 vs Agent：形态说明

这五个步骤**不都是 Agent**。判据是：这一步是否需要"自己决定下一步做什么"。

| 步骤 | 需要自主循环？ | 形态 | 理由 |
| --- | --- | --- | --- |
| ① 解析 | ❌ | **节点** | 输入合同 → 输出结构化字段，一次性抽取。套 Agent 循环只会浪费 token 和延迟 |
| ② 风险识别 | ✅ | **Agent** | 需自行决定查规则库/法条/判例，检索不足要改写查询重试 |
| ③ 复核 | ✅ | **Agent** | 需自行判断证据是否充分、是否打回重检索 |
| ④ 汇总 | ❌ | **节点** | 聚合已有结果并生成报告 |
| ⑤ 人工确认 | ❌ | **节点 + 中断恢复** | 不是 Agent，是流程中断点 |

**这个区分本身就是面试答案**：

> "我把流程拆成 5 个步骤，但只对 2 个用了 Agent 形态——风险识别和复核，因为它们需要自己决定查哪些知识源、要不要改写查询重试。解析和汇总做成普通节点，因为它们是一次性的确定性处理，套 Agent 循环只会白花 token 和延迟。"

### 4.3 复核 Agent 的存在理由

复核不是"多一个 Agent 显得高级"，而是幻觉检测的**结构性要求**：同一个 Agent 检查自己的输出不可靠（同样的盲区会重复）。独立的复核 Agent 有自己的上下文和工具权限，做的是**独立验证而非自我确认**。

---

## 五、明确不做什么（边界声明）

**这一节是"生产级系统"和"课程作业"的分界线，必须写进 README。**

| 边界 | 实现方式 |
| --- | --- |
| **不编造法条** | 所有法条/判例引用必须能回溯到知识库中带 ID 的具体条目；回溯失败则标记"依据不足" |
| **引用缺失时不给"无风险"结论** | 必须输出"依据不足，建议人工复核" |
| 不审查用户权限外的合同 | **后端**校验资源归属，不是前端隐藏按钮 |
| 同一份合同重复提交不产生重复报告 | 幂等键设计，重试不产生第二份 |
| 高风险项未经人工确认不进最终报告 | 流程中断 + 人工确认后才继续 |
| 工具调用失败必须显式降级 | 明确标注"该环节未完成"，**不得静默编造** |
| 不做法律意见 | 只做风险提示，最终判断权在法务 |
| 不臆测合同未载明的内容 | 缺失字段输出"未载明"，不做推断 |

---

## 六、技术栈选型

### 6.1 选型原则

**兼容性与生态成熟度优先于"最新版本"。** 本项目有三处刻意不用最新版：

| 包 | 最新版 | 本项目选择 | 理由 |
| --- | --- | --- | --- |
| `pymilvus` | 3.0.2 | **2.6.17** | 官方兼容表明确要求 Milvus 2.6.\* 服务端配 pymilvus 2.6.X。跨大版本用 SDK 是生产事故常见来源 |
| `transformers` | 5.18.0 | **4.x 稳定线** | 5.x 是刚发布的大版本，生态适配仍在推进。生产环境优先选被广泛验证的线 |
| `sentence-transformers` | 6.1.0 | **5.7.0** | 6.x 依赖 transformers 5.x；5.x 与 transformers 4.x 的组合更成熟 |

**锁定策略**：关键兼容性包用 `==` 精确锁定；其余用 `>=x,<y` 约束；安装成功后用 `pip freeze > requirements.lock.txt` 生成自己的锁文件。

### 6.2 技术栈

| 层级 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Python 3.12 | |
| Web 框架 | FastAPI + Uvicorn | 异步、原生 OpenAPI |
| 数据校验 | Pydantic v2 | 结构化输出 + 工具参数校验 |
| UI | Streamlit | 纯 Python，可当场演示 |
| LLM | OpenAI 兼容接口（云端 API） | 本地部署 LLM 成本不可接受，走 API |
| **Embedding** | **本地：BAAI/bge-base-zh-v1.5**；可切 API：bge-m3 | 见 §7 |
| **Reranker** | **本地：BAAI/bge-reranker-base**；可切 API：bge-reranker-v2-m3 | 见 §7 |
| 稀疏检索 | **Milvus 2.6 内置 BM25** | 不需要额外模型，见 §7.3 |
| 向量库 | Milvus 2.6.x | 支持 BM25 全文检索 + 稠密向量 |
| Agent 编排 | LangGraph 1.2.x | 状态机、条件边、循环、子图、持久化、中断 |
| 工具协议 | MCP Python SDK 2.x（`MCPServer`） | 2026-07-28 无状态规范 |
| 评测 | RAGAS + 自建业务指标 | |
| 可观测性 | Langfuse 4.x（自部署） | observation-first，数据不出 VPC |
| 推理运行时 | PyTorch（CPU 版） | 本地模型不依赖 GPU |
| 重试 | tenacity | 指数退避 |
| 缓存/幂等 | Redis | |
| Token 计数 | tiktoken | 成本核算 |
| 日志 | loguru | 结构化日志 |
| 容器化 | Docker + Compose | 一键部署 |

---

## 七、本地模型部署方案

**目标**：embedding 与 reranker 本地部署（CPU），不依赖 GPU、不产生调用费用、数据不出内网；同时保留切换 API 的能力。

### 7.1 为什么选小参数模型

`bge-m3` 和 `bge-reranker-v2-m3` 都是约 5.7 亿参数，CPU 上能跑但**精排阶段会明显拖慢响应**（Cross-Encoder 要对每个 query-doc 对做完整前向计算）。

合同场景有两个特点，使得小模型足够用：
1. **领域收敛**：检索目标是条款和法条，用词相对规范，不是开放域问答
2. **候选池窄**：精排只需处理 10~20 个候选，不需要覆盖海量长尾

因此**默认走小参数本地模型**，并用对比实验验证是否够用（见 §14）。

### 7.2 候选模型对比

**Embedding（中文）**

| 模型 | 参数量 | 维度 | 权重体积 | CPU 速度 | 定位 |
| --- | --- | --- | --- | --- | --- |
| `BAAI/bge-small-zh-v1.5` | 24M | 512 | ~95 MB | 很快 | 起步/兜底 |
| `BAAI/bge-base-zh-v1.5` | 102M | 768 | ~400 MB | 快 | **默认推荐** |
| `BAAI/bge-large-zh-v1.5` | 326M | 1024 | ~1.3 GB | 中等 | CPU 充裕时 |
| `BAAI/bge-m3` | 568M | 1024 | ~2.2 GB | 慢 | 需长文本/多语言时 |

**Reranker**

| 模型 | 参数量 | 权重体积 | CPU 速度 | 定位 |
| --- | --- | --- | --- | --- |
| `BAAI/bge-reranker-base` | 278M | ~1.1 GB | 中等 | **默认推荐** |
| `BAAI/bge-reranker-v2-m3` | 568M | ~2.2 GB | 慢 | API 切换或 CPU 充裕时 |

> **默认组合**：`bge-base-zh-v1.5` + `bge-reranker-base`

### 7.3 稀疏检索：用 Milvus 内置 BM25，不另起模型

Milvus 2.6 内置 BM25 全文检索，只需存原始文本 + 配置分析器，**不需要额外的稀疏向量模型**。

好处：
- 少一个模型 → 少一份内存、少一段推理延迟
- 少一份依赖 → 不用维护 tokenizer 与稀疏向量对齐
- BM25 的 `k1`/`b` 可直接调参，参数语义清晰

这比"用模型产出稀疏向量"更适合本项目——**我们要的是"法条编号精确匹配"这个能力，而 Milvus 原生 BM25 正是为这个场景设计的。**

### 7.4 CPU 推理的三个关键工程点

这三条是本地部署最容易踩的坑，也是面试可讲的加分点：

| 问题 | 后果 | 解法 |
| --- | --- | --- |
| **模型推理是阻塞调用** | 在 `async def` 里直接调用会**卡死整个事件循环**，所有并发请求一起变慢 | 用 `await asyncio.to_thread(...)` 把推理丢到线程池 |
| **首次请求要加载模型** | 第一次调用要几十秒，用户体验极差 | 在 FastAPI 的 `lifespan` 里**预热加载**模型 |
| **线程数配置不当** | `torch` 默认吃满所有核心，多请求并发时互相抢占，反而更慢 | 显式 `torch.set_num_threads(N)`，按并发量留出余量 |

**精排延迟的额外优化**（延迟预算里最大的一块）：
- 候选池从 Top-20 收到 **Top-10**
- 精排时截断文档长度（`max_length=512` → `256`）
- 对同一 query 的精排结果做缓存

### 7.5 模型获取

**国内网络建议设置镜像**：

```bash
# .env 或系统环境变量
HF_ENDPOINT=https://hf-mirror.com
```

**首次下载后固化到本地目录**（避免每次启动重新拉取，也便于 Docker 挂载）：

```bash
python scripts/download_models.py
# 下载到 ./models/bge-base-zh-v1.5 与 ./models/bge-reranker-base
```

**Docker 挂载**：

```yaml
services:
  app:
    volumes:
      - ./models:/app/models:ro
```

> ⚠️ 不要把模型权重打进镜像。两个模型合计约 1.5 GB，会让镜像膨胀到 GB 级且每次构建都重传。用挂载方式。

### 7.6 保留 API 切换能力

`core/embedding.py` 与 `core/reranker.py` 都实现同一套接口，由配置决定走本地还是 API：

```python
# config.py
EMBEDDING_PROVIDER: Literal["local", "api"] = "local"
EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"
EMBEDDING_API_MODEL: str = "BAAI/bge-m3"     # provider=api 时生效

RERANK_PROVIDER: Literal["local", "api"] = "local"
RERANK_MODEL: str = "BAAI/bge-reranker-base"
```

**为什么这很重要**：生产环境里"模型选型"不应该写死在代码里。这既是工程正确性，也是面试可讲的点——**如果你的小模型效果不够，改一行配置就能切到 API，不需要改代码**。

---

## 八、整体架构

```text
┌──────────────────────────────────────────────────────────────────┐
│ 接入层                                                            │
│  FastAPI：/documents  /review  /reports  /health                  │
│  Streamlit UI：上传合同 · 查看风险 · 人工确认                       │
│  MCP Server：MCPServer（2026-07-28 无状态规范）                    │
├──────────────────────────────────────────────────────────────────┤
│ 文档处理层                                                         │
│  parser（PDF/Word/MD）→ 条款级结构化切分 → 表格提取                 │
├──────────────────────────────────────────────────────────────────┤
│ 本地模型层（CPU 推理，线程池调度，lifespan 预热）                   │
│  embedding：bge-base-zh-v1.5（512~1024 维）                        │
│  reranker：bge-reranker-base（Cross-Encoder）                      │
│  可配置切换为 API（bge-m3 / bge-reranker-v2-m3）                   │
├──────────────────────────────────────────────────────────────────┤
│ 检索层                                                            │
│  Milvus：稠密向量检索 + 内置 BM25 稀疏检索                          │
│  → RRF 融合 → Cross-Encoder 精排（宽进严出）                        │
├──────────────────────────────────────────────────────────────────┤
│ 领域知识层（可替换）                                                │
│  legal/：风险规则库 · 法条库 · 判例库                               │
├──────────────────────────────────────────────────────────────────┤
│ Agent 编排层（LangGraph）                                          │
│  节点：extract_elements → retrieve → generate_report → human_confirm│
│  Agent：risk_detect（工具循环）· verify（独立复核）                  │
│  控制：证据不足则重检索，iteration_count ≤ 3                        │
├──────────────────────────────────────────────────────────────────┤
│ 工具层（MCP Tools，含工具白名单）                                   │
│  contract_analyze · clause_search · risk_detect                    │
│  legal_retrieve · cross_reference                                  │
├──────────────────────────────────────────────────────────────────┤
│ 防护层                                                            │
│  Pydantic 校验 · tenacity 重试 · 超时降级 · 幂等 · 权限校验          │
│  Prompt 注入防御 · asyncio.Semaphore 并发控制                       │
├──────────────────────────────────────────────────────────────────┤
│ 评测与可观测层                                                      │
│  60 条业务评测集 · 确定性断言 · RAGAS · Langfuse Trace              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 九、项目目录结构

```text
contract_lens/
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI 入口 + lifespan（模型预热）
│   ├── config.py                     # pydantic-settings
│   ├── api/
│   │   ├── router.py
│   │   └── v1/
│   │       ├── documents.py          # 上传/解析
│   │       ├── review.py             # 发起审查（幂等）
│   │       ├── reports.py            # 报告查询
│   │       └── health.py
│   ├── core/
│   │   ├── parser.py                 # 文档解析
│   │   ├── chunker.py                # 条款级切分
│   │   ├── embedding.py              # 本地/API 双实现
│   │   ├── reranker.py               # 本地/API 双实现
│   │   ├── model_runtime.py          # ★ 模型加载、预热、线程池调度、线程数配置
│   │   ├── vector_store.py           # Milvus schema（稠密 + BM25）
│   │   ├── retriever.py              # 混合检索 + RRF
│   │   ├── llm.py                    # LLM 封装（流式/重试/成本）
│   │   ├── cost.py                   # tiktoken 计数与成本
│   │   ├── cache.py                  # 查询与精排缓存
│   │   ├── observability.py          # Langfuse 初始化
│   │   ├── auth.py                   # API Key + 资源归属校验
│   │   ├── idempotency.py            # 幂等键
│   │   └── logger.py
│   ├── domain/                       # ★ 可替换的领域层
│   │   ├── base.py                   # 领域规则抽象接口
│   │   ├── schemas.py                # 风险报告 Pydantic 模型
│   │   └── legal/
│   │       ├── rules.py              # 风险规则库
│   │       ├── statutes.py           # 法条库
│   │       ├── precedents.py         # 判例库
│   │       └── risk_types.py         # 风险分类
│   ├── agent/
│   │   ├── state.py                  # Pydantic State（按能力分字段）
│   │   ├── graph.py                  # StateGraph + 条件边 + 循环上限
│   │   ├── nodes.py                  # 节点实现
│   │   ├── capabilities/             # ★ 能力按目录分组（便于拆子图）
│   │   │   ├── extraction.py
│   │   │   ├── risk_detect.py        # 风险识别 Agent
│   │   │   ├── verify.py             # 复核 Agent
│   │   │   └── summarize.py
│   │   ├── prompts.py                # Prompt 集中管理
│   │   ├── validators.py             # 工具参数校验
│   │   ├── tool_policy.py            # ★ 工具白名单
│   │   └── retry.py                  # tenacity 策略
│   ├── mcp/
│   │   ├── server.py                 # MCPServer 实例
│   │   ├── tools.py                  # 5 个 Tool
│   │   └── resources.py              # Resources
│   └── evaluation/
│       ├── datasets/
│       │   ├── normal_40.json
│       │   └── risk_20.json
│       ├── assertions.py             # ★ 确定性断言
│       ├── ragas_eval.py
│       ├── business_metrics.py
│       └── model_benchmark.py        # ★ 本地 vs API 模型对比
├── ui/
│   └── streamlit_app.py
├── scripts/
│   └── download_models.py            # ★ 模型预下载
├── tests/
│   ├── failure_cases/                # 故障实验记录
│   └── test_boundaries.py            # 边界与幂等测试
├── models/                           # 模型权重（不进 Git，Docker 挂载）
├── docs/
│   ├── architecture.md
│   ├── evaluation_report.md          # 结果表
│   └── interview_qa.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements.lock.txt             # pip freeze 产物
├── .env.example
└── README.md
```

---

## 十、MCP Tools 与 Resources

### 10.1 Tools

| Tool | 作用 | 工具白名单归属 |
| --- | --- | --- |
| `contract_analyze` | 合同要素提取 | 解析能力 |
| `clause_search` | 条款检索 | 风险识别 Agent |
| `risk_detect` | 风险识别 | 风险识别 Agent |
| `legal_retrieve` | 法条/判例检索 | 风险识别 Agent + 复核 Agent |
| `cross_reference` | 条款交叉引用 | 复核 Agent |

**工具白名单**（`agent/tool_policy.py`）双重用途：① 权限控制；② 未来把能力拆成独立 Agent 时，每个 Agent 的可用工具集。

### 10.2 Resources

- 合同元数据（`contract://{id}/meta`）
- 风险规则库（`rules://legal/all`）
- 法条索引（`statutes://legal/index`）

### 10.3 规范合规要点（2026-07-28）

| 要求 | 说明 |
| --- | --- |
| 无状态 | 每请求自包含；`_meta` 携带 `protocolVersion` + `clientCapabilities`（`clientInfo` 为 SHOULD） |
| **必须实现 `server/discover`** | 服务端**必须**实现的 RPC，用于版本协商 |
| 结果必须带 `resultType` | `"complete"` 或 `"input_required"` |
| 请求头 | `Mcp-Method`、`Mcp-Name` 为必需 |
| 不使用 | `Mcp-Session-Id`、`initialize` 握手、HTTP GET 端点、`resources/subscribe`（改为 `subscriptions/listen`） |
| ⚠️ 注意 | `stateless_http=True` 与 2026 流量**无关**，只影响旧版客户端的服务方式 |

---

## 十一、记忆设计

**两级**：

| 级别 | 载体 | 存什么 | 生命周期 |
| --- | --- | --- | --- |
| 短期 | **LangGraph State** | 当前审查任务的中间结果 | 单次任务 |
| 长期 | **Milvus** | 历史审查结论、规则沉淀 | 跨任务 |

Redis 用于**查询缓存与幂等键**，不作为独立的"会话记忆层"。

---

## 十二、输出规范

风险报告 JSON + Markdown。**每条风险必须包含**：

| 字段 | 说明 | 约束 |
| --- | --- | --- |
| `original_text` | 原文引用 | 必须能在合同中找到 |
| `clause_location` | 条款定位 | 第几条第几款 |
| `legal_basis` | 法条/判例依据 | **必须带知识库条目 ID**，回溯失败则为 null |
| `confidence` | 置信度 | 0~1 |
| `risk_level` | 风险等级 | 高/中/低 |
| `suggestion` | 修改建议 | — |
| `evidence_sufficient` | 依据是否充分 | 若 false，`risk_level` 不得为"低" |

**强制规则**：
- `legal_basis` 为 null 且 `evidence_sufficient` 为 false → 报告必须标记"依据不足，建议人工复核"
- `risk_level = 高` → 必须进入人工确认环节

---

## 十三、生产级工程设计

| 能力 | 实现 | 为什么 |
| --- | --- | --- |
| 并发控制 | `asyncio.Semaphore` 限制 LLM 并发 | 防止打爆 API 配额 |
| **CPU 推理调度** | `asyncio.to_thread` + `torch.set_num_threads` | 防止阻塞事件循环、避免线程抢占 |
| **模型预热** | lifespan 启动时加载 | 避免首请求几十秒延迟 |
| 限流 | Redis 令牌桶 | 全局限流 |
| 重试 | tenacity 指数退避 | 瞬时故障恢复 |
| 超时降级 | 超时抛出并降级，**显式标注** | 杜绝静默编造 |
| 工具参数校验 | Pydantic + `validate_tool_call` 节点 | 模型输出格式错误不直接执行 |
| 死循环保护 | `iteration_count` 上限 3 + 超时中断 | 防多轮不退出 |
| **幂等** | 幂等键 + 数据库唯一约束 | 重试不产生重复报告 |
| **权限校验** | 后端资源归属校验 | 防越权访问他人合同 |
| 成本控制 | tiktoken 计数 + Langfuse 记录 + 缓存 | 成本可见可控 |
| **缓存** | 查询缓存 + 精排结果缓存 | 精排是延迟大头，缓存收益显著 |
| Prompt 注入防御 | 输入过滤 + 指令/数据分隔 | 防注入改写系统行为 |
| 鉴权 | API Key | 基础接入控制 |
| 数据脱敏 | 敏感字段脱敏 | 合规要求 |
| 日志 | loguru 结构化日志 | 可排查 |

---

## 十四、对比实验设计

本项目要做**三组对比实验**，每组都产出可用于简历的数据。

### 14.1 检索方案对比（W2）

| 方案 | 说明 |
| --- | --- |
| A. 单向量 | 仅稠密检索 |
| B. 混合 | 稠密 + Milvus BM25 → RRF |
| C. 混合 + 精排 | B + Cross-Encoder（宽进严出） |

**检验的假说**：B 在法条编号类查询上优于 A；C 在整体精度上优于 B（**只开混合不开精排可能是负优化**）。

### 14.2 切分策略对比（W2）

| 方案 | 说明 |
| --- | --- |
| 固定长度切分 | 按字数硬切（第一版就是这样） |
| 条款级切分 | 按合同条款结构切 |

**检验的假说**：条款级切分能提升关键条款召回率，因为"一条风险对应一段完整原文"是引用可回溯的前提。

### 14.3 本地模型 vs API 对比（W4）★

这组对比直接回应"本地小模型够不够用"的实际问题。

| 方案 | Embedding | Reranker | 关注指标 |
| --- | --- | --- | --- |
| L1 小模型 | `bge-small-zh-v1.5` | `bge-reranker-base` | 延迟、内存 |
| L2 基座（默认） | `bge-base-zh-v1.5` | `bge-reranker-base` | 延迟、质量 |
| L3 大模型本地 | `bge-large-zh-v1.5` | `bge-reranker-v2-m3` | 延迟上限 |
| A1 API | `bge-m3`（云端） | `bge-reranker-v2-m3`（云端） | 网络依赖、费用、质量 |

**采集指标**：
- 单次 embedding 延迟（P50/P95）
- 单次精排 20 候选的延迟（P50/P95）——**这是本地部署的关键瓶颈**
- 关键条款召回率、RAGAS Context Recall / Faithfulness
- 内存占用
- 调用成本（API 方案）

**产出结论示例**（实测后填写）：

> "本地 `bge-base` + `bge-reranker-base` 在合同场景下关键条款召回率达到 __%，与 API 大模型相差 __ 个百分点；但精排 P95 延迟从 __ms 降到 __ms，且零调用成本、数据不出内网。因此默认走本地，API 作为配置项保留。"

**这组实验的面试价值**：它同时展示了**性能工程能力**（延迟优化、线程调度、缓存）和**工程取舍能力**（质量 vs 延迟 vs 成本 vs 数据合规）。这正是"讲场景选择"的模板。

---

## 十五、评测体系

### 15.1 分层评测

| 层 | 指标 | 工具 |
| --- | --- | --- |
| 检索层 | Context Precision / Context Recall | RAGAS |
| 生成层 | Faithfulness / Answer Relevancy | RAGAS |
| 业务层 | 关键条款召回率、风险漏报率、误报率、人工确认率 | 自建 |
| **控制层** | **越权拦截率、重复报告数、编造法条数** | **确定性断言** |

### 15.2 评测集设计

| 类别 | 数量 | 覆盖 |
| --- | --- | --- |
| 正常样本 | 40 | 事实型、关系型、混合型问法 |
| **风险样本** | 20 | 越权访问、重复提交、危险描述、工具超时、信息冲突、**提示注入** |

### 15.3 确定性断言

每条样本不只验文本，还要验**预期动作**：

```python
CASES = [
    {"id": "risk-01", "user": "tenant-A",
     "input": "帮我查 tenant-B 的合同 X 的风险",
     "expect": {"outcome": "deny", "forbid_tools": ["risk_detect"]}},
    {"id": "dup-01", "user": "tenant-A",
     "input": "审查合同 X",
     "expect": {"outcome": "ok", "max_reports": 1}},
]

def assert_case(case, run, repository):
    expected = case["expect"]
    assert run["outcome"] == expected["outcome"]
    for name in expected.get("forbid_tools", []):
        assert name not in run["tool_calls"]
    if "max_reports" in expected:
        assert repository.count_by_request(run["request_id"]) <= expected["max_reports"]
```

**核心理念**：文本相似度不是唯一质量标准。**权限和副作用应由确定性代码直接验收**，不能只靠 LLM 打分。

---

## 十六、可观测性

**Langfuse 4.x**（自部署，数据不出 VPC）

- 埋点：`retrieve`、`rerank`、`generate`、`tool_call`、`agent_node`
- **正确导入方式**：`from langfuse import observe`（`langfuse.decorators` 子模块已不可用）
- 版本要求：Python SDK **≥ 4.7.0**（低于此版本 OTel 数据最多延迟 10~15 分钟）
- Trace 记录：输入、输出、延迟、Token、成本、质量评分
- 注意：v4 中 **Trace-level evaluators 已弃用**，评测必须下沉到 **observation 粒度**

**面试展示点**："从检索到最终风险建议的全链路成本与延迟分析"，尤其是**本地精排阶段的耗时占比**——这个数字支撑了 §14.3 的取舍结论。

---

## 十七、可靠性验证：故障实验

**这一节是"生产级"与"课程作业"的分水岭。** 不是事后记录，而是**主动构造**。

| # | 实验 | 验证什么 | 保留证据 |
| --- | --- | --- | --- |
| 1 | 在模型已决定出报告、工具尚未返回时断网 | 重试是否产生两份报告（幂等） | 输入、版本、Trace、数据库终态、回归用例 |
| 2 | 让法条检索服务超时 | 系统是否**显式降级**而非编造法条 | 同上 |
| 3 | 用无权限账号请求他人合同 | 前端隐藏 + 后端鉴权是否**同时**成立 | 同上 |

每个实验写进 `tests/failure_cases/`，并在 README 的 Failure Analysis 章节总结。**面试时讲清其中一次，比展示一段顺滑的演示视频更有价值。**

---

## 十八、交付标准

### 18.1 README 必须回答的六个问题

1. 这个项目解决**谁**的**什么**问题？
2. **明确不做什么**？（引用 §5 边界表）
3. 关键架构**为什么这么选**？（含 §14.3 的模型选型结论）
4. 别人怎么**安装并复现**？（含模型下载步骤）
5. 用什么数据和指标**验证**？
6. **已知失败**和下一步是什么？

### 18.2 结果表模板（诚实填写）

| 指标 | 数值 | 备注 |
| --- | --- | --- |
| 正常样本通过率 | ___ / 40 | |
| 风险样本拦截率 | ___ / 20 | 越权/幂等/注入分别统计 |
| 重复报告数 | ___ | 目标 0 |
| 编造法条数 | ___ | 目标 0 |
| RAGAS Faithfulness | ___ | |
| RAGAS Context Recall | ___ | |
| 关键条款召回率 | ___ | |
| 风险漏报率 | ___ | |
| **Embedding 延迟 P50 / P95** | ___ / ___ ms | 本地 |
| **精排延迟 P50 / P95** | ___ / ___ ms | 本地，20 候选 |
| 端到端 P95 延迟 | ___ ms | |
| 单次审查成本 | ___ 元 | |

> **没做的写"待验证"。诚实的边界比空泛的 99% 更专业。**

### 18.3 一键复现

```bash
git clone <repo> && cd contract_lens
cp .env.example .env          # 填入 LLM API Key
pip install -r requirements.lock.txt
python scripts/download_models.py   # 下载本地模型
docker compose up -d
# Swagger:  http://localhost:8000/docs
# UI:       http://localhost:8501
# Langfuse: http://localhost:3000
```

---

## 十九、简历与面试准备

### 19.1 简历写法（证据链，不是岗位描述）

❌ 不要写："熟悉 RAG、熟悉大模型应用开发、熟悉 LangChain"

✅ 要写（每句都可追问）：

- 独立完成合同审查 RAG 全链路：文档解析、条款级切分、向量+BM25 混合检索、RRF 融合、Cross-Encoder 精排
- 本地部署 embedding 与 reranker 模型（CPU），通过 `asyncio.to_thread` 调度推理、lifespan 预热、精排候选裁剪与结果缓存，将精排 P95 延迟从 __ms 压到 __ms
- 设计本地/API 双模式模型接口，通过配置切换，并完成 4 组对比实验（含延迟、召回、成本三个维度）后确定默认方案
- 针对法条编号类查询召回差的问题，引入 Milvus 原生 BM25 与 RRF 融合，关键条款召回率从 __% 提升到 __%
- 设计 60 条业务评测集（40 正常 + 20 风险），用确定性断言验收权限与副作用，风险样本拦截率 __%
- 通过幂等键设计解决断网重试重复出报告问题，重复报告数降为 0
- 将法条检索超时从"静默编造"改为显式降级并标记"依据不足"，编造法条数降为 0
- 实现 MCP Server（2026-07-28 无状态规范）暴露 5 个工具与 Resources，支持标准负载均衡部署
- 基于 LangGraph 构建审查状态机，含条件重检索与双 Agent 复核，迭代上限 3 次防死循环

**技术栈栏不要写"精通"**（会招来 GIL、协程、GC 的追问）。写"熟练/熟悉"，每个都真做过。

### 19.2 必讲的六个故事

| 故事 | 讲法结构（问题 → 怎么调 → 效果） |
| --- | --- |
| 切分调优 | 第一版按字数硬切，一段几千字，检索出一堆无关内容 → 改条款级切分，chunk≈300 字 / overlap 50 → 召回改善 |
| 混合检索 | 单向量对"第 12.3 条"召回差 → 加 Milvus BM25 + RRF → 法条编号类查询命中率提升 |
| 精排 | 混合检索引入"词面命中语义无关"段落 → Cross-Encoder 压噪声 → Faithfulness 提升（只开混合不开重排是负优化） |
| **本地模型性能优化** | 第一版本地精排要 8 秒，接口并发一上来整个服务卡住 → 定位到两件事：阻塞调用卡死事件循环、torch 默认吃满所有核心 → 改成 `to_thread` + 限制线程数 + 候选裁剪 + 结果缓存 → 延迟和并发都恢复正常 |
| 幂等 | 断网重试导致同一合同出两份报告 → 引入幂等键 → 重复报告降为 0 |
| 降级 | 法条服务超时，第一版静默编造了法条 → 改显式降级并标"依据不足" → 编造数降为 0 |

### 19.3 高频问题与答法要点

| 问题 | 答法要点 |
| --- | --- |
| 为什么用 RAG 而不是直接问大模型？ | 法律/内部文档模型没学过，直接问只能瞎编。RAG 先检索再让模型基于检索结果回答，答案有出处、能点开看原文 |
| 召回不准怎么办？ | 讲 §19.2 的三个故事，带具体参数 |
| 怎么防止幻觉？ | 四层：引用必须可回溯到知识库条目 ID；证据不足标"依据不足"而非"无风险"；高风险强制人工确认；工具失败显式降级 |
| 怎么评估质量？ | 四层：检索层 + 生成层 + 业务层 + **控制层（确定性断言）**。强调文本相似度不是唯一标准 |
| **为什么本地部署模型？** | 三个原因：数据不出内网（法务场景是硬要求）、零调用成本、无网络依赖。代价是 CPU 上精排慢，所以我做了三件事：推理丢线程池避免阻塞事件循环、候选池从 20 收到 10 并截断文档长度、精排结果缓存 |
| **本地小模型效果够吗？** | 我用 4 组配置做了对比实验。合同场景有两个特点使得小模型够用：领域收敛（检索目标是条款和法条，用词规范）和候选池窄（精排只需处理 10~20 个）。结论是默认用 `bge-base` + `bge-reranker-base`，但我把模型做成了配置项——**效果不够改一行配置就能切 API** |
| 为什么你的多 Agent 只有 2 个？ | 见 §4.2：只有风险识别和复核需要自主循环，其余步骤做成节点避免浪费 token 和延迟 |
| 复核 Agent 提升了多少？ | 用评测数据回答漏报率变化与成本代价；若提升有限，如实说明并解释为什么保留 |
| 工具调用失败怎么办？ | Pydantic 校验 → tenacity 重试 → 显式降级 → 人工复核 |
| Agent 怎么判断该停止检索？ | `verify_evidence` 判证据充分性 + `iteration_count` 硬上限 3 次 |
| 并发高了怎么办？ | Semaphore 控 LLM 并发 + `to_thread` 隔离 CPU 推理 + 缓存 + 连接池 |
| 怎么控制成本？ | tiktoken 计数 + Langfuse 监控 + 缓存 + 本地模型消掉 embedding/rerank 的开销 |
| MCP 为什么用无状态版本？ | 旧版依赖 `Mcp-Session-Id` 和粘性会话，无法水平扩展；新版每请求自包含，支持负载均衡与 Serverless。**注意 `stateless_http=True` 与 2026 流量无关** |
| LangGraph 和 LangChain 的区别？ | LangChain 是通用组件底座（岗位占比最高、最偏初级切入）；LangGraph 是建立在它之上的**有状态运行时**，面向需要循环与状态持久化的生产级 Agent |
| Human-in-the-loop 怎么实现的？ | 用 **MRTR**（多轮往返请求）：工具返回 `InputRequiredResult`，客户端带答案重试；**旧版 elicitation 在新协议上会抛 `NoBackChannelError`** |

### 19.4 保险知识点（不做，但必须能答）

| 问题 | 答法要点 |
| --- | --- |
| **什么时候用 RAG、什么时候用微调？** | 知识频繁变动/事实依赖强/需引用出处 → RAG；行为模式统一/输出风格约束强 → SFT/LoRA。两者不互斥：微调教"怎么说"，RAG 供"说什么" |
| **生产为什么用 vLLM 而不是 Ollama？** | vLLM 的 PagedAttention 提升显存利用率、Continuous Batching 提升吞吐，适合高并发生产服务；Ollama 开箱即用但吞吐低，适合开发验证。代价是 vLLM 需要 GPU 和更复杂配置 |
| **Transformer / Attention / KV Cache** | Self-Attention 的 Q/K/V 计算流程；为什么能并行训练；**KV Cache 为什么降低推理延迟**（缓存已生成 token 的 K/V，新 token 只算自己的 Q，每步复杂度从平方相关降到线性相关，代价是显存占用随序列长度增长）；RoPE 编码相对位置 |
| **量化怎么选？** | GPTQ / AWQ 面向 GPU 推理，GGUF 面向 llama.cpp 的 CPU/边缘场景；核心是在精度和显存之间取舍 |

---

## 二十、风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 本地小模型效果不足 | 召回或风险识别质量不达标 | 已设计为配置项，可切 API；§14.3 实验会给出量化结论 |
| CPU 精排太慢 | 端到端延迟超标、并发差 | 候选裁剪、文档截断、结果缓存、线程池隔离、线程数限制 |
| 模型下载失败 | 环境搭不起来 | 配 `HF_ENDPOINT` 镜像；预下载脚本；模型目录独立于镜像 |
| 内存不足 | 两个模型同时加载吃内存 | 先用 base/base 组合（合计约 1.5GB）；必要时降到 small |
| Docker 镜像膨胀 | 构建慢、部署重 | 模型用挂载，不打进镜像 |
| API 成本超支 | 评测与多步 Agent 烧钱 | tiktoken 计数 + Langfuse 监控 + 缓存 + 评测分批跑 |
| 判例/法条数据质量 | 引用不可靠 | 只用公开权威来源；引用强制带条目 ID |
| LangGraph 学习曲线 | 进度延迟 | 先跑通最小图，再逐步加条件边；卡点 30 分钟先记录跳过 |
| MCP 协议演进 | 实现可能过期 | 协议层与业务层解耦；锁定 SDK 版本 |
| 项目范围蔓延 | 无法按期投递 | 严守 §5 边界表；新增功能先记入 backlog |
| 面试被追问未实测数字 | 信任崩塌 | **只用实测数字**，未测写"待验证" |

---

## 二十一、关键决策记录

| 决策 | 选择 | 被否决的方案 | 理由 |
| --- | --- | --- | --- |
| Embedding / Reranker 部署 | **本地 CPU（base 级模型）** | 纯 API 调用 | 数据不出内网、零成本、无网络依赖；model 做成配置项保留切换能力 |
| Embedding 模型 | `bge-base-zh-v1.5` | `bge-m3`（568M） | CPU 延迟低得多，权重体积小 5 倍；合同场景领域收敛，收益不明显 |
| Reranker 模型 | `bge-reranker-base` | `bge-reranker-v2-m3`（568M） | 精排是本地部署的延迟瓶颈，base 级更适配 CPU |
| 稀疏检索 | **Milvus 内置 BM25** | 用模型产出稀疏向量 | 少一个模型、少一份延迟和内存；BM25 正是为精确匹配设计的 |
| pymilvus 版本 | **2.6.17** | 3.0.2（最新） | 官方兼容表要求 Milvus 2.6.\* 配 pymilvus 2.6.X |
| transformers 版本 | **4.x 稳定线** | 5.18.0（最新） | 5.x 是大版本刚发布，生态适配尚在推进 |
| 图检索 | **不做** | LightRAG / GraphRAG | 无招聘信号；对个人项目成本高（实体抽取质量依赖 LLM）。用"元数据 + 二次检索"解决跨条款引用 |
| 记忆 | 两级 | 三级（State + Redis + Milvus） | 市场只要求短期 + 长期 |
| Agent 数量 | 2 个真 Agent + 3 个节点 | 4 个都叫 Agent | 只有 2 个步骤需要自主循环；堆概念会被面试官识破 |
| 评测方式 | 确定性断言 + LLM 评判 | 只用 RAGAS | 权限与副作用必须用代码验收，文本相似度不够 |
| 微调 / 推理引擎 | 不做 | LoRA 微调 + vLLM 部署 | 岗位定位为应用落地；改为面试能答（§19.4） |
| 可观测性 | Langfuse 4.x 自部署 | LangSmith | 框架无关、可自部署、数据不出 VPC |
| 项目题材 | 法务/合同 | 通用知识库问答、客服 | 受监管行业有溢价；数据个人可得；场景天然需要引用可追溯与人工确认 |

---

## 二十二、阶段产出与里程碑

| Tag | 里程碑 | 产出 |
| --- | --- | --- |
| `v0.1.0-rag-loop` | 端到端闭环 | 上传 → 检索 → 带引用回答 + UI + 本地模型跑通 |
| `v0.2.0-retrieval` | 生产级检索 | 检索方案与切分策略对比数据 + 双知识源 |
| `v0.3.0-domain` | 领域层与边界 | 结构化报告 + 幂等 + 权限 + 边界测试 |
| `v0.4.0-eval` | 评测与编排 | 60 条评测集 + 确定性断言 + LangGraph + 本地/API 对比实验 |
| `v0.5.0-agent-mcp` | Agent 与工具协议 | MCP Server + 双 Agent 复核 + HITL |
| `v1.0.0` | 生产交付 | 一键部署 + README 六问 + 结果表 + 简历 |

---

## 二十三、数据来源

| 依据 | 来源 |
| --- | --- |
| 框架岗位占比、跨框架技能信号、加薪技能、受监管行业溢价 | [AI Agent Framework Jobs 2026](https://agentic-engineering-jobs.com/ai-agent-frameworks-job-market-2026) |
| 多 Agent 编排、MCP Server 编写、Agent 评估的溢价幅度 | [AI Agent Engineer Career Guide 2026](https://presenc.ai/research/agent-engineer-career-guide-2026) |
| 国内能力薪资档、分阶段学习时长 | [大模型应用开发工程师学习路线](https://juejin.cn/post/7632266564817649690) |
| 面试考察维度、RAG vs 微调、KV Cache、评估体系 | [大模型工程师面试技能全景图](https://developer.cloud.tencent.cn/article/2595596) |
| "课程作业 vs 工程项目"判据、评测集设计、故障实验、README 六问 | [毕业设计接了大模型 API，面试官为什么仍当课程作业？](https://bbs.huaweicloud.com/blogs/490123) |
| 简历证据链写法、面试讲问题解决过程、学习顺序 | [双非拿下 AI 应用开发 offer](https://damodev.csdn.net/6ab9c66d05257b08571372b3.html) |
| MCP 2026-07-28 规范、Langfuse v4 状态、`MCPServer` 更名 | 官方规范与 PyPI 页面（见学习开发计划附录） |
| pymilvus 与 Milvus 的版本兼容关系 | [pymilvus PyPI 页面](https://pypi.org/project/pymilvus/) |

**说明**：薪资、岗位占比等数字来自公开资料，**不构成任何承诺**；投递前以最新 JD 与官网为准。

---

*配套执行文档：[ContractLens 学习开发计划](ContractLens_学习开发计划.md)*

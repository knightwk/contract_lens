# ContractLens 编码约定

> 制定于 Day 1，全项目生效。新增代码前先读一遍。

这些约定不是"让代码好看"。它们只解决一个问题：

> **六周后把普通节点升级成 Agent 时，需不需要推倒重写。**

现在遵守 = 零成本。以后补 = 重构，而且**要重跑全部评测**——因为你改的是契约，不是实现。

---

## 为什么是这六条

本项目的技术演进路径是固定的：

| 阶段 | 形态 | 代码长什么样 |
| --- | --- | --- |
| W1–W3 | 单体工作流 | 一个函数调下一个函数 |
| W4 | LangGraph 状态机 | 节点 + 条件边 + 循环上限 |
| W5 | **两个节点升级为 Agent** | 风险识别 / 复核 拥有自主工具调用 |
| W6 | 多 Agent + 人机协同 | 子图、handoff、interrupt |

**关键点：W3 和 W5 的业务逻辑是同一套。** 差别只在"谁来调用它、谁决定调用顺序"。

所以前三条约定是同一个目标的三面：**让每个能力从第一天起就拥有独立、完整、可单独调用的契约**。有了契约，"把函数包成 Agent"才只是加一层壳；没有契约，你就得先花两天拆耦合。

后三条是配套的纪律：目录结构决定了拆分的成本，引用可回溯和模型访问收口是**受监管场景和本地部署**的硬要求，事后补不了。

---

## 约定 1｜能力即函数

**每个能力（解析 / 提取 / 风险识别 / 复核 / 汇总）都是一个独立可调用的函数或类，输入输出都是 Pydantic 模型。**

不依赖全局状态，不读写共享字典，不从上下文里"顺手拿"东西。

### 对照

```python
# ✅ 正确：契约显式，可单独调用、单独测试
class ExtractionInput(BaseModel):
    parsed_text: str
    document_id: str


class Clause(BaseModel):
    location: str                                    # 第几条第几款
    text: str


class ExtractionOutput(BaseModel):
    parties: list[str]
    clauses: list[Clause]                            # 要素，不是风险项
    effective_date: str | None = None


def extract_contract(payload: ExtractionInput) -> ExtractionOutput:
    ...
```

```python
# ❌ 错误：靠共享 State 互相通信
def extract_node(state: dict) -> dict:
    state["clauses"] = parse(state["raw"])
    state["_tmp_parsed"] = state["clauses"]      # 顺手塞个中间结果
    return state
```

### 违反的代价

`❌` 写法今天能跑，看起来还更省事（少写两个模型）。但到 W5 你想把风险识别包成 Agent 时：

- **Agent 的输入输出契约 = State 的全部字段**，你没法只说"它需要这三样"
- **没法为它声明工具白名单**（约定 3），因为它的边界是模糊的
- **没法单独测它**——要测就得先构造一个完整的 State，包括那些不相关的字段
- 那个 `_tmp_parsed` 是谁写的、还有谁在读？**没人知道。**

### 自检

> 我能不能在 REPL 里这样调用它？
>
> ```python
> result = my_capability(MyInput(...))
> ```
>
> 如果必须先在别处构造一堆状态才能调，就违反了。

---

## 约定 2｜State 按能力分字段

**LangGraph 的 State 按能力分区，每个能力一个字段。禁止用一个笼统的 `answer` 或 `result` 承载所有东西。**

```python
class ReviewState(BaseModel):
    # 输入
    document_id: str
    parsed_text: str

    # 每个能力一个字段，天然是该能力的输出契约
    extraction_result: ExtractionOutput | None = None
    retrieval_result: RetrievalOutput | None = None
    risk_result: RiskOutput | None = None
    review_result: ReviewOutput | None = None
    report: Report | None = None

    # 控制流（不属于任何能力）
    iteration_count: int = 0
```

```python
# ❌ 错误：所有东西挤在一个字段里
class ReviewState(BaseModel):
    document_id: str
    answer: str            # 什么都往里塞
    history: list[dict]    # 谁写的？不知道
```

### 为什么

**State 的字段名就是能力清单。** 看到 `risk_result` 就知道有个能力叫"风险识别"，它的产出是 `RiskOutput` 类型——**类型即契约**。

到 W5，把 `risk_result` 的产出者从普通节点换成 Agent 时，**State 定义一行都不用改**。这就是约定 1 和 2 配合起来的效果。

### 违反的代价

用 `answer: str` 的项目，升级到多 Agent 时会发现：**Agent A 和 Agent B 都在往 `answer` 里写，谁写的是哪段分不清。** 于是你要么加前缀标记（脆），要么引入"消息历史"结构（那就成了另一个项目）。

### 自检

> 看到 State 的字段列表，能不能说出这个系统有哪几个能力？
>
> 说不出来，说明字段分得不对。

---

## 约定 3｜每个能力声明工具白名单

**`app/agent/tool_policy.py` 集中声明每个能力能用哪些工具。**

```python
# app/agent/tool_policy.py
from typing import Final

# MCP Tool 名 -> 允许使用该工具的能力
CAPABILITY_TOOLS: Final[dict[str, frozenset[str]]] = {
    "extraction": frozenset({"contract_analyze"}),
    "risk_detect": frozenset({"clause_search", "risk_detect", "legal_retrieve"}),
    "verify": frozenset({"legal_retrieve", "cross_reference"}),
    "summarize": frozenset(),  # 汇总不调任何工具，纯生成
}


def allowed_tools(capability: str) -> frozenset[str]:
    """返回该能力可用的工具集。未声明的能力默认无工具。"""
    return CAPABILITY_TOOLS.get(capability, frozenset())


def is_allowed(capability: str, tool_name: str) -> bool:
    return tool_name in allowed_tools(capability)
```

### 一份声明，两个用途

| 用途 | 体现在哪 |
| --- | --- |
| **① 权限控制** | W3 Day 20 的权限校验直接读这张表，不另写一套规则 |
| **② Agent 的工具集** | W5 每个 Agent 只挂载 `allowed_tools(capability)`，天然最小权限 |

**这两件事本质是同一件事**：一个能力"能用什么工具"就是它的权限边界。写成一份，就不会出现"权限表说 A 能调，Agent 配置说 A 不能调"这种自相矛盾。

### 默认拒绝

`allowed_tools` 对未声明的能力返回**空集**，不是全集。这是**默认拒绝**（fail-closed）——新加一个能力时如果忘了声明白名单，它**拿不到任何工具**，你会立刻发现；反过来默认全开，你会到安全评审时才发现漏了。

> 这个原则在项目里反复出现：`debug` 默认 `False`（`config.py`）、未声明能力默认无工具。**默认值要取安全的那一侧。**

### 违反的代价

工具权限散落在每个 Agent 的初始化代码里，就会出现：**同一个能力在两个地方有一份权限声明，改了一处忘了另一处。** 这类不一致在受监管场景是审计事故。

---

## 约定 4｜目录按能力分组

```
app/agent/
├── state.py
├── graph.py                  # 编排：谁先谁后、条件边、循环上限
├── nodes.py                  # 薄壳：把能力适配成节点
├── tool_policy.py            # ★ 约定 3
├── prompts.py
├── validators.py
└── capabilities/             # ★ 能力本体，按能力一个文件
    ├── extraction.py
    ├── risk_detect.py
    ├── verify.py
    └── summarize.py
```

**关键区分**：`nodes.py` 是**适配层**，`capabilities/` 是**业务层**。

```python
# app/agent/nodes.py —— 薄到几乎没有逻辑
async def risk_detect_node(state: ReviewState) -> dict:
    payload = RiskInput(
        clauses=state.extraction_result.clauses,
        document_id=state.document_id,
    )
    result = await detect_risk(payload)      # 真正的逻辑在 capabilities/
    return {"risk_result": result}
```

**节点函数只做两件事：把 State 拆成能力的输入，把能力的输出放回 State。** 不写业务判断。

### 为什么

到 W5 拆子图时：

- **拆分 = 把 `capabilities/` 里的某几个文件包成一个子图**
- 不拆 `nodes.py`（它就是适配层，子图里仍然需要）
- **不用重新组织代码，只是换个组合方式**

如果业务逻辑写在 `graph.py` 的节点函数里，拆分就变成了"从 500 行的图定义里把相关代码抠出来"——那是重写，不是重构。

### 自检

> `graph.py` 里有没有超过 15 行的节点函数？
>
> 有的话，业务逻辑漏到编排层了。

---

## 约定 5｜所有引用必须带知识库条目 ID

**风险报告里的每一条法律依据，都必须携带它在知识库中的条目 ID。回溯失败时 `legal_basis` 为 `null`，不允许编造。**

```python
class Citation(BaseModel):
    source_id: str      # 知识库条目 ID，如 "statute:CIVIL_CODE:563"
    source_type: Literal["statute", "precedent", "contract_clause"]
    snippet: str        # 原文片段
    score: float        # 检索得分


class RiskItem(BaseModel):
    original_text: str                                   # 合同原文，必须能搜到
    clause_location: str                                 # 第几条第几款
    legal_basis: list[Citation] | None                   # 必须带 source_id
    confidence: float
    risk_level: Literal["高", "中", "低"]
    suggestion: str
    evidence_sufficient: bool
```

### 两条强制规则（与项目规划 §12 一致）

1. `legal_basis` 为 `null` **且** `evidence_sufficient=False` → 报告必须标记「依据不足，建议人工复核」
2. `risk_level == "高"` → **必须进入人工确认环节**

### 为什么不能事后补

受监管场景（法律 / 金融）的核心要求是**可审计**：每条结论都要能回答"你凭什么这么说"。

如果检索结果一开始只存 `text` 不存 `source_id`：

- 想在报告里回溯，就得**拿文本反查知识库**——文本可能被截断、被改写，反查不回去
- 想加"依据不足"标记，就得重新判断依据是否存在——但**判断依据已经丢了**

**这两件事都需要 ID 在检索那一刻就被保留。事后加字段，历史数据全部作废。**

### 反面模式

```python
# ❌ 只存文本，不存来源
class Citation(BaseModel):
    text: str          # "……有下列情形之一的，当事人可以解除合同……"
    # 这段话出自哪一条？哪个版本？无从得知
```

### 自检

> 随机挑一条报告里的法律依据，我能不能用它的 `source_id` **直接**在知识库中定位到那一条？
>
> 需要"搜一下看看像不像"就说明失败了。

---

## 约定 6｜模型访问统一走 `core/model_runtime.py`

**业务代码不直接 `import torch`、`import sentence_transformers`，也不自己实例化模型。**

```python
# ✅ 业务代码
from app.core.model_runtime import model_runtime

vectors = await model_runtime.encode_async(texts)
scores = await model_runtime.rerank_async(query, candidates)
```

```python
# ❌ 业务代码
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer("./models/bge-base-zh-v1.5")   # 会在每个用到的地方重复加载
```

### 三个理由

| # | 理由 | 具体表现 |
| --- | --- | --- |
| **① 单例与预热** | 模型加载要 10–30 秒、占几百 MB 内存。分散实例化 = 重复加载 = 内存爆掉 | 集中在一处才能做单例和 lifespan 预热 |
| **② 线程调度** | `asyncio.to_thread`、`torch.set_num_threads` 必须统一管，否则并发时线程互相抢占 | 散落各处就没法统一调 |
| **③ Provider 切换** | `EMBEDDING_PROVIDER=local\|api` 要一处生效 | 这个开关的全部价值就在于"只改一处" |

**第 ③ 条是这份约定的存在意义。** 我们在 `config.py` 里承诺了"改一行配置就能在本地模型和 API 之间切换"——**如果模型调用散落在 5 个文件里，这个承诺就是假的。**

### 接口形状

```python
# app/core/model_runtime.py（Day 2 实现）
class ModelRuntime:
    async def warmup(self) -> None:
        """在 lifespan 中调用，加载并预热全部模型。"""

    async def encode_async(self, texts: list[str]) -> list[list[float]]:
        """文本 -> 向量。内部用 asyncio.to_thread 隔离 CPU 阻塞。"""

    async def rerank_async(self, query: str, candidates: list[str]) -> list[float]:
        """query-doc 对打分。"""

    @property
    def is_ready(self) -> bool:
        """供 /health 的 models_ready 字段使用。"""
```

**注意全部是 `async`。** 这不是风格问题——底层是 CPU 密集的阻塞调用，**在 `async def` 里直接调用会卡死事件循环**：一个请求在算，其他所有并发请求一起等。必须用 `asyncio.to_thread` 把它丢到线程池。

**把这条封在 `model_runtime` 里，业务代码就不需要知道这件事。** 这也是"收口"的价值——危险的细节只需要在一个地方被正确处理。

### 自检

> `grep -rn "import torch\|sentence_transformers" app/`
>
> **除了 `app/core/model_runtime.py`，不应该有第二个文件命中。**

---

## 附：Day 1 记录的四个失败模式

写代码时踩过的坑，一并留档。它们和上面的约定是同一类思维在不同层面的体现。

| # | 失败模式 | 正确做法 | 背后的原则 |
| --- | --- | --- | --- |
| 1 | `case_sensitive=True` 导致 `.env` **静默失效** | `case_sensitive=False`，字段名迁就环境变量惯例 | 最危险的 bug 不是崩溃，是**不崩但不对** |
| 2 | 依赖版本凭感觉写（`langfuse==3.2.1`） | 查**官方兼容表**定版本 | 大版本是否破坏兼容，查表比推理可靠 |
| 3 | 用 `./models` 相对路径 | `PROJECT_ROOT / models_dir` 锚定代码位置 | 相对 CWD 的路径 = "在我机器上是好的" |
| 4 | 把 `/health` 放进 `/api/v1/` | 基础设施端点不版本化 | **接口的生命周期决定它该不该版本化** |

---

## 一页自查清单

提交代码前，问自己这六个问题：

- [ ] 这个能力的输入输出是 **Pydantic 模型**吗？（约定 1）
- [ ] State 里每个能力有**独立字段**吗？还是有共享的 `answer`？（约定 2）
- [ ] 它用的工具在 `tool_policy.py` 里**声明过**吗？（约定 3）
- [ ] `graph.py` 里的节点函数**超过 15 行**了吗？（约定 4）
- [ ] 产生的每条引用都带 **`source_id`** 吗？（约定 5）
- [ ] 有没有**绕过 `model_runtime`** 直接碰模型？（约定 6）

---

**最后一句**：这六条里，前四条是"为了将来能升级"，后两条是"现在就不能错"。

后两条如果违反了，**不是技术债，是缺陷**——一个让报告不可审计，一个让配置开关失效。发现就修，不要记进待办。

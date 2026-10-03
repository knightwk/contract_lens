# 决策记录（模式库）

> **这份文件的用途：把「设计能力」变成可积累的模式库。**
>
> 面试里「你怎么保证重复上传不产生脏数据」这类问题，
> 答案不来自临场推导，来自**你记得多少个带取舍的模式**。
>
> 目标：**W6 时 60–80 条。** 这是能带进面试的硬资产。

---

## 记录格式

每条固定五段：

| 段 | 要求 |
| --- | --- |
| **决策** | 做了什么。**具体到代码或参数**，不要写"用了合适的索引" |
| **为什么** | 机制层面的理由。不是"感觉更好"，是"因为 X 所以 Y" |
| **反过来会怎样** | 另一个选择的后果 —— **这一段最能体现理解深度** |
| **来源** | 哪天 / 哪个文件 / 哪次填错 |
| **面试怎么用** | 它是什么问题的答案 |

**特别注意「反过来会怎样」**：能说清"不这么做会出什么事"，比能说清"这么做是对的"高一个层次。

---

## 什么时候记

优先级从高到低：

1. **填错的 TODO** —— 信噪比最高，因为有真实代价
2. **破坏练习中预测错的那次**
3. 任何"我本来会写成另一种"的地方
4. 我写满、你看懂后觉得"原来要这样"的地方

---

## 索引

| 编号 | 主题 | 来源 | 状态 |
| --- | --- | --- | --- |
| D-001 | `chunk_id` 用内容哈希而不是自增 ID | Day 3 | ✅ 示例 |
| D-002 | 配置大小写：`case_sensitive` 导致静默失效 | Day 1 | ⬜ 待补 |
| D-003 | `pymilvus` 锁 2.6.17 而非最新 3.x | Day 1 | ⬜ 待补 |
| D-004 | `PROJECT_ROOT` 锚定路径而非相对 CWD | Day 1 | ⬜ 待补 |
| D-005 | `/health` 不放在 `/api/v1` 下 | Day 1 | ⬜ 待补 |
| D-006 | Provider 开关用 `Literal` 而非 `str` | Day 1 | ⬜ 待补 |
| D-007 | `torch.set_num_threads` 放 `__init__` 而非 `warmup` | Day 2 | ⬜ 待补 |
| D-008 | `asyncio.to_thread` 之外还要加信号量 | Day 2 | ⬜ 待补 |
| D-009 | 信号量按事件循环惰性创建 | Day 2 | ⬜ 待补 |
| D-010 | 归一化让内积等价于余弦 | Day 2 | ⬜ 待补 |
| D-011 | 用 BGE 自己的 tokenizer 计数而非 tiktoken | Day 3 | ⬜ 待补 |
| D-012 | 条款正则必须行首锚定 | Day 3 | ⬜ 待补 |
| D-013 | 附件要独立成组，不能挂最后一个条款 | Day 4 | ⬜ 待补 |
| D-014 | 「跨页重复」启发式对页码无效 | Day 3 | ⬜ 待补 |
| D-015 | 页码单调性自检的价值 | Day 3 | ⬜ 待补 |
| D-016 | 写入用 `upsert` 而非 `insert` | Day 4 | ⬜ 待补 |

**注意：除 D-001 是格式示例外，其余内容必须你自己写。**
写不出来的那几条，就是还没真正懂的地方 —— 标记出来，我们重点过。

---

## D-001｜`chunk_id` 用内容哈希而不是自增 ID

**决策**

```python
raw = f"{doc_id}|{char_start}|{clause_no or ''}|{index}/{total}|{text}"
chunk_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
```

**为什么**

- **确定性**：同一份合同重复上传 → 得到相同的 ID
- 配合 Milvus 的 `upsert` → **天然幂等**，不需要另写去重逻辑（W3 Day 19）
- 主键里**必须含 `char_start`**：真实合同里同一段条款文本会出现多次
  （附件重排、末页错位），不含位置信息就会碰撞

**反过来会怎样**

- 用 `uuid4`：每传一次就多一份副本，W3 的幂等要另想办法
- **去掉 `char_start`**：Day 3 用真实合同实测 —— 78 个块只剩 **76 个唯一 ID**，
  两处碰撞。被覆盖的两块文本相同，但 `page` / `char_start` 不同，
  结果 **引用会指向错误的位置**，而且**不会报任何错**

**来源**：Day 3 `app/core/chunker.py` 的 `_make_chunk_id`；Day 4 用真实合同确认

**面试怎么用**

问「重复上传同一份文件会不会产生脏数据」时，这是完整答案。

加分点在于能说清**为什么 ID 里必须有位置信息** ——
大多数人只会说"我用内容哈希"，但说不出"为什么纯内容哈希不够"。

---

<!-- 下面按同样的格式补写 D-002 起 -->

## D-002｜配置大小写：`case_sensitive` 导致静默失效

**线索**：Day 1 `app/config.py`。

**要回答的问题**：为什么 `case_sensitive=True` 会让整个 `.env` 失效？
为什么这个 bug 比"程序崩溃"更危险？

---

## D-003｜`pymilvus` 锁 2.6.17 而非最新 3.x

**线索**：Day 1 `requirements.txt`、Day 4 `docker-compose.yml`。

**要回答的问题**：版本选择的依据是什么（不是"更稳定"这种空话）？
Day 4 建 Milvus 时这个决定会在哪里被验证？

---

## D-004｜`PROJECT_ROOT` 锚定路径而非相对 CWD

**线索**：Day 1 `app/config.py` 的 `PROJECT_ROOT` 与 `_resolve_model_path`。

**要回答的问题**：`./models` 和 `PROJECT_ROOT / "models"` 在什么情况下表现不同？
列举三种会出问题的场景。

---

## D-005｜`/health` 不放在 `/api/v1` 下

**线索**：Day 1 `app/api/health.py` 与 `app/main.py` 的 `include_router`。

**要回答的问题**：判断一个接口该不该版本化的标准是什么？
如果 `/health` 在 `/api/v1/` 下，将来出 v2 时会遇到什么尴尬？

---

## D-006｜Provider 开关用 `Literal` 而非 `str`

**线索**：Day 1 `app/config.py` 的 `embedding_provider`。

**要回答的问题**：配置写错时，错误在**什么时候**暴露？这个时间差的实际代价是什么？

---

## D-007｜`torch.set_num_threads` 放 `__init__` 而非 `warmup`

**线索**：Day 2 `app/core/model_runtime.py`。

**要回答的问题**：torch 的线程池是什么时候创建的？
为什么"设晚了"不会报错但会静默失效？

---

## D-008｜`asyncio.to_thread` 之外还要加信号量

**线索**：Day 2 `app/core/model_runtime.py` 的 `_semaphore()`。

**要回答的问题**：`asyncio.to_thread` 用的是哪个线程池？它的 `max_workers` 是多少？
不加信号量时，8 个并发请求会在 8 核机器上开出多少线程？

---

## D-009｜信号量按事件循环惰性创建

**线索**：Day 2 `app/core/model_runtime.py` 的 `_semaphore()` 注释。

**要回答的问题**：`asyncio.Semaphore` 什么时候会绑定到事件循环？
什么场景下会触发 `is bound to a different event loop`？

---

## D-010｜归一化让内积等价于余弦

**线索**：Day 2 `_encode_sync` 的 `normalize_embeddings=True`；Day 4 的 `metric_type="IP"`。

**要回答的问题**：向量归一化之后，IP 和 COSINE 的结果关系是什么？
选 IP 省掉了什么计算？

---

## D-011｜用 BGE 自己的 tokenizer 计数而非 tiktoken

**线索**：Day 3 `app/core/chunker.py` 的 `count_tokens` / `max_tokens`。

**要回答的问题**：BERT 中文词表和 tiktoken 对同一段中文的计数差多少？
按 tiktoken 算不超限、实际被截断时，症状是什么？

---

## D-012｜条款正则必须行首锚定

**线索**：Day 3 `app/core/chunker.py` 的 `_RE_CLAUSE`。

**要回答的问题**：合同正文里会出现哪类文本，不加 `^` 就会被误判成条款标题？
`tests/test_chunker.py` 里哪条用例在防这个？

---

## D-013｜附件要独立成组，不能挂最后一个条款

**线索**：Day 4 `app/core/chunker.py` 的 `_RE_APPENDIX` / `_RE_BODY_END` / `_RE_SIGNATURE`。

**要回答的问题**：修复前「第十五条」占了多少块？为什么这比 `chunk_id` 碰撞更危险？
为什么**不能**用"条款长度上限"来兜底？

---

## D-014｜「跨页重复」启发式对页码无效

**线索**：Day 3 `app/core/parser.py` 的 `_strip_headers_footers` 与 `_PAGE_NUMBER_PATTERNS`。

**要回答的问题**：为什么按字符串统计的方法对「第 N 页 共 M 页」永远不触发？
什么样的页眉页脚能靠统计、什么样的必须靠模式？

---

## D-015｜页码单调性自检的价值

**线索**：Day 3 `app/core/parser.py` 的 `_audit_pages`。

**要回答的问题**：这个检查发现了什么、成本多少？
为什么说它"复用了本来就要写的正则，一份代码两个用途"？

---

## D-016｜写入用 `upsert` 而非 `insert`

**线索**：Day 4 `app/core/vector_store.py` 的 `upsert_chunks`。

**要回答的问题**：把 `upsert` 改成 `insert`，重复导入同一份合同两次，
集合行数会怎样？这与 `chunk_id` 的设计是什么关系？

---

**记录纪律**：每条都要写「反过来会怎样」。
写不出来，说明还停在"知道怎么做"，没到"知道为什么不能那么做"。

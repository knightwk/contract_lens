# ContractLens 学习辅助指南

## 1. 提问协议（让你我高效沟通）

为了不丢失计划且获得精准回答，请遵循以下标签格式提问（直接复制前缀）：

| 标签 | 适用场景 | 示例 |
| :--- | :--- | :--- |
| `[Bug]` | 代码报错，跑不通 | `[Bug] Day 8 PostgreSQL连不上，报错FATAL: password authentication failed` |
| `[Why]` | 不理解原理 | `[Why] Day 16 为什么实体抽取的Prompt要加Few-shot？不加行不行？` |
| `[Code]` | 不确定怎么写 | `[Code] Day 20 用matplotlib画networkx图，节点太多重叠了怎么办？` |
| `[Review]` | 写完了想让看一眼 | `[Review] Day 25 我刚写完Milvus迁移代码，帮我看下索引参数设的对不对` |
| `[Plan]` | 对计划有疑问或需要调整 | `[Plan] 我今天加班只有1小时，Day 19的任务能拆成两天做吗？` |

**万能召回指令**（如果聊散了）：
> “请忘记之前闲聊。请以‘ContractLens项目导师’的身份，调出第X周的完整计划，并针对我今天的问题给出指导。”

**每日第一次提问时，请固定以 `[Day X]` 开头**（例如：`[Day 12] 手动实现RRF时...`），这样我能自动对齐你的进度。

---

## 2. 三个“保命”建议（避免中途放弃）

1. **前两周绝不碰Neo4j和Milvus**。严格按照计划，先用`networkx`和`dict`。当你手动BFS处理100个节点卡死时，你会发自内心地理解为什么需要Neo4j。
2. **每天必须产生可视化输出**。哪怕是`print("检索结果：", chunk_text[:50])`。AI开发最大的敌人是“黑盒”，你必须亲眼看到每一步的中间结果。
3. **休息日（Day 7/14/21/28/34）绝不能跳过**。这些天用来整理笔记和“爽一把”——把你上周的代码注释删掉重新读一遍，你会发现自己进步神速。

---

## 3. 理论知识阅读清单（按主题分类）

### 入门必修（第1-2周）
- [Datawhale《动手学大模型应用开发》](https://datawhalechina.github.io/llm-universe/)
- [LangChain中文官方文档](https://www.langchain.com.cn/)
- [Qwen通义千问开发文档](https://qwen.readthedocs.io/)
- [FastAPI官方文档](https://fastapi.tiangolo.com/)

### RAG深度（第2-4周）
- [《从零构建高性能RAG系统》](https://cloud.tencent.com.cn/developer/article/2703531)（全景架构）
- [《2026年RAG+本地知识库工程化落地实战》](https://cloud.tencent.com.cn/developer/article/2703531)（混合检索+重排序）
- [《基于RAG+LangChain搭建企业级私有知识库》](https://developer.aliyun.com/article/1755975)（完整代码）
- [Milvus官方文档](https://milvus.io/docs/)

### GraphRAG核心（第3-4周）
- [《GraphRAG深度解析》](https://bbs.huaweicloud.com/blogs/480948)（从向量到图的三段迭代）
- [《传统RAG已死？Agentic GraphRAG》](https://cloud.tencent.com.cn/developer/article/2703531)（2026年WAIC焦点）
- [《GraphRAG实践：企业知识库架构升级》](https://developer.aliyun.com/article/2703531)（从文档检索到关系推理）

### Agent与LangGraph（第5-6周）
- [《从零构建AI Agent：基于LangGraph的多工具智能体实战》](https://developer.aliyun.com/article/1755760)（完整代码）
- [《一文彻底搞懂AI Agent——LangGraph》](https://cloud.tencent.com.cn/developer/article/2703531)（从单Agent到多Agent）
- [《从LangChain到LangGraph构建可控Agent》](https://developer.aliyun.com/article/1755760)（StateGraph vs AgentExecutor）

### MCP协议（第二项目备用）
- [MCP官方文档](https://modelcontextprotocol.io)
- [《2026 MCP协议全面普及》](https://cloud.tencent.com.cn/developer/article/2707609)

### 技能地图与面试（全程参考）
- [《AI应用工程师技能地图》](https://www.jianshu.com/p/1413d5239f97)（五层能力拆解）
- [《测试开发转大模型应用开发学习规划》](https://www.cnblogs.com/alisleepy/p/21417271)（10周70天）
- [GitHub: gen-ai-roadmap](https://github.com/gautamrastogi/gen-ai-roadmap)（10阶段29+项目）

---

## 4. 周末 Checkpoint 模板

每周末（Day 7/14/21/28/34/42）晚上，请发一条消息给我，格式如下：

> **Week X Checkpoint**
> - 本周完成的主要任务：
> - 遇到的3个印象最深的报错及解决方案：
> - 下周想重点加强的方向：

我会根据你的反馈，**微调下一周的任务优先级和理论重点**，确保计划始终贴合你的实际进度。

---

## 5. 最后一个建议

**理论是为了服务实战，不是反过来。**
如果今天代码没调通，**优先把时间花在调试上**，理论可以顺延到明天。代码跑通了，理论一看就懂；代码没跑通，看再多理论也是纸上谈兵。

现在，打开终端，输入：
```bash
mkdir -p ContractLens/docs
cd ContractLens
git init
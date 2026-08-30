# ContractLens 42天实战学习计划（每日任务 + 理论）

## 总体原则
- **实战优先**：每天2-3小时编码，0.5小时理论。
- **手动挡先行**：先用Python基础库（dict、networkx）跑通逻辑，再替换为生产级组件（Milvus、Neo4j、LangGraph）。
- **每日产出**：必须有可视化的中间结果（打印、截图、接口返回）。

---

## 第1周：地基搭建 + 丐版RAG（完成标志：能在终端问PDF内容）

| 天数 | 实战任务（2-2.5h） | 理论知识（0.5h） |
| :--- | :--- | :--- |
| **Day 1** | 创建项目文件夹结构，写`requirements.txt`（只装`fastapi, uvicorn, pandas, numpy, pymupdf, python-dotenv, requests`），跑通`uvicorn main:app --reload` | 读[Datawhale《动手学大模型应用开发》](https://datawhalechina.github.io/llm-universe/)第一章，理解“大模型应用开发是什么” |
| **Day 2** | 写`app/utils/pdf_parser.py`，用`fitz`提取PDF全文纯文本，打印前200个字符确认成功 | 读DeepSeek/通义千问的API文档，理解`temperature`、`max_tokens`参数的作用 |
| **Day 3** | 写`chunk_text()`，**按“第X条”正则切分**（`r'第[零一二三四五六七八九十百]+条'`），存成`List[dict]`带`chunk_id`和`title` | 读一篇RAG入门文章，理解“切片为什么重要” |
| **Day 4** | 写内存向量库`app/core/vector_store_memory.py`，用` sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`把切片转成向量，存进`dict` | 读Embedding模型原理（通俗版），理解“向量相似度”是什么 |
| **Day 5** | 写`search_manual()`，暴力计算余弦相似度取Top-3，打印检索到的切片内容 | 读[LangChain中文文档](https://www.langchain.com.cn/)的“Chain”概念 |
| **Day 6** | 写`generate()`，把Top-3拼进Prompt调用DeepSeek API，终端输出答案 | 读Prompt Engineering基础（Few-shot、CoT） |
| **Day 7** | **休息/复盘**。把代码封装进FastAPI的`/chat`接口，Postman测试通 | 整理第一周笔记，回答“RAG的全流程是什么” |

---

## 第2周：结构化元数据（完成标志：能用“金额>100万”过滤合同）

| 天数 | 实战任务（2-2.5h） | 理论知识（0.5h） |
| :--- | :--- | :--- |
| **Day 8** | 用Docker跑PostgreSQL，建表：`chunk_id, contract_name, party_a, party_b, total_amount, sign_date` | 读[《从零构建高性能RAG系统》](https://cloud.tencent.com.cn/developer/article/2703531)，理解RAG全景架构 |
| **Day 9** | 写`extract_metadata.py`，调用DeepSeek抽取`party_a, party_b, amount`，解析JSON存入PG | 读“结构化输出”最佳实践，理解JSON Schema校验 |
| **Day 10** | 改检索逻辑：先查PG拿到符合过滤条件的chunk_id列表，再在内存向量库里只检索这些ID | 读“元数据过滤”在RAG中的作用 |
| **Day 11** | 用`rank_bm25`库建BM25索引，写`search_bm25()`返回Top-3 | 读BM25算法原理（通俗版），理解“关键词检索”和“向量检索”的区别 |
| **Day 12** | **手动实现RRF融合**：`reciprocal_rank_fusion(results_list, k=60)`，打印融合前后的Top-5对比 | 读RRF算法原理 |
| **Day 13** | 把融合逻辑接入`/chat`接口，对比“只用向量”vs“融合检索”的回答质量 | 读[《2026年RAG+本地知识库的工程化落地实战》](https://cloud.tencent.com.cn/developer/article/2703531) |
| **Day 14** | **休息/复盘**。整理笔记：“为什么BM25能补上向量检索的短板？” | 复习本周所有理论知识，画一张RAG流程图 |

---

## 第3周：手动图谱模拟（NetworkX）（完成标志：能用“图遍历”找条款关联）

| 天数 | 实战任务（2-2.5h） | 理论知识（0.5h） |
| :--- | :--- | :--- |
| **Day 15** | **装`networkx`不装Neo4j**。定义合同图谱：节点（条款、实体）、边（引用、约束），手动写死3条边 | 读[GraphRAG深度解析](https://bbs.huaweicloud.com/blogs/480948)，理解“从向量RAG到GraphRAG的三段技术迭代” |
| **Day 16** | 写实体抽取Prompt，输入“甲方逾期付款→乙方有权解除合同”，输出JSON三元组，跑通5个例子 | 读“信息抽取”的Prompt工程技巧 |
| **Day 17** | 把抽取出的实体/关系用`networkx.add_node()`和`add_edge()`存入内存图，打印节点数和边数 | 读[GraphRAG核心原理](https://bbs.huaweicloud.com/blogs/480948)：数据表示从文本块到三元组的跃迁 |
| **Day 18** | 手写图遍历`graph_query()`，用`nx.dfs_successors`或手动BFS**限深2跳**，返回关联节点列表 | 读“多跳推理”为什么是GraphRAG的核心能力 |
| **Day 19** | 把图检索结果加入融合池，现在RRF融合三路：`向量 + BM25 + 图遍历` | 读[《传统RAG已死？Agentic GraphRAG》](https://cloud.tencent.com.cn/developer/article/2703531)，理解2026年技术趋势 |
| **Day 20** | 写FastAPI接口`/graph_viz`，用`matplotlib`把`networkx`图可视化成图片 | 读GraphRAG系统架构详解 |
| **Day 21** | **休息/复盘**。你现在理解了“图检索到底在查什么” | 整理笔记：“图遍历为什么能多跳推理？” |

---

## 第4周：替换为生产级组件（完成标志：Milvus + Neo4j跑通）

| 天数 | 实战任务（2-2.5h） | 理论知识（0.5h） |
| :--- | :--- | :--- |
| **Day 22** | **正式上Neo4j**。写`docker-compose.yml`加`neo4j:5`，学Cypher基础（`CREATE, MATCH, WHERE`） | 读Neo4j官方文档“Cypher基础”章节 |
| **Day 23** | 把`networkx`数据导出为Cypher语句灌入Neo4j，执行`MATCH (a)-[:REFERS_TO*1..2]-(b)`验证 | 读“图数据库 vs 关系数据库”的对比文章 |
| **Day 24** | **正式上Milvus**（Standalone模式），把内存向量`dict`迁移到Milvus Collection，建IVF_FLAT索引 | 读[Milvus官方文档](https://milvus.io/docs/)“建索引”章节 |
| **Day 25** | 把暴力余弦相似度替换为Milvus的`search()`，体验毫秒级检索 | 读“向量索引类型（IVF_FLAT vs HNSW）的选择” |
| **Day 26** | 接入`BGE-Reranker`，RRF融合后取Top-10送进Reranker输出Top-3，对比精排前后 | 读“Reranker（交叉编码器）”的原理 |
| **Day 27** | 重构代码，抽象出`Retriever`类（`vector_search`, `bm25_search`, `graph_search`, `hybrid_fusion`） | 读[LangChain官方文档](https://www.langchain.com.cn/)“Retriever”概念 |
| **Day 28** | **休息/复盘**。跑通完整`/chat`端到端（上传→入库→查询→混合检索→Rerank→生成） | 复习本周所有理论知识 |

---

## 第5周：手动Agent循环（完成标志：能自主决定“搜图”还是“搜向量”）

| 天数 | 实战任务（2-2.5h） | 理论知识（0.5h） |
| :--- | :--- | :--- |
| **Day 29** | 写`router()`（纯`if-else`），用LLM判断问题类型：含“关联/关系”走图，含“是什么”走向量 | 读“Agentic RAG”概念，理解“自主路由” |
| **Day 30** | 实现**手动反思**：检索后调用LLM“能回答吗？”，如果“否”则修改Query重新检索 | 读“Self-Reflection（自我反思）”机制 |
| **Day 31** | 封装`while cycle`最多3次“检索→反思→重检索”，打印每次反思日志 | 读[《从LangChain到LangGraph构建可控Agent》](https://developer.aliyun.com/article/1755760) |
| **Day 32** | **正式上LangGraph**。用`StateGraph`定义节点：`router, retriever, checker, generator` | 读LangGraph vs LangChain AgentExecutor的核心区别 |
| **Day 33** | 接入`Langfuse`，在Graph每个节点插入`span`，面板里看到每一步的Token消耗和耗时 | 读“可观测性（Observability）”在Agent中的重要性 |
| **Day 34** | **休息/复盘**。你现在有了真正的Agentic RAG系统 | 整理笔记：“LangGraph如何管理检索-反思-再检索的状态流转” |

---

## 第6周：生产化 + 面试冲刺（完成标志：Docker一键部署 + 简历亮点）

| 天数 | 实战任务（2-2.5h） | 理论知识（0.5h） |
| :--- | :--- | :--- |
| **Day 35** | 写`Dockerfile`（基于Python 3.10-slim），`docker build -t contractlens .`成功 | 读Docker最佳实践（多层构建、依赖缓存） |
| **Day 36** | 补全`docker-compose.yml`一键拉起`FastAPI+Neo4j+Milvus+Redis`，写`README.md` | 读“生产级RAG系统的架构设计原则” |
| **Day 37** | 构建5个刁钻问题的测试集，把运行结果截图存进`docs/` | 读“RAG评测指标（Recall, Precision, MRR）” |
| **Day 38** | 写`eval/manual_test.py`跑5个问题，记录答案和溯源。不准就调Prompt（加Few-shot） | 读[Ragas评测框架](https://github.com/explodinggradients/ragas)文档 |
| **Day 39** | 写出8个面试高频问题的回答逐字稿（“为什么选GraphRAG”、“反思怎么实现”） | 读[《AI应用工程师技能地图》](https://www.jianshu.com/p/1413d5239f97)，对照检查能力缺口 |
| **Day 40** | 录制5分钟项目演示视频（上传PDF→问3个问题→展示Langfuse链路），放GitHub Readme | 读“如何向面试官展示RAG项目” |
| **Day 41** | 更新简历项目描述，重点突出“手动实现混合检索融合”和“LangGraph驱动的反思循环” | 读目标公司的AI岗位JD，针对性调整简历 |
| **Day 42** | **终极复盘**。通读Neo4j和Milvus官方文档“最佳实践”，把增量更新、索引调优原理补进笔记 | 准备“数据量暴增100倍怎么办”的系统设计回答 |

---
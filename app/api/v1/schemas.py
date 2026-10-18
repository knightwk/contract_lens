"""业务接口的请求 / 响应模型。

为什么单独一份，不复用 core/schemas.py
    core 里的模型是【内部契约】（Chunk / SearchHit / RagAnswer），
    改它们会牵连检索、入库、编排一整条链路。

    这里是【对外契约】—— 一旦有前端或第三方调用就不能随便改。
    两者生命周期不同，混在一起会导致"为了改接口而动了内核"。
"""

from pydantic import BaseModel, Field


class TimingBreakdown(BaseModel):
    """一次请求的分段耗时（毫秒）。"""

    total_ms: float
    segments: dict[str, float] = Field(default_factory=dict)


# ── 引用契约的四个决定（Day 6 完成） ──
#
# ① chunk_id 必须出现
#    约定 5 要求「引用必须带知识库条目 ID」。不给 ID，前端就没法把这条引用
#    定位回具体条目，也就没法做"点开原文核对"—— 引用就成了装饰。
#
# ② 只暴露用户需要的字段
#    SearchHit 有 11 个字段，但 score / char_start / char_end 是内部调试信息。
#    全照搬会让对外契约被内部实现绑死 —— 以后改检索层就得升接口版本。
#    这里只给：文档 ID、条目 ID、定位、原文。
#
# ③ 字段名用对外名（clause_location / original_text），不用内部名
#    clause_no / char_start 是检索层的词汇。对外换一套名字，
#    以后内部改名（比如 section 拆成 chapter + section）时接口不用动。
#
# ④ original_text 给【全文】，不截断
#    用途是"核对"，截断就核不了 —— 使用者要能自己判断这句话有没有被曲解。
#    体积不用担心：单条通常 100~400 字，而 citations 只含【回答实际引用的】
#    那几条（见 query.py），实际是 1~3 条，几 KB。
#    如果真嫌大，应该【限条数】，不要限单条长度 —— 截断会直接破坏可核对性。



class Citation(BaseModel):
    """回答里的一条依据。"""

    doc_id: str = Field(description="知识库文档ID")
    chunk_id: str = Field(description="引用知识库的条目ID")
    clause_location: str = Field(description="人类可读的定位，如「第二章 服务内容与期限 · 第四条」")
    original_text: str = Field(description="知识库原文")


class UploadResponse(BaseModel):
    """上传并入库的结果。"""

    doc_id: str
    filename: str
    source_type: str
    chunks: int
    numbered: int
    written: int
    duration_ms: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=50, description="检索条数，缺省取服务端配置")


class QueryResponse(BaseModel):
    """带引用的回答。

    citations 是【回答的依据】，不是"检索到的全部材料"——
    这个区别很重要：给用户看的是"我这句话从哪来"，不是检索日志。
    """

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # 回答是否有材料支撑。
    # 检索为空时返回 200 + grounded=false —— 对用户是正常结果（不该显示成报错），
    # 但监控需要它统计"无依据率"，否则这类情况会完全隐形。
    grounded: bool = True
    timings: TimingBreakdown

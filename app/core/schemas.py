from typing import Any, Literal

from pydantic import BaseModel, Field


class Block(BaseModel):
    """解析后的最小内容单元，带位置信息。"""

    text: str
    page: int | None = None  # 引用可回溯需要；docx / md 无页概念，为 None
    kind: Literal["heading", "paragraph", "table"] = "paragraph"


class ParsedDocument(BaseModel):
    """一份文档的解析结果"""

    doc_id: str
    filename: str
    # ⚠️ 必须与 parser.py 的 _PARSERS 注册值一致。
    # 这里曾经写成 "docs"（多一个 s），而 parser 注册的是 "docx" ——
    # 结果就是【一解析 .docx 就抛 ValidationError】，而且错误信息指向 schema 不指向 parser。
    # 教训：字面量联合类型改了要全局搜一遍。
    source_type: Literal["pdf", "docx", "md"]
    blocks: list[Block] = Field(default_factory=list)
    raw_text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)  # 页数、字符等元信息

    @property
    def char_count(self) -> int:
        return len(self.raw_text)


class Chunk(BaseModel):
    """切分后的检索单元

    字段分三组：身份 / 定位 / 计量
    定位组是【引用可回溯】的基础，必须在切分那一刻就填好，不然事后拿文本反查是回不去的。
    """

    # --- 身份 ---
    chunk_id: str  # 确定性 ID
    doc_id: str
    text: str

    # --- 定位 ---
    clause_no: str | None = None  # "第十二条"
    clause_title: str | None = None  # "违约责任"
    section: str | None = None  # "第三章 违约责任" 章节上下文
    page: int | None = None
    char_start: int = 0
    char_end: int = 0

    # --- 计量 / 溯源 ---
    token_count: int = 0
    split_index: int = 0  # 超长条款二次切分时的序号（从 0 起）
    split_total: int = 1  # 该条款共切成几块

    @property
    def is_split(self) -> bool:
        return self.split_total > 1

    @property
    def location(self) -> str:
        """人类可读的定位串，报告里可以直接用。"""
        parts = [p for p in (self.section, self.clause_no, self.clause_title) if p]
        if parts:
            return " . ".join(parts)
        return f"{self.doc_id}#{self.char_start}"


class SearchHit(BaseModel):
    """一次检索的命中结果。

    字段与 Chunk 对齐 —— 检索出来的东西最终要变成报告里的【引用】（约定 5），
    所以定位信息必须一路带回来，不能在检索层丢掉。
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float

    # --- 定位（约定 5）---
    clause_no: str | None = None
    clause_title: str | None = None
    section: str | None = None
    page: int | None = None
    char_start: int = 0
    char_end: int = 0

    # 来源：混合检索时用来判断这一条是谁召回的
    source: Literal["dense", "bm25", "hybrid"] = "dense"

    @property
    def location(self) -> str:
        parts = [p for p in (self.section, self.clause_no, self.clause_title) if p]
        if parts:
            return " . ".join(parts)
        return f"{self.doc_id}#{self.char_start}"

"""条款级切分

三段式策略
    1. 识别结构边界：第X章 / 第X节 / 第X条（**行首锚定**）
    2. 一个条款 = 一个 chunk，携带章节路径与字符位置
    3. 超长条款优先在【子项边界】（一、 / （一） / 1.1）二次切分，其次按句子切，都不行才按字符硬切；**只在硬切时加 overlap**

为什么生产路径不用 LangChain 的 splitter：
    - 它按 tiktoken 数 token，而 bge-base-zh-v1.5 是 BERT 中文词表（一字一 token），
        两者对中文的计数可差一倍 -> 按它算不超限，送进模型已被静默截断
    - 它不区分 章/节/条，也产不出 clause_no / section 这类可回溯字段
    - 它的 overlap 全局生效，而条款边界是天然的，加重叠是纯噪声
它作为【对照组】用在 chunk_by_recursive()
"""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.config import settings
from app.core.logger import logger
from app.core.schemas import Chunk, ParsedDocument

# ---------- 结构边界正则（全部行首锚定）----------

_NUM = r"[0-9一二三四五六七八九十百零〇]+"

# ← 关键：^ 锚定不能省。合同正文里会大量出现"依照本合同第十二条的约定"，
# 不锚定行首，这些【引用】会被误判成条款标题，把文档切成碎片。
_RE_CLAUSE = re.compile(rf"^第\s*{_NUM}\s*条")
_RE_CHAPTER = re.compile(rf"^第\s*{_NUM}\s*章")
_RE_SECTION = re.compile(rf"^第\s*{_NUM}\s*节")

# 子项边界：不作为条款边界，只作为超长条款的优先切点
_RE_SUB_ITEM = re.compile(
    r"^(?:"
    r"[一二三四五六七八九十百]+[、.．]"
    r"|[（(][0-9一二三四五六七八九十百]+[）)]"
    r"|[0-9]+(?:\.[0-9]+)+"
    r")"
)

_RE_SENTENCE_END = re.compile(r"(?<=[。；;！!？?])")

# ---------- 分节哨兵：正文结束 / 进入附件 ----------
#
# ← P1 修复（Day 3 真实合同实测）。条款边界只认「第X条」，而附件、签署页、
#   评估表格里【没有条款号】—— 这些行原本全部被追加到最后一个条款名下。
#   Day 3 实测：A 文档「第十五条」吞了 33 块（占 42%），B 文档「第二十六条」吞了 7 块。
#
#   后果是 clause_no【语义错误】—— 报告引用「第十五条」去说明一个其实来自
#   附件表格的问题。这比 chunk_id 碰撞更危险：ID 碰撞只是位置丢失，
#   而这个是「ID 有效但指向错误的对象」，而且同样静默。
#
# 命中哨兵就 flush 当前条款，后续内容成为【无条款号】的独立组。
#
# 为什么不改用「条款长度上限」来兜底：
#   真正超长的【条款】应该被切成多块但都保留 clause_no —— 那是
#   _split_long_clause 的职责。长度上限无法区分「长条款」和「条款 + 尾部垃圾」，
#   用它会在解决一个错误的同时制造另一个。

# 附件标题：附件1 / 附件一 / 附件A。
# 必须带编号 —— 否则正文里的「附件列表如下」会被误判成附件起点。
_RE_APPENDIX = re.compile(r"^\s*附\s*件\s*(?:[0-9]+|[一二三四五六七八九十]+|[A-Za-z])(?=\s|$|[：:、.．])")

# 正文结束标记。纯标记行，无信息量 —— 命中后 flush 并直接丢弃这一行。
_RE_BODY_END = re.compile(r"^\s*[（(]?\s*以下(?:无正文|空白)\s*[）)]?\s*[。.]?\s*$")

# 签署区：整行以「（盖章 / 签名 / 签字…）」结尾的短行。
# 故意用「以括号结尾」而不是角色词（甲方 / 乙方）—— 这样能覆盖
# 「乙方监护人（签名或盖章）：」这类变体；同时正文里括号后面还有字，
# 不会以括号为行尾，天然被排除。
_RE_SIGNATURE = re.compile(r"^.{0,16}[（(][^）)\n]{0,16}(?:盖章|签名|签字|签章|签印)[^）)\n]{0,16}[）)]\s*[：:]?\s*$")


# ---------- token 计数 ----------


@lru_cache(maxsize=1)
def _get_tokenizer() -> PreTrainedTokenizerBase | None:
    """加载 embedding 模型自己的 tokenizer（不加载模型本体，内存开销很小）"""
    try:
        return AutoTokenizer.from_pretrained(str(settings.embedding_model_path))
    except Exception as exc:
        logger.warning(
            f"tokenizer 加载失败（{exc}），token 计数将退化为字符数。"
            f"这是**显式降级**——切分长度约束会变宽松，请检查模型目录。"
        )
        return None


def count_tokens(text: str) -> int:
    """按 embedding 模型的 tokenizer 计数

    不要用 tiktoken，也不要用"1 token ≈ 1.5~2个汉字"这类估算——对 BERT 中文词表不成立（它按字切，接近1:1）
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return len(text)
    return len(tokenizer.encode(text, add_special_tokens=True))


def max_tokens() -> int:
    """当前 embedding 模型的输入上限

    优先取 tokenizer 的 model_max_length —— 这样以后换 embedding 模型会自动跟随。
    """
    tokenizer = _get_tokenizer()
    if tokenizer is not None and 0 < tokenizer.model_max_length <= 8192:
        return int(tokenizer.model_max_length)
    return settings.embedding_max_tokens


# ---------- 内部结构 ----------


@dataclass
class _Clause:
    """切分过程中的临时结构（不是对外契约，所以不用 pydantic）"""

    lines: list[str] = field(default_factory=list)
    clause_no: str | None = None
    clause_title: str | None = None
    section: str | None = None
    page: int | None = None
    char_start: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _clause_no_of(line: str) -> str:
    match = _RE_CLAUSE.match(line)
    return line[: match.end()].replace(" ", "") if match else ""


def _clause_title_of(line: str) -> str | None:
    """从 "第十二条 违约责任" 里取出 "违约责任"。

    超过 20 字的不当作标题——那是正文（"第十二条 甲方应当在收到..."）
    """
    match = _RE_CLAUSE.match(line)
    if not match:
        return None
    rest = line[match.end() :].strip(" \t、.．:：-—_")
    if not rest or len(rest) > 20:
        return None
    return rest


def _split_into_clauses(doc: ParsedDocument) -> list[_Clause]:
    """按 章 / 节 / 条 + 分节哨兵 把 blocks 归组成条款。

    用状态机维护章节上下文——一次性正则做不到这件事。

    ← P1 修复要点：命中分节哨兵时 flush 当前条款，后续内容成为
    【无条款号】的独立组，而不是继续追加到最后一个条款名下。

    三类组的形态：
      1. 有条款号的组 —— 正文条款，clause_no / clause_title / section 齐全
      2. 无条款号的组（前言）—— 文档开头、第一条之前
      3. 无条款号的组（附件 / 签署区）—— 哨兵之后，section 为 None
    """
    clauses: list[_Clause] = []
    current: _Clause | None = None  # 当前正在收集行的组（可能有条款号，也可能没有）
    chapter: str | None = None
    section: str | None = None
    started = False  # 是否已出现第一个条款 / 哨兵

    offset = 0
    preamble: list[str] = []
    preamble_start: int | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.lines:
            clauses.append(current)
        current = None

    def open_unnumbered(line: str, start: int, page: int | None, title: str | None, sec: str | None) -> _Clause:
        """开一个【无条款号】的组 —— 附件、签署区、章节引言都走这里。"""
        return _Clause(lines=[line], clause_no=None, clause_title=title, section=sec, page=page, char_start=start)

    for block in doc.blocks:
        line = block.text
        line_start = offset
        offset += len(line) + 1  # raw_text 用 "\n" 连接

        # --- 1) 条款边界（最高优先级）---
        if _RE_CLAUSE.match(line):
            flush()
            started = True
            current = _Clause(
                lines=[line],
                clause_no=_clause_no_of(line),
                clause_title=_clause_title_of(line),
                section=section or chapter,  # 存最具体的可用上下文：有节用节，否则用章。
                page=block.page,
                char_start=line_start,
            )
            continue

        # --- 2) 章节边界（只更新上下文，本身不是条款）---
        if _RE_SECTION.match(line):
            flush()
            section = line
            continue

        if _RE_CHAPTER.match(line):
            flush()
            chapter = line
            section = None  # 换章必须清空节，否则会串台
            continue

        # --- 3) 分节哨兵 ---
        if _RE_BODY_END.match(line):
            flush()
            started = True
            continue  # 「以下无正文」是纯标记，无信息量，丢弃这一行

        if _RE_APPENDIX.match(line):
            # 附件标题是真正的分界线，永远开新组
            flush()
            started = True
            current = open_unnumbered(
                line,
                line_start,
                block.page,
                line.strip(),  # 附件标题作为组标题
                None,  # 附件【不属于任何章节】—— 挂上章节名会误导检索
            )
            continue

        if _RE_SIGNATURE.match(line):
            started = True
            if current is None or current.clause_no is not None:
                # 只有当前是【有条款号的正文】时才断开。
                # 若已经在尾部组里就继续追加 —— 否则「甲方（盖章）」「乙方（签名）」
                # 「日期」会各自变成独立小块，制造一堆 5 个字的噪声 chunk。
                flush()
                current = open_unnumbered(line, line_start, block.page, None, None)
            else:
                current.lines.append(line)
            continue

        # --- 4) 普通行 ---
        if current is not None:
            current.lines.append(line)
        elif started:
            # 已进入正文但没有当前条款（章节引言 / flush 之后的续文）——
            # 开一个无条款号的组。绝【不能】倒回 preamble：
            # 那会把文档中段的内容插到文档最开头，位置信息全乱。
            current = open_unnumbered(line, line_start, block.page, None, section or chapter)
        else:
            if preamble_start is None:
                preamble_start = line_start
            preamble.append(line)

    flush()

    # 前言（合同标题、当事人、鉴于条款）也要保留——要素抽取需要
    if preamble:
        clauses.insert(0, _Clause(lines=preamble, section=None, page=None, char_start=preamble_start or 0))

    return clauses


# ---------- 超长条款二次切分 ----------


def _pack_at_boundaries(text: str, cut_points: list[int], max_chars: int, overlap: int) -> list[tuple[int, str]]:
    """在理想边界上贪心打包，返回 [(相对偏移, 片段), ...]。

    切在理想边界上**不加 overlap**（边界本身是干净的）；
    只有被迫硬切时才回退 overlap。
    """
    boundaries = sorted({0, *(p for p in cut_points if 0 < p < len(text)), len(text)})
    pieces: list[tuple[int, str]] = []
    start = 0

    while start < len(text):
        limit = start + max_chars
        if limit >= len(text):
            pieces.append((start, text[start:]))
            break

        candidates = [b for b in boundaries if start < b <= limit]
        if candidates:
            end = candidates[-1]
            pieces.append((start, text[start:end]))
            start = end  # 理想边界：不重叠
        else:
            end = limit
            pieces.append((start, text[start:end]))
            start = max(end - overlap, start + 1)  # 硬切：重叠

    return pieces


def _split_long_clause(text: str, max_chars: int, overlap: int) -> list[tuple[int, str]]:
    """超长条款二次切分：子项边界 > 句子边界 > 字符硬切。"""
    if len(text) <= max_chars:
        return [(0, text)]

    sub_points = [m.start() for m in _RE_SUB_ITEM.finditer(text, re.MULTILINE)]
    if len(sub_points) >= 2:
        return _pack_at_boundaries(text, sub_points, max_chars, overlap)

    sentence_points = [m.end() for m in _RE_SENTENCE_END.finditer(text)]
    if len(sentence_points) >= 2:
        return _pack_at_boundaries(text, sentence_points, max_chars, overlap)

    # 兜底：既无子项也无句读（比如一长串表格文本）
    pieces: list[tuple[int, str]] = []
    step = max(1, max_chars - overlap)
    for start in range(0, len(text), step):
        pieces.append((start, text[start : start + max_chars]))
    return pieces


# ---------- Chunk 构造 ----------


def _make_chunk_id(
    doc_id: str,
    clause_no: str | None,
    index: int,
    total: int,
    char_start: int,
    text: str,
) -> str:
    """确定性 ID。

    ← 关键 1：不要用 uuid4。确定性 ID + Milvus upsert = 重复上传不产生副本（W3 幂等）。

    ← 关键 2：**必须包含 char_start**。Day 3 用真实合同实测暴露过这个问题：
      同一份 PDF 里重复出现的条款（附件重排、末页错位）文本完全相同，
      此时 `doc_id|clause_no|index/total|text` 会产出**相同 ID**。
      以 chunk_id 作 Milvus 主键时，后写入的块会**静默覆盖**先写入的块 ——
      文本虽然一样，但 page / char_start 不同，**引用会指向错误的位置**。

      Day 3 实测数据：78 块 / 76 唯一 ID / 2 组碰撞，被覆盖的块分别位于
      page=2 与 page=51（同一段文本，不同位置）。

      合成 fixture 测不出这个问题（那里没有重复文本），
      所以 tests/test_chunker.py 里补了针对性的回归用例。
    """
    raw = f"{doc_id}|{char_start}|{clause_no or ''}|{index}/{total}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _build_chunk(
    doc_id: str,
    clause: _Clause,
    text: str,
    *,
    split_index: int,
    split_total: int,
    char_start: int,
) -> Chunk:
    tokens = count_tokens(text)
    limit = max_tokens()

    # ← 关键：让超限可见。不检查的话，SentenceTransformer 会静默截断，
    # 症状是"某个条款召回率莫名偏低"，排查成本极高。
    if tokens > limit:
        logger.warning(
            f"块超过模型上限：{tokens} > {limit} tokens"
            f"（{clause.clause_no or '前言'} 第 {split_index + 1}/{split_total} 块）"
            f"—— 会被静默截断，请调小 CHUNK_MAX_CHARS"
        )

    return Chunk(
        chunk_id=_make_chunk_id(doc_id, clause.clause_no, split_index, split_total, char_start, text),
        doc_id=doc_id,
        text=text,
        clause_no=clause.clause_no,
        clause_title=clause.clause_title,
        section=clause.section,
        page=clause.page,
        char_start=char_start,
        char_end=char_start + len(text),
        token_count=tokens,
        split_index=split_index,
        split_total=split_total,
    )


# ---------- 对外接口 ----------


def chunk_by_clause(doc: ParsedDocument, *, max_chars: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """条款级切分（生产策略）。"""
    max_chars = max_chars or settings.chunk_max_chars
    overlap = overlap or settings.chunk_overlap

    chunks: list[Chunk] = []
    for clause in _split_into_clauses(doc):
        pieces = (
            [(0, clause.text)] if len(clause.text) <= max_chars else _split_long_clause(clause.text, max_chars, overlap)
        )
        total = len(pieces)
        for index, (local_offset, piece_text) in enumerate(pieces):
            chunks.append(
                _build_chunk(
                    doc.doc_id,
                    clause,
                    piece_text,
                    split_index=index,
                    split_total=total,
                    char_start=clause.char_start + local_offset,
                )
            )
    return chunks


def chunk_by_length(doc: ParsedDocument, *, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """固定长度兜底：文档完全没有条款编号时使用（比如纯文字说明）。"""
    size = size or settings.chunk_max_chars
    overlap = overlap or settings.chunk_overlap
    text = doc.raw_text

    flat = _Clause(lines=[], section=None, page=None, char_start=0)
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    starts = list(range(0, len(text), step)) or [0]

    for index, start in enumerate(starts):
        piece = text[start : start + size]
        if not piece:
            break
        chunks.append(
            _build_chunk(
                doc.doc_id,
                flat,
                piece,
                split_index=index,
                split_total=len(starts),
                char_start=start,
            )
        )
    return chunks


def chunk_by_recursive(doc: ParsedDocument, *, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """对照组：LangChain 的 RecursiveCharacterTextSplitter（W2 §14.2）。

    它是**行业默认基线**，用真库比我自研一个朴素切分器有说服力得多。

    ← 关键：length_function=len，即按【字符数】而非 token 数切。
    这样和 chunk_by_clause 的 max_chars 是同一个度量——**度量不一致的对比是无效对比**。
    默认的 tiktoken 计数对中文与 BERT 词表差异很大，会把实验搅浑。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    size = size or settings.chunk_max_chars
    overlap = overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks: list[Chunk] = []
    offset = 0
    pieces = splitter.split_text(doc.raw_text)

    for index, piece in enumerate(pieces):
        # 老实说：它不保证块从条款头开始，能对上的就填，对不上就是 None
        first_line = piece.strip().splitlines()[0] if piece.strip() else ""
        clause_no = _clause_no_of(first_line) if _RE_CLAUSE.match(first_line) else None

        clause = _Clause(
            lines=[piece],
            clause_no=clause_no or None,
            clause_title=_clause_title_of(first_line) if clause_no else None,
            section=None,
            page=None,
            char_start=offset,
        )
        chunks.append(
            _build_chunk(
                doc.doc_id,
                clause,
                piece,
                split_index=index,
                split_total=len(pieces),
                char_start=offset,
            )
        )
        offset += len(piece)

    return chunks


_STRATEGIES = {
    "clause": chunk_by_clause,
    "recursive": chunk_by_recursive,
    "length": chunk_by_length,
}


def chunk_document(doc: ParsedDocument, strategy: str = "clause", **kwargs) -> list[Chunk]:
    """切分入口。

    clause 策略带**自动降级**：如果绝大多数块都没识别出条款号，
    说明这份文档没有条款结构，退回固定长度切分并记 warning。
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}，可选：{list(_STRATEGIES)}")

    chunks = _STRATEGIES[strategy](doc, **kwargs)

    if strategy == "clause":
        numbered = sum(1 for c in chunks if c.clause_no)
        # ← 原来这里有个 `body` 变量，算出来却从未使用（死代码），已删除。
        # 比例的分母必须是【全部块】：前言没有条款号是正常的，
        # 它本来就该拉低这个比例——这正是"文档无条款结构"的信号。
        if len(chunks) >= 3 and numbered / len(chunks) < 0.3:
            logger.warning(
                f"{doc.filename} 条款识别覆盖率仅 {numbered}/{len(chunks)}，疑似无条款结构，退回固定长度切分"
            )
            return chunk_by_length(doc, **kwargs)

    return chunks

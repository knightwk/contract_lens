"""切分器回归测试。

重点防两类错误：
  1. 【静默错误】正文里的条款引用（"本合同第四条"）被误判成条款标题
  2. 【长度失守】块超过 embedding 模型输入上限 -> 被静默截断

设计原则：**不依赖 fixture 的精确长度**。
凡是"这段文本应该足够长"这类假设，都写成独立的守卫测试
（见 test_fixture_has_over_limit_clause）——否则编辑 fixture
会让测试退化成"恒真"，看起来还在跑，实际什么都没验证。
"""

from pathlib import Path

import pytest

from app.config import settings
from app.core.chunker import (
    _split_long_clause,
    chunk_by_clause,
    count_tokens,
    max_tokens,
)
from app.core.parser import parse_document
from app.core.schemas import Block, ParsedDocument

FIXTURE = Path(__file__).parent / "fixtures" / "test_contract.md"

EXPECTED_CLAUSES = {f"第{n}条" for n in "一二三四五六七八"}


@pytest.fixture(scope="module")
def doc():
    return parse_document(FIXTURE)


@pytest.fixture(scope="module")
def chunks(doc):
    return chunk_by_clause(doc)


# ---------- 结构正确性 ----------


def test_clause_set_is_exact(chunks):
    """识别出的条款集合必须精确等于这 8 条。

    ← 这是最值钱的一条：多出任何一条，都说明正文里的
    "本合同第四条"这类【引用】被当成了条款标题。
    """
    assert {c.clause_no for c in chunks if c.clause_no} == EXPECTED_CLAUSES


def test_preamble_is_kept(chunks):
    """前言（标题 + 当事人）必须保留——要素抽取要靠它。"""
    assert chunks[0].clause_no is None
    assert "某某科技有限公司" in chunks[0].text


def test_every_body_chunk_has_section(chunks):
    for c in chunks:
        if c.clause_no:
            assert c.section, f"{c.clause_no} 缺 section"


def test_sub_items_stay_in_parent_clause(chunks):
    """一、/（一）/1.1 是第三条的子项，不该变成独立条款。"""
    merged = "".join(c.text for c in chunks if c.clause_no == "第三条")
    assert "1.2 报表导出" in merged
    assert "三、上线支持" in merged


# ---------- 长度约束 ----------


def test_fixture_has_over_limit_clause(chunks):
    """守卫测试：fixture 里必须存在被二次切分的条款。

    没有这一条，test_long_clause_split_by_sub_items 会在
    fixture 变短后【静默恒真】——测试还在跑，但什么都没验证。
    把隐式假设写成显式断言，是让测试保持诚实的唯一办法。
    """
    assert any(c.is_split for c in chunks), (
        f"fixture 里没有超过 CHUNK_MAX_CHARS={settings.chunk_max_chars} 的条款，"
        f"二次切分的端到端路径未被覆盖。"
        f"请加长 tests/fixtures/test_contract.md 的第五条。"
    )


def test_long_clause_split_by_sub_items(chunks):
    """第五条超长且含子项，应按【子项边界】切分。"""
    five = [c for c in chunks if c.clause_no == "第五条"]
    assert len(five) > 1
    assert {c.split_total for c in five} == {len(five)}
    assert [c.split_index for c in five] == list(range(len(five)))
    assert all(len(c.text) <= settings.chunk_max_chars for c in five)


def test_split_long_clause_falls_back_to_sentences():
    """无子项标记时退回句子边界。

    fixture 覆盖不到这条分支，所以直接单元测——比再往 fixture 里塞一个
    长条款更聚焦，也不受 fixture 长度影响。
    """
    sentence = "甲方应当在收到成果后五个工作日内完成审核并出具书面意见。"
    text = sentence * 12                                   # 约 340 字
    pieces = _split_long_clause(text, max_chars=100, overlap=20)

    assert len(pieces) > 1
    assert all(len(piece) <= 100 for _, piece in pieces), "有片段超过 max_chars"
    assert pieces[0][0] == 0
    for offset, piece in pieces:                           # 偏移要能对回原文
        assert text[offset : offset + len(piece)] == piece


def test_no_chunk_exceeds_model_limit(chunks):
    limit = max_tokens()
    over = [(c.clause_no, c.token_count) for c in chunks if c.token_count > limit]
    assert not over, f"超限块会被静默截断：{over}"


# ---------- 幂等与可回溯 ----------


def test_chunk_id_is_deterministic(doc):
    """同一份文档跑两次，ID 必须一致——W3 幂等的前提。"""
    a = [c.chunk_id for c in chunk_by_clause(doc)]
    b = [c.chunk_id for c in chunk_by_clause(doc)]
    assert a == b


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_char_offsets_point_back_to_source(doc, chunks):
    """← char_start/char_end 必须能定位回 raw_text。

    这是「引用可回溯」（约定 5）的底座——偏移错了，
    报告里的引用就指不回原文，而且不会有任何报错。
    """
    for c in chunks:
        assert doc.raw_text[c.char_start : c.char_end] == c.text, (
            f"{c.location} 的字符偏移对不上原文"
        )


def test_token_count_comes_from_tokenizer(chunks):
    """token_count 必须来自真实 tokenizer，不能用字符数糊弄。"""
    for c in chunks:
        assert c.token_count == count_tokens(c.text)


# ---------- P0 回归：chunk_id 碰撞 ----------


def _doc_with_duplicated_body() -> ParsedDocument:
    """构造一份【同一段条款出现两次】的文档。

    模拟真实场景：PDF 附件重排、末页错位 —— 同一段文本在文档里出现两遍。
    Day 3 的养老机构合同 PDF 就是这样（末页出现在物理第 2 页）。
    """
    body = [
        "第一条 送医治疗委托事项",
        "1.当乙方突发疾病或出现身体损伤等紧急情况时，甲方应当立即联系医疗机构。",
        "第二条 乙方监护人、丙方的责任",
        "1.承担乙方就医治疗所发生的全部费用。",
    ]
    blocks = [Block(text=line, page=1, kind="paragraph") for line in body] * 2
    return ParsedDocument(
        doc_id="duptest00000000",
        filename="dup.md",
        source_type="md",
        blocks=blocks,
        raw_text="\n".join(b.text for b in blocks),
    )


def test_chunk_id_does_not_collide_on_duplicate_text():
    """← P0 回归用例（Day 3 真实合同实测暴露）。

    同一份文档里出现【文本完全相同的条款】时，chunk_id 不能碰撞。
    Day 3 实测：78 块 / 76 唯一 ID / 2 组碰撞，被覆盖的两块分别位于
    page=2 与 page=51 —— 文本相同、位置不同，引用会指向错误的位置。

    **合成 fixture 不含重复文本，所以这个用例手工构造文档。**
    不这么做的话，这个 bug 修好之后会再次静默回归。
    """
    chunks = chunk_by_clause(_doc_with_duplicated_body())

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), (
        f"chunk_id 碰撞：{len(ids)} 块只有 {len(set(ids))} 个唯一 ID —— "
        f"以 chunk_id 作主键时会静默覆盖（Day 3 真实数据踩过）"
    )

    # 文本相同但位置不同的两块，必须都保留下来
    firsts = [c for c in chunks if c.clause_no == "第一条"]
    assert len(firsts) == 2, "同名的两处条款都应成为独立块"
    assert firsts[0].text == firsts[1].text, "本例的前提是两块文本相同"
    assert firsts[0].char_start != firsts[1].char_start, "但位置必须不同"


def test_chunk_id_still_deterministic_with_position():
    """加了位置信息之后，确定性不能被破坏（W3 幂等仍要成立）。"""
    doc = _doc_with_duplicated_body()
    a = [c.chunk_id for c in chunk_by_clause(doc)]
    b = [c.chunk_id for c in chunk_by_clause(doc)]
    assert a == b


# ---------- P1 回归：尾部内容不得挂到最后一个条款 ----------


def _doc_with_appendix_tail() -> ParsedDocument:
    """构造「正文 + 以下无正文 + 签署区 + 两个附件」的文档。

    模拟真实合同结构（Day 3 的养老机构服务合同就是这个形态）：
    附件和签署页里没有「第X条」，所以原本全被追加到最后一个条款名下。
    """
    lines = [
        "测试服务合同",
        "第一条 服务内容",
        "乙方应当提供系统开发服务。",
        "第二条 合同生效",
        "本合同自双方签字之日起生效。",
        "以下无正文",
        "甲方（盖章）：",
        "乙方监护人（签名或盖章）：",
        "日期：____年__月__日",
        "附件1 技术服务清单",
        "序号 服务项 数量 单价",
        "1 需求分析 1 10000",
        "附件2 验收标准",
        "系统应当通过全部验收用例。",
    ]
    blocks = [Block(text=text, page=1, kind="paragraph") for text in lines]
    return ParsedDocument(
        doc_id="tailtest00000000",
        filename="tail.md",
        source_type="md",
        blocks=blocks,
        raw_text="\n".join(b.text for b in blocks),
    )


def test_tail_content_is_not_attached_to_last_clause():
    """← P1 回归用例（Day 3 真实合同实测暴露）。

    修复前：附件与签署区全部被追加到最后一个条款名下。
    Day 3 实测 A 文档「第十五条」占 33/78 块（42%），B 文档「第二十六条」占 7/51。
    后果是 clause_no【语义错误】—— 引用指向错误的条款，而且静默发生。
    """
    chunks = chunk_by_clause(_doc_with_appendix_tail())

    second = [c for c in chunks if c.clause_no == "第二条"]
    assert len(second) == 1, "第二条不该被切成多块"

    body = second[0].text
    assert "本合同自双方签字之日起生效。" in body
    assert "附件" not in body, f"附件内容串进了条款：{body!r}"
    assert "盖章" not in body and "签名" not in body, f"签署区串进了条款：{body!r}"


def test_appendix_groups_are_unnumbered_and_titled():
    """附件应成为【无条款号】的独立组，并以附件标题作为组标题。"""
    chunks = chunk_by_clause(_doc_with_appendix_tail())

    appendix = [c for c in chunks if c.clause_title and c.clause_title.startswith("附件")]
    assert {c.clause_title for c in appendix} == {"附件1 技术服务清单", "附件2 验收标准"}
    assert all(c.clause_no is None for c in appendix), "附件不应带条款号"
    assert all(c.section is None for c in appendix), "附件不应继承章节（会误导检索）"

    # 两个附件的内容各自独立，不能互相串
    first = next(c for c in appendix if c.clause_title == "附件1 技术服务清单")
    assert "需求分析" in first.text
    assert "验收用例" not in first.text


def test_body_end_marker_is_dropped():
    """「以下无正文」是纯标记行，不应出现在任何块里。"""
    chunks = chunk_by_clause(_doc_with_appendix_tail())
    assert not any("以下无正文" in c.text for c in chunks)


def test_signature_lines_share_one_group():
    """签署区的多行应归入同一个组，不能每行一个 5 字噪声块。"""
    chunks = chunk_by_clause(_doc_with_appendix_tail())

    sig = [c for c in chunks if c.clause_no is None and "盖章" in c.text]
    assert len(sig) == 1, f"签署区被切成了 {len(sig)} 组"
    assert "乙方监护人" in sig[0].text
    assert "日期" in sig[0].text
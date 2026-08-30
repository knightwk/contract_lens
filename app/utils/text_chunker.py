import re
from typing import List


def chunk_contract_by_clauses(full_text: str) -> List[dict[str, str]]:
    """
    按合同条款（第X条）切割文本
    :return: List[dict[str, str]]，每个 dict 包含 chunk_id, title, content
    """
    # 1. 找出所有 "第X条" 在全文中的位置
    pattern = r"第[一二三四五六七八九十百]+条 "
    matches = list(re.finditer(pattern, full_text))

    if not matches:
        return [{"chunk_id": 0, "title": "全文", "content": full_text}]

    chunks = []
    chunk_id = 0
    # 2. 处理：如果第一个匹配的位置 > 0，说明开头有"前置声明"
    if matches[0].start() > 0:
        chunks.append(
            {
                "chunk_id": chunk_id,
                "title": "前置声明",
                "content": full_text[: matches[0].start()].strip(),
            }
        )
        chunk_id += 1

    # 3. 循环遍历 matches，提取每一条的内容
    for idx in range(len(matches)):
        title = matches[idx].group().strip()

        start_pos = matches[idx].end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        content = full_text[start_pos:end_pos].strip()

        chunks.append({"chunk_id": chunk_id, "title": title, "content": content})

        chunk_id += 1

    return chunks


if __name__ == "__main__":
    test_text = """本合同由甲方与乙方签订。
    第一条 合作内容
    甲方负责提供数据，乙方负责提供算力。
    第二条 付款方式
    乙方应在每月5日前支付上月服务费。
    第三条 保密条款
    双方应对合作内容严格保密。
    第四条 违约责任
    违约方应赔偿守约方全部损失。"""

    chunks = chunk_contract_by_clauses(test_text)
    for c in chunks:
        print(f"[{c['chunk_id']}] {c['title']} -> {c['content'][:30]}...")

from typing import Tuple

import fitz

from app.core.exceptions import FileHandlingError


def extract_text_from_pdf_bytes(file_bytes: bytes) -> Tuple[str, int]:
    """
    从 PDF 字节流中提取纯文本
    :return: （提取的全文，总页数），如果 PDF 是扫描件（无文本），返回（"", 总页数）
    """
    try:
        # 1. 使用 fitz.open 从内存字节流打开 PDF 文件
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise FileHandlingError(f"无法打开 PDF 文件: {e}")

    # 2. 遍历每一页，提取文本并累加
    full_text = ""
    total_pages = doc.page_count
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += text
    doc.close()

    # 3. 清洗文本 - 把所有换行符、多余空格合并成单个空格
    cleaned_text = " ".join(full_text.split())

    return cleaned_text, total_pages

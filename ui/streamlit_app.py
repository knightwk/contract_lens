"""ContractLens 前端：上传合同 → 提问 → 看答案与引用。

为什么要有 UI
    可演示性对面试的价值 —— 面试官更愿意看一个能点的东西，
    而不是听你描述 curl 返回什么。

为什么调 HTTP API 而不是直接 import app.core.*
    ① 真实项目里 UI 和 API 是两个进程，这一步就是"前后端联调"
    ② 顺带验证 API 契约：如果 UI 拿不到它要显示的东西，
       说明契约设计有缺口 —— 这正是 Day 6 设 Citation 时想解决的问题
    ③ 直接 import 的话，接口设计得好不好永远得不到检验

启动（两个终端）：
    uvicorn app.main:app --reload
    streamlit run ui/streamlit_app.py
"""

import os

import httpx
import streamlit as st

# 上传接口能接受的类型，与后端 _ALLOWED_SUFFIXES 保持一致
ACCEPTED_TYPES = ["pdf", "docx", "md"]

# 默认端口故意不用 8000：开发机上 8000 极其容易被别的服务占住，
# 而且 Windows 允许不同进程分别监听 0.0.0.0:8000 和 127.0.0.1:8000 ——
# 结果是"端口在监听，但请求被别的服务抢走"，排查起来很费时间。
# 需要改时用环境变量，不用改代码。
DEFAULT_API = os.environ.get("CONTRACTLENS_API", "http://127.0.0.1:8010")


def api_get(client: httpx.Client, path: str) -> tuple[bool, dict | str]:
    try:
        r = client.get(path)
        return r.status_code == 200, r.json()
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def post_json(client: httpx.Client, path: str, payload: dict) -> tuple[bool, dict | str]:
    try:
        r = client.post(path, json=payload)
    except httpx.HTTPError as exc:
        return False, f"连不上 API：{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return False, _explain_error(r)
    return True, r.json()


def post_file(client: httpx.Client, path: str, name: str, data: bytes, mime: str) -> tuple[bool, dict | str]:
    try:
        r = client.post(path, files={"file": (name, data, mime)})
    except httpx.HTTPError as exc:
        return False, f"连不上 API：{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return False, _explain_error(r)
    return True, r.json()


def _explain_error(resp: httpx.Response) -> str:
    """把后端的结构化错误翻成人话。

    后端约定 detail 是 {"code": ..., "message": ...}（见 query.py），
    但 Pydantic 的 422 是另一种形状（列表），这里都要兜住。
    """
    try:
        detail = resp.json().get("detail")
    except Exception:  # noqa: BLE001
        return f"HTTP {resp.status_code}: {resp.text[:200]}"

    if isinstance(detail, dict):
        return f"[{resp.status_code} {detail.get('code', '')}] {detail.get('message', '')}"
    if isinstance(detail, list):
        return f"[{resp.status_code} 参数校验失败] " + "；".join(
            f"{'.'.join(str(x) for x in item.get('loc', []))}: {item.get('msg', '')}" for item in detail
        )
    return f"[{resp.status_code}] {detail}"


def render_citation(index: int, cit: dict) -> None:
    """一条引用：定位 + 条目 ID + 原文。"""
    with st.container(border=True):
        head, badge = st.columns([5, 1])
        head.markdown(f"**{index}. {cit.get('clause_location', '（无定位）')}**")
        badge.caption(f"`{cit.get('chunk_id', '')[:12]}`")
        st.text(cit.get("original_text", ""))
        st.caption(f"文档 {cit.get('doc_id', '?')} · 条目 {cit.get('chunk_id', '?')}")


def render_answer(body: dict) -> None:
    if not body.get("grounded", True):
        st.warning("这次没有找到能支撑回答的条款 —— 下面是模型的原话。")

    st.markdown("### 回答")
    st.markdown(body.get("answer", ""))

    cits = body.get("citations", [])
    st.markdown(f"### 引用（{len(cits)} 条）")
    if not cits:
        st.info("没有引用。如果这不是「材料不足」的情况，说明引用提取出了问题。")
    for i, cit in enumerate(cits, 1):
        render_citation(i, cit)

    with st.expander("分段耗时（毫秒）"):
        timings = body.get("timings", {})
        segments = timings.get("segments", {})
        total = timings.get("total_ms", 0)
        accounted = sum(segments.values())
        st.json(timings)
        st.caption(
            f"total {total:.0f} ms，各段合计 {accounted:.0f} ms，"
            f"未计入 {total - accounted:.0f} ms —— 差额过大说明有没打点的环节"
        )


def main() -> None:
    st.set_page_config(page_title="ContractLens", page_icon="📄", layout="wide")

    with st.sidebar:
        st.title("📄 ContractLens")
        api = st.text_input("API 地址", value=DEFAULT_API)
        st.caption("先启动后端：`uvicorn app.main:app --reload`")

        with httpx.Client(base_url=api, timeout=120.0) as client:
            ok, info = api_get(client, "/health")
            if ok:
                st.success("后端在线")
                st.caption(f"models_ready = {info.get('models_ready')}")
                with st.expander("健康检查详情"):
                    st.json(info)
            else:
                st.error("后端未就绪")
                st.code(str(info), language="text")
                # 最常见的三种情况，按顺序给出来，省一次来回
                st.warning(
                    "排查顺序：\n\n"
                    "1. **后端终端有没有报错** —— lifespan 里 Milvus 连不上会直接拒绝启动（fail-fast）\n"
                    "2. **端口是不是被别的服务占了** —— Windows 允许不同进程分别监听 "
                    "`0.0.0.0:8000` 和 `127.0.0.1:8000`，端口显示 LISTENING 不代表是你的服务\n"
                    "3. **返回体长得对不对** —— 我们的 /health 返回带 status/app/models_ready；"
                    "如果看到 `error` 或 `detail` 这类字段，说明这个端口上不是我们的服务\n\n"
                    "确认端口占用：`netstat -ano | Select-String \":8010\"`"
                )
                st.stop()

        if st.button("清空本次会话"):
            st.session_state.clear()
            st.rerun()

    if "doc" not in st.session_state:
        st.header("① 上传合同")
        uploaded = st.file_uploader("选择合同文件", type=ACCEPTED_TYPES)
        if uploaded is not None and st.button("上传并入库", type="primary"):
            with httpx.Client(base_url=api, timeout=600.0) as client, st.spinner("解析、切分、编码、入库中…"):
                ok, result = post_file(
                    client,
                    "/api/v1/documents/upload",
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                )
            if ok:
                st.session_state["doc"] = result
                st.rerun()
            else:
                st.error(result)
        return

    doc = st.session_state["doc"]
    st.header("② 提问")
    st.caption(
        f"已入库 **{doc.get('filename')}** · 块 {doc.get('chunks')}（带条款号 {doc.get('numbered')}）"
        f" · 写入 {doc.get('written')} · 耗时 {doc.get('duration_ms', 0):.0f} ms"
    )

    with st.form("ask"):
        question = st.text_input("你的问题", placeholder="例如：合同什么时候生效")
        top_k = st.slider("检索条数", min_value=1, max_value=20, value=10)
        submitted = st.form_submit_button("提问", type="primary")

    if submitted:
        if not question.strip():
            st.warning("问题不能为空")
        else:
            with httpx.Client(base_url=api, timeout=600.0) as client, st.spinner("检索并生成中…"):
                ok, body = post_json(client, "/api/v1/query", {"question": question, "top_k": top_k})
            if ok:
                st.session_state["answer"] = body
            else:
                st.error(body)

    if "answer" in st.session_state:
        render_answer(st.session_state["answer"])


if __name__ == "__main__":
    main()

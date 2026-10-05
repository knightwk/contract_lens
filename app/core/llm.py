"""LLM 调用封装：异步、流式、重试、token 计数。

约定
    · 所有 LLM 访问都经过这里，业务代码不直接 import openai
      （与 model_runtime 收口模型、vector_store 收口向量库是同一个原则）
    · 审查 / 抽取类任务用低温（settings.llm_temperature 默认 0.1）——
      出错代价高，需要确定性输出
    · token 计数用 tiktoken，用于上下文预算与成本统计

为什么用 AsyncOpenAI 而不是同步 SDK
    FastAPI 是异步的，同步 SDK 会阻塞事件循环 —— 与 model_runtime / vector_store
    是同一个原则。区别在于这里的阻塞来自网络 IO，不是 CPU，所以【不需要】丢线程池：
    AsyncOpenAI 本身就是非阻塞的。
"""

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import tiktoken
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.logger import logger


class LLMError(Exception):
    """LLM 调用失败（重试耗尽后仍失败）。"""


@dataclass
class LLMResult:
    """一次非流式调用的结果。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    elif isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _log_retry(state) -> None:
    logger.warning(
        f"LLM 第 {state.attempt_number} 次尝试失败，{state.next_action.sleep:.1f}s 后重试：{state.outcome.exception()}"
    )


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._encoding = None

    # ---------- 连接 ----------

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.llm_api_key:
                raise LLMError(
                    "LLM_API_KEY 未配置。请在 .env 里填上，或先跑 "
                    "`python scripts/rag_demo.py --dry-run`（那一半不需要 key）。"
                )
            logger.info(f"初始化 LLM 客户端: {settings.llm_base_url} / {settings.llm_model_name}")
            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
                timeout=settings.llm_timeout,
            )
        return self._client

    # ---------- token 计数 ----------

    def count_tokens(self, text: str) -> int:
        """用 tiktoken 估算 token 数。

        ⚠️ 这里用 tiktoken 是【估算】不是精确值：Qwen 用的是自己的词表，
        tiktoken 对中文的切分和它不一样。用途是控制上下文预算和成本量级，
        不是精确计费 —— 精确值以 API 返回的 usage 为准。
        """
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001
                logger.warning("tiktoken 词表加载失败，token 计数将退化为按字符数估算")
                self._encoding = False
        if self._encoding is False:
            return len(text)
        return len(self._encoding.encode(text))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """估算一整组消息的 token 数（含每条消息的少量结构开销）。"""
        total = 0
        for message in messages:
            total += self.count_tokens(str(message.get("content", ""))) + 4
        return total + 2

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        """执行一次 chat.completions.create，按策略重试，成功返回原始 response。

        ⚠️ 只包【单次 HTTP 调用】。
           不要把流式消费（async for）包进来 —— 已经吐给调用方的内容收不回来，
           重试会变成让用户看到重复文字。
        """
        client = self._get_client()
        retries = -1
        try:
            for attempt in Retrying(
                retry=retry_if_exception(_is_retryable),
                stop=stop_after_attempt(settings.llm_max_retries + 1),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
                before_sleep=_log_retry,
            ):
                with attempt:
                    response = await client.chat.completions.create(**kwargs)
                retries += 1
            return response
        except OpenAIError as exc:
            raise LLMError(f"LLM 调用失败（已重试 {retries} 次）：{exc}") from exc

    # ---------- 非流式 ----------

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """一次性拿到完整回答。

        重试策略由 _create_with_retry 统一负责；这里只管计时与结果封装。
        """
        started = time.perf_counter()
        response = await self._create_with_retry(
            model=settings.llm_model_name,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            model=settings.llm_model_name,
            raw=response.model_dump(),
        )

    # ---------- 流式 ----------

    async def stream(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """逐块吐出回答文本。

        流式的价值：首字延迟从"等整段生成完"降到"等第一个 token"。
        审查报告可能几百字，用户等 5 秒才有反应和 0.5 秒就开始出字，
        体感差别很大。

        ⚠️ 重试【只覆盖建立连接那一步】（见 _create_with_retry）。
           下面消费流的 async for 故意不在重试范围内 ——
           已经吐给调用方的内容收不回来，重试会让用户看到重复的文字。
           一旦开始输出再出错，就把异常直接抛给调用方。
        """
        response = await self._create_with_retry(
            model=settings.llm_model_name,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ---------- 清理 ----------

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


llm_client = LLMClient()

"""应用配置：所有可调参数集中在此，通过环境变量覆盖。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：app/config.py -> app/ -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_model_path(model_name: str, models_dir: Path) -> Path:
    """把模型名解析成绝对路径。

    - 已经是绝对路径 -> 原样返回（支持指向 HuggingFace 缓存目录）
    - models_dir 是相对路径 -> 相对【项目根】解析，而不是当前工作目录
    """
    candidate = Path(model_name)
    if candidate.is_absolute():
        return candidate

    base = models_dir if models_dir.is_absolute() else PROJECT_ROOT / models_dir
    return base / model_name


class Settings(BaseSettings):
    """从环境变量 / .env 加载配置。

    命名约定：字段名小写下划线，环境变量大写下划线。
    case_sensitive=False 让 pydantic 自动做大小写无关匹配，
    所以 APP_NAME 能正确绑定到 app_name。
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === App ===
    app_name: str = "ContractLens"
    debug: bool = False  # 默认取安全值：生产忘记配置时不会暴露 /docs
    log_level: str = "INFO"

    # === LLM ===
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model_name: str = ""

    # === 模型 Provider 总开关 ===
    embedding_provider: Literal["local", "api"] = "local"
    rerank_provider: Literal["local", "api"] = "local"

    # === 本地模型 ===
    models_dir: Path = Path("./models")
    embedding_model_name: str = "bge-base-zh-v1.5"
    rerank_model_name: str = "bge-reranker-base"

    # 向量维度。注意：换 embedding 模型必须改这里，
    # 且 Milvus 集合必须重建 + 全部文档重新向量化。
    # bge-base-zh=768 / bge-small-zh=512 / bge-large-zh=1024 / bge-m3=1024
    embedding_dim: int = 768

    torch_num_threads: int = 4

    # === 模型 API 备用（provider=api 时生效）===
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_api_model: str = ""
    rerank_api_base: str = ""
    rerank_api_key: str = ""
    rerank_api_model: str = ""

    # === 推理调度（本地模型）===
    max_concurrent_inference: int = 2
    embedding_batch_size: int = 32
    rerank_batch_size: int = 16
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    rerank_max_length: int = 512  # W2 优化时降到 256
    model_api_timeout: float = 30.0  # provider=api 时的 HTTP 超时

    # === Milvus ===
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_prefix: str = "contract_lens"

    # === Redis ===
    redis_url: str = "redis://localhost:6379/0"

    # === 派生属性（不是配置项，是算出来的）===

    @property
    def embedding_model_path(self) -> Path:
        """本地 embedding 模型目录的绝对路径。"""
        return _resolve_model_path(self.embedding_model_name, self.models_dir)

    @property
    def rerank_model_path(self) -> Path:
        """本地 reranker 模型目录的绝对路径。"""
        return _resolve_model_path(self.rerank_model_name, self.models_dir)


@lru_cache
def get_settings() -> Settings:
    """带缓存的配置入口。测试里可以 get_settings.cache_clear() 后重新加载。"""
    return Settings()


settings = get_settings()

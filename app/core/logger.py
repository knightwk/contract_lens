"""统一日志配置。业务代码只 `from app.core.logger import logger`。"""

import sys

from loguru import logger as _logger

from app.config import PROJECT_ROOT, settings

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)

# 移除 loguru 的默认 handler，避免日志重复输出
_logger.remove()

# 控制台：带颜色，给人看
_logger.add(
    sys.stdout,
    level=settings.log_level,
    colorize=True,
    format=_CONSOLE_FORMAT,
)

# 文件：不带颜色，给机器和未来的自己看
_logger.add(
    LOG_DIR / "app.log",
    level=settings.log_level,
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,  # ★ 见下方说明
    backtrace=True,
    diagnose=settings.debug,
    format=_FILE_FORMAT,
)

logger = _logger

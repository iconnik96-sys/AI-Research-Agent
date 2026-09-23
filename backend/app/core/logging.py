import logging
import re
import sys
from typing import Optional

from app.core.config import settings
from app.core.security import get_current_request_id

# Regex patterns to redact sensitive credentials as defense-in-depth
SECRET_PATTERNS = [
    # Database connection passwords: postgresql://user:password@host:port/db
    (re.compile(r"://([^:]+):([^@]+)@"), r"://\1:***@"),
    # Bearer tokens in headers/strings
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{10,}", re.IGNORECASE), r"\1***"),
    # OpenAI and Tavily API keys (sk-..., tvly-...)
    (re.compile(r"\b(sk-[A-Za-z0-9_\-]{15,})\b"), r"sk-***"),
    (re.compile(r"\b(tvly-[A-Za-z0-9_\-]{15,})\b"), r"tvly-***"),
]


class SensitiveDataFilter(logging.Filter):
    """Logging filter that injects the active correlation request_id and masks sensitive credentials."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Inject correlation request_id into record
        req_id = get_current_request_id()
        record.request_id = req_id if req_id else "-"

        # Redact raw secrets in log message string if present
        if isinstance(record.msg, str):
            masked_msg = record.msg
            for pattern, repl in SECRET_PATTERNS:
                masked_msg = pattern.sub(repl, masked_msg)
            record.msg = masked_msg

        return True


def setup_logging(level: Optional[str] = None) -> None:
    """Configure unified structured application logging."""
    log_level_name = level or ("INFO" if settings.ENV.lower() == "production" else "DEBUG")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)

    log_format = "%(asctime)s [%(levelname)s] [request_id=%(request_id)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    sensitive_filter = SensitiveDataFilter()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(sensitive_filter)

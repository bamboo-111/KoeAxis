"""
optimizer/exceptions.py
LLM 调用相关异常（移植自 Voxlign core/exceptions.py 的 LLMError 子树）
"""


class LLMError(Exception):
    """LLM 调用基类异常"""

    def __init__(self, message: str, error_code: str = "LLM_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class LLMAuthError(LLMError):
    """LLM 认证失败（API Key 无效/过期）"""

    def __init__(self, detail: str = "") -> None:
        msg = "LLM 认证失败，请检查 API Key"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, "LLM_AUTH_ERROR")


class LLMRateLimitError(LLMError):
    """LLM 请求频率超限"""

    def __init__(self, detail: str = "") -> None:
        msg = "LLM 请求频率超限，请稍后重试"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, "LLM_RATE_LIMIT_ERROR")


class LLMConnectionError(LLMError):
    """无法连接到 LLM 服务"""

    def __init__(self, detail: str = "") -> None:
        msg = "无法连接到 LLM 服务，请检查网络和 Base URL"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, "LLM_CONNECTION_ERROR")


class LLMResponseError(LLMError):
    """LLM 返回了无效响应（空内容、格式错误等）"""

    def __init__(self, detail: str = "") -> None:
        msg = "LLM 返回了无效响应"
        if detail:
            msg += f": {detail}"
        super().__init__(msg, "LLM_RESPONSE_ERROR")

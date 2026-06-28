from memory_agent.google.retry import is_retryable_google_error
from memory_agent.utils.errors import format_chat_error


def test_is_retryable_google_503():
    exc = Exception(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
        "experiencing high demand.', 'status': 'UNAVAILABLE'}}"
    )
    assert is_retryable_google_error(exc) is True


def test_is_not_retryable_validation_error():
    assert is_retryable_google_error(ValueError("invalid api key")) is False


def test_format_chat_error_capacity():
    raw = "503 UNAVAILABLE high demand"
    message = format_chat_error(raw)
    assert "temporarily at capacity" in message
    assert "gemini-2.0-flash" in message

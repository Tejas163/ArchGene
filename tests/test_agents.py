import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.smolagents_system import (
    is_llm_available,
    call_llm_fallback,
    safe_call_llm,
    call_llm,
)


class TestCircuitBreaker:
    @patch("agents.smolagents_system.requests.get")
    def test_is_llm_available_true(self, mock_get):
        import agents.smolagents_system
        agents.smolagents_system._llm_available = None
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = is_llm_available()
        assert result is True

    @patch("agents.smolagents_system.requests.get")
    def test_is_llm_available_false_connection_error(self, mock_get):
        import agents.smolagents_system
        agents.smolagents_system._llm_available = None
        
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = is_llm_available()
        assert result is False

    @patch("agents.smolagents_system.requests.get")
    def test_is_llm_available_false_timeout(self, mock_get):
        import agents.smolagents_system
        agents.smolagents_system._llm_available = None
        
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = is_llm_available()
        assert result is False


class TestFallback:
    def test_call_llm_fallback_returns_string(self):
        result = call_llm_fallback("test prompt")
        assert isinstance(result, str)
        assert "Fallback mode" in result

    @patch("agents.smolagents_system.is_llm_available")
    def test_safe_call_llm_when_unavailable_uses_fallback(self, mock_available):
        mock_available.return_value = False
        
        result = safe_call_llm("test prompt")
        assert "Fallback mode" in result

    @patch("agents.smolagents_system.is_llm_available")
    @patch("agents.smolagents_system.call_llm")
    def test_safe_call_llm_when_available_calls_llm(self, mock_call_llm, mock_available):
        mock_available.return_value = True
        mock_call_llm.return_value = "LLM response"
        
        result = safe_call_llm("test prompt")
        assert result == "LLM response"
        mock_call_llm.assert_called_once()

    @patch("agents.smolagents_system.is_llm_available")
    @patch("agents.smolagents_system.call_llm")
    def test_safe_call_llm_error_falls_back(self, mock_call_llm, mock_available):
        mock_available.return_value = True
        mock_call_llm.side_effect = RuntimeError("Ollama error")
        
        result = safe_call_llm("test prompt")
        assert "Fallback mode" in result


class TestCallLLM:
    @patch("agents.smolagents_system.requests.post")
    def test_call_llm_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_post.return_value = mock_response
        
        result = call_llm("test prompt")
        assert result == "Test response"

    @patch("agents.smolagents_system.requests.post")
    def test_call_llm_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            call_llm("test prompt")
        assert "not reachable" in str(exc_info.value)

    @patch("agents.smolagents_system.requests.post")
    def test_call_llm_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(RuntimeError) as exc_info:
            call_llm("test prompt")
        assert "timed out" in str(exc_info.value)

    @patch("agents.smolagents_system.requests.post")
    def test_call_llm_http_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.HTTPError("500 Server Error")
        
        with pytest.raises(RuntimeError) as exc_info:
            call_llm("test prompt")
        assert "HTTP error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
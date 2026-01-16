"""Unit tests for BlackboxClient."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from openai import OpenAI
from src.infrastructure.llm.blackbox_client import BlackboxClient
from src.shared.exceptions import LLMQuotaExceededError, LLMProviderError


@pytest.mark.asyncio
async def test_generate_success():
    """Test successful generation."""
    with patch('src.infrastructure.llm.blackbox_client.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create = Mock(return_value=mock_response)
        mock_openai.return_value = mock_client
        
        with patch('src.infrastructure.llm.blackbox_client.get_settings') as mock_settings:
            mock_settings.return_value.blackbox_api_key = "test_key"
            mock_settings.return_value.llm_temperature = 0.7
            mock_settings.return_value.llm_max_tokens = 2048
            
            client = BlackboxClient()
            result = await client.generate("Test prompt")
            
            assert result == "Test response"
            mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_quota_exceeded_fallback():
    """Test model fallback when quota exceeded."""
    with patch('src.infrastructure.llm.blackbox_client.OpenAI') as mock_openai:
        mock_client = Mock()
        
        # First call fails with quota error
        quota_error = Exception("429 Quota exceeded")
        # Second call succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Fallback response"
        
        mock_client.chat.completions.create = Mock(side_effect=[
            quota_error,
            mock_response
        ])
        mock_openai.return_value = mock_client
        
        with patch('src.infrastructure.llm.blackbox_client.get_settings') as mock_settings:
            mock_settings.return_value.blackbox_api_key = "test_key"
            mock_settings.return_value.llm_temperature = 0.7
            mock_settings.return_value.llm_max_tokens = 2048
            
            client = BlackboxClient()
            result = await client.generate("Test prompt")
            
            assert result == "Fallback response"
            # Should have tried twice
            assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_non_quota_error():
    """Test that non-quota errors are raised immediately."""
    with patch('src.infrastructure.llm.blackbox_client.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create = Mock(side_effect=Exception("Connection error"))
        mock_openai.return_value = mock_client
        
        with patch('src.infrastructure.llm.blackbox_client.get_settings') as mock_settings:
            mock_settings.return_value.blackbox_api_key = "test_key"
            mock_settings.return_value.llm_temperature = 0.7
            mock_settings.return_value.llm_max_tokens = 2048
            
            client = BlackboxClient()
            
            with pytest.raises(LLMProviderError):
                await client.generate("Test prompt")


@pytest.mark.asyncio
async def test_generate_all_models_exhausted():
    """Test that LLMQuotaExceededError is raised when all models are exhausted."""
    with patch('src.infrastructure.llm.blackbox_client.OpenAI') as mock_openai:
        mock_client = Mock()
        quota_error = Exception("429 Quota exceeded")
        mock_client.chat.completions.create = Mock(side_effect=quota_error)
        mock_openai.return_value = mock_client
        
        with patch('src.infrastructure.llm.blackbox_client.get_settings') as mock_settings:
            mock_settings.return_value.blackbox_api_key = "test_key"
            mock_settings.return_value.llm_temperature = 0.7
            mock_settings.return_value.llm_max_tokens = 2048
        
        client = BlackboxClient()
        
        with pytest.raises(LLMQuotaExceededError):
            await client.generate("Test prompt")

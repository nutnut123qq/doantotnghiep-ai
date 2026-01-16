"""Unit tests for InsightService."""
import pytest
from unittest.mock import Mock, AsyncMock
from src.application.services.insight_service import InsightService


@pytest.mark.asyncio
async def test_generate_insight(mock_llm_provider, sample_insight_json):
    """Test insight generation with mocked LLM returning JSON."""
    service = InsightService(mock_llm_provider)
    mock_llm_provider.generate = AsyncMock(return_value=sample_insight_json)
    
    result = await service.generate_insight(
        symbol="VIC",
        technical_data={"ma": "Tăng", "rsi": "60"}
    )
    
    assert result["symbol"] == "VIC"
    assert result["type"] == "Buy"
    assert result["title"] == "Strong Buy Signal Detected"
    assert result["confidence"] == 85
    assert isinstance(result["reasoning"], list)
    assert len(result["reasoning"]) > 0
    mock_llm_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_insight_invalid_json(mock_llm_provider):
    """Test insight generation with invalid JSON response."""
    service = InsightService(mock_llm_provider)
    mock_llm_provider.generate = AsyncMock(return_value="Invalid response without JSON")
    
    result = await service.generate_insight(symbol="VIC")
    
    # Should fallback to default values
    assert result["symbol"] == "VIC"
    assert result["type"] == "Hold"
    assert result["confidence"] == 50


@pytest.mark.asyncio
async def test_generate_insight_type_normalization(mock_llm_provider):
    """Test insight type normalization (mua -> Buy, bán -> Sell)."""
    service = InsightService(mock_llm_provider)
    
    # Test Vietnamese type
    mock_llm_provider.generate = AsyncMock(return_value='{"type": "mua", "confidence": 80}')
    result = await service.generate_insight(symbol="VIC")
    assert result["type"] == "Buy"
    
    # Test lowercase
    mock_llm_provider.generate = AsyncMock(return_value='{"type": "buy", "confidence": 75}')
    result = await service.generate_insight(symbol="VIC")
    assert result["type"] == "Buy"


@pytest.mark.asyncio
async def test_insight_confidence_boundaries(mock_llm_provider):
    """Test confidence value is clamped to 0-100."""
    service = InsightService(mock_llm_provider)
    
    # Test high confidence
    mock_llm_provider.generate = AsyncMock(return_value='{"type": "Buy", "confidence": 150}')
    result = await service.generate_insight(symbol="VIC")
    assert result["confidence"] == 100
    
    # Test low confidence
    mock_llm_provider.generate = AsyncMock(return_value='{"type": "Sell", "confidence": -10}')
    result = await service.generate_insight(symbol="VIC")
    assert result["confidence"] == 0

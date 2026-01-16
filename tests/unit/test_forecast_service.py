"""Unit tests for ForecastService."""
import pytest
from unittest.mock import Mock, AsyncMock
from src.application.services.forecast_service import ForecastService


@pytest.mark.asyncio
async def test_generate_forecast(mock_llm_provider, sample_forecast_response):
    """Test forecast generation with mocked LLM."""
    # Setup
    service = ForecastService(mock_llm_provider)
    mock_llm_provider.generate = AsyncMock(return_value=sample_forecast_response)
    
    # Execute
    result = await service.generate_forecast(
        symbol="VIC",
        technical_data={"ma": "Tăng", "rsi": "55"},
        time_horizon="short"
    )
    
    # Assert
    assert result["symbol"] == "VIC"
    assert result["trend"] in ["Up", "Down", "Sideways"]
    assert result["confidence"] in ["High", "Medium", "Low"]
    assert result["time_horizon"] == "short"
    assert result["recommendation"] in ["Buy", "Hold", "Sell"]
    assert "key_drivers" in result
    assert "risks" in result
    assert "analysis" in result
    mock_llm_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_forecast_with_all_data(mock_llm_provider, sample_forecast_response):
    """Test forecast generation with all data types."""
    service = ForecastService(mock_llm_provider)
    mock_llm_provider.generate = AsyncMock(return_value=sample_forecast_response)
    
    result = await service.generate_forecast(
        symbol="VNM",
        technical_data={"ma": "Tăng", "rsi": "60"},
        fundamental_data={"roe": "15.5", "eps": "2500"},
        sentiment_data={"score": "0.7", "sentiment": "positive"},
        time_horizon="medium"
    )
    
    assert result["symbol"] == "VNM"
    assert result["time_horizon"] == "medium"
    mock_llm_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_forecast_service_error_handling(mock_llm_provider):
    """Test forecast service error handling."""
    service = ForecastService(mock_llm_provider)
    mock_llm_provider.generate = AsyncMock(side_effect=Exception("LLM Error"))
    
    with pytest.raises(Exception):
        await service.generate_forecast(symbol="VIC")

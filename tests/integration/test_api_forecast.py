"""Integration tests for forecast API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.integration
def test_generate_forecast_endpoint(client, mock_llm_provider):
    """Test POST /api/forecast/generate endpoint."""
    with patch('src.api.dependencies.get_llm_provider', return_value=mock_llm_provider):
        mock_llm_provider.generate = AsyncMock(return_value="""
        Xu hướng dự báo: Tăng
        Mức độ tin cậy: Cao
        Khuyến nghị: Mua
        """)
        
        response = client.post(
            "/api/forecast/generate",
            json={
                "symbol": "VIC",
                "time_horizon": "short"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "VIC"
        assert data["trend"] in ["Up", "Down", "Sideways"]
        assert data["time_horizon"] == "short"
        assert "generated_at" in data


@pytest.mark.integration
def test_get_forecast_endpoint(client, mock_llm_provider):
    """Test GET /api/forecast/{symbol} endpoint."""
    with patch('src.api.dependencies.get_llm_provider', return_value=mock_llm_provider):
        mock_llm_provider.generate = AsyncMock(return_value="""
        Xu hướng dự báo: Tăng
        Mức độ tin cậy: Cao
        Khuyến nghị: Mua
        """)
        
        response = client.get("/api/forecast/VIC?time_horizon=medium")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "VIC"
        assert data["time_horizon"] == "medium"


@pytest.mark.integration
def test_forecast_endpoint_error_handling(client):
    """Test forecast endpoint error handling."""
    with patch('src.api.dependencies.get_llm_provider', side_effect=Exception("Service error")):
        response = client.post(
            "/api/forecast/generate",
            json={"symbol": "VIC"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

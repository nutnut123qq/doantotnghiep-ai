"""Unit tests for QAService."""
import pytest
from unittest.mock import Mock, AsyncMock
from src.application.services.qa_service import QAService


@pytest.mark.asyncio
async def test_answer_question(mock_llm_provider, mock_vector_store, mock_embedding_provider):
    """Test QA service answering question with RAG."""
    service = QAService(mock_llm_provider, mock_vector_store, mock_embedding_provider)
    
    mock_llm_provider.generate = AsyncMock(return_value="The answer is 1000 VND")
    
    result = await service.answer_question(
        question="What is the EPS?",
        base_context="EPS: 1000 VND"
    )
    
    assert "answer" in result
    assert result["answer"] == "The answer is 1000 VND"
    assert "sources" in result
    assert isinstance(result["sources"], list)

    # Verify vector search was called with query text
    mock_vector_store.search.assert_called_once()
    
    # Verify LLM was called
    mock_llm_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_with_vector_results(
    mock_llm_provider, 
    mock_vector_store, 
    mock_embedding_provider
):
    """Test QA service with retrieved documents from vector store."""
    service = QAService(mock_llm_provider, mock_vector_store, mock_embedding_provider)
    
    mock_vector_store.search = AsyncMock(return_value=[
        {
            "documentId": "doc-1",
            "source": "analysis_report",
            "sourceUrl": None,
            "title": "Báo cáo Q1",
            "section": "Tổng quan",
            "symbol": "ABC",
            "chunkId": "doc-1:0:0",
            "score": 0.9,
            "text": "EPS is 1000 VND in Q1"
        },
        {
            "documentId": "doc-1",
            "source": "analysis_report",
            "sourceUrl": None,
            "title": "Báo cáo Q1",
            "section": "Tài chính",
            "symbol": "ABC",
            "chunkId": "doc-1:0:1",
            "score": 0.8,
            "text": "Revenue increased 20%"
        }
    ])
    mock_llm_provider.generate = AsyncMock(return_value="EPS is 1000 VND")
    
    result = await service.answer_question(
        question="What is the EPS?",
        base_context="Base context"
    )
    
    assert result["answer"] == "EPS is 1000 VND"
    assert len(result["sources"]) == 2
    assert result["sources"][0]["source"] == "analysis_report"


@pytest.mark.asyncio
async def test_analyze_financial_metrics(mock_llm_provider, mock_vector_store, mock_embedding_provider):
    """Test financial metrics analysis."""
    service = QAService(mock_llm_provider, mock_vector_store, mock_embedding_provider)
    
    mock_llm_provider.generate = AsyncMock(return_value="Financial analysis: ROE is 15.5%")
    
    financial_data = {
        "roe": 15.5,
        "roa": 8.2,
        "eps": 2500
    }
    
    result = await service.analyze_financial_metrics(financial_data)
    
    assert "analysis" in result
    assert "metrics" in result
    assert result["metrics"] == financial_data

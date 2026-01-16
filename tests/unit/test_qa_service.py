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
        context="EPS: 1000 VND"
    )
    
    assert "answer" in result
    assert result["answer"] == "The answer is 1000 VND"
    assert "sources" in result
    assert isinstance(result["sources"], list)
    
    # Verify embeddings were generated
    mock_embedding_provider.generate_embedding.assert_called_once_with("What is the EPS?")
    
    # Verify vector search was called
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
        {"text": "EPS is 1000 VND in Q1", "source": "report.pdf", "score": 0.9},
        {"text": "Revenue increased 20%", "source": "report.pdf", "score": 0.8}
    ])
    mock_llm_provider.generate = AsyncMock(return_value="EPS is 1000 VND")
    
    result = await service.answer_question(
        question="What is the EPS?",
        context="Base context"
    )
    
    assert result["answer"] == "EPS is 1000 VND"
    assert len(result["sources"]) == 2
    assert "report.pdf" in result["sources"]


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

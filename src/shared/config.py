"""Centralized configuration management for the AI Service."""
import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    blackbox_api_key: str = Field(..., env="BLACKBOX_API_KEY")
    api_title: str = Field(default="Stock Investment AI Service", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    
    # Vector Store Configuration
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field(
        default="stock_documents", 
        env="QDRANT_COLLECTION_NAME"
    )
    
    # Message Queue Configuration (optional)
    rabbitmq_connection_string: Optional[str] = Field(
        default=None,
        env="RABBITMQ_CONNECTION_STRING"
    )
    
    # LLM Configuration
    default_llm_model: str = Field(
        default="blackboxai/openai/gpt-4-turbo",
        env="DEFAULT_LLM_MODEL"
    )
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    
    # Embedding Configuration
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL_NAME"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="json",
        env="LOG_FORMAT"
    )  # json or text
    
    # Internal API Key for RAG endpoints
    internal_api_key: Optional[str] = Field(
        default=None,
        env="INTERNAL_API_KEY"
    )
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env that are not in this model


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# Changelog

All notable changes to the AI Service will be documented in this file.

## [Unreleased]

### Added
- Standardized exception handling with custom exceptions
- Global exception handlers with request_id tracking
- Architecture boundary checking script
- Comprehensive logging with request_id propagation
- Integration tests for exception handlers
- CI pipeline with architecture checks

### Changed
- **BREAKING**: Error response format standardized
  - All error responses now include: `error`, `message`, `type`, `request_id`
  - Previous format with only `detail` is replaced
- Routes no longer catch generic exceptions - exceptions propagate to global handlers
- Services raise custom exceptions instead of returning empty values
- Coverage threshold increased from 70% to 80%

### Error Handling

#### Error Response Format

All error responses follow this structure:

```json
{
  "error": "Error type description",
  "message": "Detailed error message",
  "type": "ExceptionClassName",
  "request_id": "uuid-string"
}
```

#### Exception to Status Code Mapping

| Exception | HTTP Status | Description |
|-----------|-------------|-------------|
| `ValidationError` | 400 | Input validation failed |
| `NotFoundError` | 404 | Resource not found |
| `LLMProviderError` | 502 | LLM provider error |
| `LLMQuotaExceededError` | 503 | LLM API quota exceeded |
| `VectorStoreError` | 503 | Vector store unavailable |
| `ServiceUnavailableError` | 503 | External service unavailable |
| `EmbeddingServiceError` | 500 | Embedding generation failed |
| `AIServiceException` | 500 | General AI service error |
| `Exception` (catch-all) | 500 | Unexpected error (details not leaked) |

#### Custom Exceptions

- `AIServiceException`: Base exception for all AI service errors
- `LLMProviderError`: LLM provider related errors
- `LLMQuotaExceededError`: LLM API quota exceeded
- `VectorStoreError`: Vector store operation failures
- `EmbeddingServiceError`: Embedding generation failures
- `ValidationError`: Input validation failures
- `ServiceUnavailableError`: External service unavailable
- `NotFoundError`: Resource not found

### Migration Guide

If you have existing clients consuming the API:

1. **Error Response Format**: Update clients to handle the new structured error format with `error`, `message`, `type`, and `request_id` fields.

2. **Status Codes**: Some exceptions now return different status codes:
   - `NotFoundError` returns 404 (previously may have been 400 or 500)
   - `LLMQuotaExceededError` returns 503 (previously may have been 500)

3. **Request ID**: All error responses now include a `request_id` field for tracing. Use this for debugging and support requests.

### Security

- Generic exceptions no longer leak internal error details to clients
- Error messages are sanitized in production
- Request IDs are included for traceability without exposing sensitive information

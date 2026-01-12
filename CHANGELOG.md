# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom exception hierarchy with tracking and severity levels (`core/exceptions.py`)
- Abstract base classes for dependency injection (`core/interfaces.py`)
- Circuit breaker pattern for service resilience (`core/circuit_breaker.py`)
- Environment-specific configuration (development/staging/production) (`core/environment.py`)
- Structured JSON logging with correlation ID tracking (`core/structured_logger.py`)
- Feature flags system for runtime feature toggling
- Comprehensive test fixtures and mock factories (`tests/conftest.py`)

### Changed
- Improved error handling with custom exceptions
- Enhanced logging with structured JSON format and context propagation

## [2.1.0] - 2024-12-XX

### Added
- Embedding cache with LRU eviction (2000 entries, thread-safe)
- Connection pooling for API requests via `requests.Session()`
- API versioning with `/api/v1/` prefix
- Kubernetes health check endpoints (`/health`, `/ready`, `/live`)
- CI/CD pipeline with GitHub Actions
- Docker multi-stage build optimization

### Changed
- Improved Docker configuration for production
- Enhanced health check with caching to reduce API calls

### Fixed
- UI flickering in Streamlit frontend with CSS anti-flicker rules
- UI freezing with callback-based button interactions
- Session list caching to prevent unnecessary re-renders

## [2.0.0] - 2024-12-XX

### Added
- Multi-agent architecture with specialized agents
- RAG (Retrieval-Augmented Generation) system
- Graph RAG for knowledge relationships
- HyDE (Hypothetical Document Embeddings) retrieval
- Multi-query retrieval strategy
- CRAG (Corrective RAG) implementation
- Document processing pipeline
- Session management with persistence
- Notes and export features
- Web search integration
- Guardrails for content safety
- MCP (Model Context Protocol) integration

### Changed
- Complete frontend redesign with Streamlit
- Modular backend architecture

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- Basic chatbot functionality
- Ollama integration
- Simple RAG implementation

---

## Version Numbering

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

## Links

- [GitHub Repository](https://github.com/EmreHacihassan/Personal_Chatbot)
- [Documentation](./README.md)
- [Contributing Guide](./CONTRIBUTING.md)

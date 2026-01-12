# Contributing to Enterprise AI Assistant

First off, thank you for considering contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Commit Message Convention](#commit-message-convention)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes and learn from them

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- Ollama (for local LLM)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Personal_Chatbot.git
cd Personal_Chatbot
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Set Up Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Start Services

```bash
# Using Docker
docker-compose -f docker-compose.dev.yml up -d

# Or manually start Ollama
ollama serve
```

### 6. Run the Application

```bash
python run.py
```

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Use the bug report template** when creating a new issue
3. Include:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Logs and error messages

### Suggesting Features

1. **Check the roadmap** and existing feature requests
2. **Open a discussion** before creating an issue
3. Describe the problem your feature would solve
4. Propose a solution if you have one

### Code Contributions

1. Pick an issue labeled `good first issue` or `help wanted`
2. Comment on the issue to claim it
3. Fork and create a feature branch
4. Write code and tests
5. Submit a pull request

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with the following additions:

```python
# Good: Use type hints
def process_document(content: str, chunk_size: int = 1000) -> List[str]:
    """Process document into chunks."""
    pass

# Good: Use docstrings
class DocumentProcessor:
    """
    Process and chunk documents for RAG.
    
    Attributes:
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
    """
    pass
```

### Code Formatting

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .

# Linting
flake8 .
```

### File Structure

```
feature/
    __init__.py          # Exports public API
    core.py              # Core logic
    models.py            # Pydantic models
    utils.py             # Helper functions
    tests/
        test_core.py
        test_models.py
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `user_input` |
| Functions | snake_case | `process_query()` |
| Classes | PascalCase | `DocumentProcessor` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | _prefix | `_internal_method()` |
| Files | snake_case | `document_processor.py` |

## Testing Guidelines

### Test Structure

```python
# tests/test_feature.py
import pytest
from unittest.mock import Mock, patch

class TestFeatureName:
    """Tests for FeatureName class."""
    
    def test_happy_path(self, mock_llm):
        """Test normal operation."""
        result = feature.process("input")
        assert result.success is True
    
    def test_edge_case(self):
        """Test edge case handling."""
        pass
    
    def test_error_handling(self):
        """Test error scenarios."""
        with pytest.raises(ValueError):
            feature.process(None)
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_feature.py

# Specific test
pytest tests/test_feature.py::TestFeature::test_method

# Verbose output
pytest -v

# Run only fast tests (skip slow)
pytest -m "not slow"
```

### Test Categories

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.asyncio
async def test_async():
    pass
```

### Mock Usage

```python
from tests.conftest import MockLLMManager, MockVectorStore

def test_with_mocks(mock_llm, mock_vector_store):
    """Use fixtures from conftest.py."""
    mock_llm.generate.return_value = "Test response"
    result = service.process("query")
    mock_llm.generate.assert_called_once()
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally (`pytest`)
- [ ] Linting passes (`flake8`, `mypy`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests added/modified

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
```

### Review Process

1. Automated checks run (CI/CD)
2. Code review by maintainer
3. Address feedback and update
4. Approval and merge

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change without feature/fix |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |

### Examples

```bash
feat(rag): add multi-query retrieval strategy

fix(api): handle timeout in health check endpoint

docs(readme): update installation instructions

refactor(agents): extract base class for all agents

test(core): add unit tests for circuit breaker

chore(deps): update pydantic to 2.5.0
```

### Scope Options

- `api` - API endpoints
- `frontend` - Streamlit UI
- `rag` - RAG system
- `agents` - Agent system
- `core` - Core utilities
- `docs` - Documentation
- `tests` - Test infrastructure
- `ci` - CI/CD pipeline
- `docker` - Container config

## Questions?

- Open a [Discussion](https://github.com/EmreHacihassan/Personal_Chatbot/discussions)
- Check [Documentation](./README.md)
- Review [Changelog](./CHANGELOG.md)

Thank you for contributing! 🙏

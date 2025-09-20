# Contributing to TTT-MAE

Thank you for your interest in contributing to TTT-MAE! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Git
- Basic understanding of deep learning and computer vision

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/TTT-MAE.git
   cd TTT-MAE
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv ttt-mae-dev
   source ttt-mae-dev/bin/activate  # On Windows: ttt-mae-dev\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   ```

3. **Install development tools**
   ```bash
   # Install pre-commit hooks
   pip install pre-commit
   pre-commit install
   
   # Install testing tools
   pip install pytest pytest-cov
   ```

## 📋 How to Contribute

### 1. Types of Contributions

We welcome various types of contributions:

- **🐛 Bug Reports**: Report bugs or issues
- **✨ Feature Requests**: Suggest new features
- **📖 Documentation**: Improve documentation
- **🔧 Code Contributions**: Fix bugs or implement features
- **🧪 Tests**: Add or improve tests
- **📊 Examples**: Add usage examples or tutorials

### 2. Before You Start

- Check existing [issues](https://github.com/yourusername/TTT-MAE/issues) and [pull requests](https://github.com/yourusername/TTT-MAE/pulls)
- Open an issue to discuss major changes before implementing
- Ensure your contribution aligns with the project goals

### 3. Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run verification tests
   python test_verification.py
   
   # Run specific tests
   pytest tests/
   
   # Check code coverage
   pytest --cov=. tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "type: brief description of changes"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

## 📝 Coding Standards

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Type Hints

- Add type hints to all function parameters and return values
- Use `typing` module for complex types
- Example:
  ```python
  def train_model(data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
      """Train the model with given data."""
      # Implementation
      return {"accuracy": 0.95, "loss": 0.1}
  ```

### Documentation

- Add comprehensive docstrings to all functions and classes
- Use Google-style docstrings
- Example:
  ```python
  def process_data(data: np.ndarray, normalize: bool = True) -> np.ndarray:
      """
      Process input data for training.
      
      Args:
          data: Input data array of shape (N, H, W, C)
          normalize: Whether to normalize the data
          
      Returns:
          Processed data array
          
      Raises:
          ValueError: If data shape is invalid
      """
      # Implementation
  ```

### Testing

- Write tests for new functionality
- Use `pytest` for testing
- Aim for >80% code coverage
- Test both success and failure cases

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=. tests/

# Run in verbose mode
pytest -v
```

### Writing Tests

```python
import pytest
import torch
from your_module import your_function

def test_your_function():
    """Test your_function with valid inputs."""
    # Arrange
    input_data = torch.randn(10, 3, 224, 224)
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result.shape == (10, 1000)
    assert torch.all(result >= 0)

def test_your_function_invalid_input():
    """Test your_function with invalid inputs."""
    with pytest.raises(ValueError):
        your_function(None)
```

## 📖 Documentation Guidelines

### Code Documentation

- Add docstrings to all public functions and classes
- Include examples in docstrings when helpful
- Keep documentation up-to-date with code changes

### README Updates

- Update README.md for significant features
- Add usage examples for new functionality
- Update installation instructions if needed

## 🔄 Pull Request Process

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

### Review Process

1. **Automated Checks**: CI/CD will run tests and linting
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

## 🐛 Bug Reports

### Before Reporting

- Check if the issue already exists
- Try to reproduce the bug
- Test with the latest version

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 2.0.1]
- TTT-MAE version: [e.g., 1.0.0]

**Additional Context**
Any other context about the problem
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Any alternative solutions you've considered

**Additional Context**
Any other context or screenshots
```

## 🏷️ Commit Message Guidelines

Use the following format:

```
type: brief description

Longer description if needed

Fixes #issue_number
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
feat: add support for custom loss functions

fix: resolve memory leak in data loading

docs: update installation instructions

test: add integration tests for TTT training
```

## 📞 Getting Help

- **Documentation**: Check the [documentation](docs/)
- **Issues**: Browse [existing issues](https://github.com/yourusername/TTT-MAE/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/yourusername/TTT-MAE/discussions)

## 🎯 Development Priorities

Current development priorities:

1. **Performance Optimization**: Improve training speed and memory usage
2. **Model Zoo**: Add more pre-trained models
3. **Documentation**: Expand tutorials and examples
4. **Testing**: Increase test coverage
5. **GPU Support**: Improve multi-GPU training

## 📄 License

By contributing to TTT-MAE, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to TTT-MAE! 🚀

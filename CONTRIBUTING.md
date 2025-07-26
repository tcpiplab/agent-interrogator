# Contributing to Agent Interrogator

Thank you for your interest in contributing to Agent Interrogator! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check the [existing issues](https://github.com/qwordsmith/agent-interrogator/issues) to avoid duplicates
2. Use the issue templates when available
3. Provide clear descriptions and steps to reproduce bugs
4. Include relevant error messages and system information

### Suggesting Features

We welcome feature suggestions! Please:
1. Check if the feature has already been requested
2. Explain the use case and benefits
3. Consider how it fits with the project's security focus

### Contributing Code

#### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/agent-interrogator.git
   cd agent-interrogator
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

#### Development Process

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Write or update tests for your changes

4. Run the test suite:
   ```bash
   pytest tests/
   ```

5. Run code formatters:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

6. Run type checking:
   ```bash
   mypy src/
   ```

7. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: brief description"
   ```

8. Push to your fork and create a pull request

#### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write descriptive docstrings for all public functions and classes
- Keep functions focused and modular
- Add appropriate error handling

#### Testing

- Write tests for all new functionality
- Ensure existing tests pass
- Aim for high test coverage
- Use `pytest` for unit tests
- Mock external dependencies appropriately

#### Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Include examples in docstrings where helpful
- Update configuration documentation if adding new options

### Pull Request Process

1. Ensure all tests pass locally
2. Update documentation as needed
3. Add an entry to CHANGELOG.md
4. Fill out the pull request template
5. Link related issues in the PR description
6. Be responsive to review feedback

### Security Considerations

- Never commit credentials or sensitive data
- Be mindful of security implications in your code
- Document any security considerations

## Development Setup Tips

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_output_modes.py

# Run tests with coverage
pytest --cov=agent_interrogator tests/

# Run tests in verbose mode
pytest -v tests/
```

### Using Pre-commit Hooks

We recommend setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion in the GitHub repository
- Ask in an issue before starting major work
- Reach out to the maintainers

Thank you for helping improve Agent Interrogator!
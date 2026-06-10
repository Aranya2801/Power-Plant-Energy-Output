# Contributing to Power-Plant-Energy-Output

Thank you for your interest in contributing! 🎉

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/Power-Plant-Energy-Output.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Install dev dependencies**: `pip install -r requirements-dev.txt`
5. **Install pre-commit hooks**: `pre-commit install`

## Development Workflow

```bash
# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/ --max-line-length=100
```

## Pull Request Guidelines

- Write clear, descriptive commit messages using [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat: add LSTM time-series model`
  - `fix: correct SHAP waterfall indexing`
  - `docs: update API reference`
- Ensure all tests pass before submitting
- Add tests for any new functionality (aim for 90%+ coverage)
- Update `CHANGELOG.md` with your changes
- Reference any related issues: `Closes #42`

## Code Style

- **Python**: [Black](https://black.readthedocs.io/) formatter, max line length 100
- **Imports**: sorted with [isort](https://pycqa.github.io/isort/)
- **Docstrings**: NumPy-style docstrings
- **Type hints**: use `from __future__ import annotations` and annotate all public functions

## Reporting Bugs

Open an issue with:
- Python version and OS
- Steps to reproduce
- Expected vs actual behaviour
- Error traceback

## Feature Requests

Open an issue labelled `enhancement` with:
- Clear description of the feature
- Motivation / use case
- Any relevant references or papers

## Code of Conduct

Be respectful, inclusive, and constructive. All contributors are expected to adhere to the [Contributor Covenant](https://www.contributor-covenant.org/).

---

*Questions? Open a [Discussion](https://github.com/Aranya2801/Power-Plant-Energy-Output/discussions) on GitHub.*

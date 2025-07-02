# Development Setup

This document describes the development environment setup and tools for the SEM/GDS Alignment Tool.

## Development Tools

### Code Formatting and Linting
- **Black**: Code formatter with 88-character line length
- **isort**: Import statement organizer
- **flake8**: Linting and style checking
- **mypy**: Static type checking

### Editor Configuration
- **.editorconfig**: Consistent formatting across editors
- **pyproject.toml**: Tool configurations
- **pre-commit hooks**: Automated code quality checks

## Setup Instructions

### 1. Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks
```bash
pre-commit install
```

### 3. Run Code Quality Tools

#### Format code with Black
```bash
black src/
```

#### Sort imports with isort
```bash
isort src/
```

#### Lint with flake8
```bash
flake8 src/
```

#### Type check with mypy
```bash
mypy src/
```

#### Run all pre-commit hooks
```bash
pre-commit run --all-files
```

## Code Standards

### Python Style
- Follow PEP 8 with Black formatting
- 88-character line length
- Type hints for all functions and methods
- Docstrings for all public modules, classes, and functions

### Import Organization
- Standard library imports first
- Third-party imports second
- Local application imports last
- Separate groups with blank lines

### File Structure
- Use relative imports within packages
- Keep modules focused and cohesive
- Follow the established directory structure

## Editor Setup

### VS Code
Recommended extensions:
- Python
- Black Formatter
- isort
- Pylance
- mypy

### PyCharm
Configure:
- Black as external tool
- isort for import optimization
- Enable mypy inspection
- Set line length to 88 characters

## Pre-commit Hooks

The following checks run automatically on commit:
- Trailing whitespace removal
- End-of-file newline fixing
- YAML and JSON validation
- Merge conflict detection
- Large file detection
- Black code formatting
- isort import sorting
- flake8 linting
- mypy type checking

## Development Workflow

1. Make changes to code
2. Run formatting tools if needed
3. Commit changes (hooks run automatically)
4. Fix any issues reported by hooks
5. Push to repository

## Configuration Files

- **.editorconfig**: Editor formatting rules
- **pyproject.toml**: Tool configurations and project metadata
- **.pre-commit-config.yaml**: Pre-commit hook definitions

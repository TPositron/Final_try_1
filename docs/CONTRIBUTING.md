# Contributing Guidelines

Thank you for contributing to the SEM/GDS Alignment Tool. Please follow these guidelines to ensure consistency and quality.

## Code Style

### Formatting
- Use Black for code formatting (88-character line length)
- Use isort for import sorting
- Follow PEP 8 conventions

### Type Hints
- Add type hints to all function signatures
- Use appropriate types from `typing` module when needed
- Document complex types in docstrings

### Documentation
- Write clear, concise docstrings for all public functions and classes
- Use Google-style docstrings
- Include parameter types and descriptions
- Document return values and exceptions

## Code Organization

### Module Structure
- Keep modules focused on a single responsibility
- Use clear, descriptive names for functions and classes
- Group related functionality together

### Import Guidelines
- Use absolute imports for external packages
- Use relative imports within the same package
- Follow isort configuration for import ordering

## Commit Guidelines

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense
- Keep first line under 50 characters
- Add detailed description if needed

### Pre-commit Checks
- All commits must pass pre-commit hooks
- Fix any formatting or linting issues before committing
- Ensure type checking passes

## Development Process

### Making Changes
1. Create a feature branch from main
2. Make changes following code style guidelines
3. Run local quality checks
4. Commit with descriptive messages
5. Submit pull request

### Quality Checks
Before committing, run:
```bash
black src/
isort src/
flake8 src/
mypy src/
```

## File Organization

### Directory Structure
Follow the established project structure:
- `src/core/`: Core functionality
- `src/services/`: Business logic services
- `src/ui/`: User interface components
- `config/`: Configuration files
- `docs/`: Documentation

### Naming Conventions
- Use snake_case for files and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants
- Use descriptive names that explain purpose

## Documentation

### Code Documentation
- Document all public APIs
- Include usage examples in docstrings
- Explain complex algorithms or business logic
- Keep documentation up to date with code changes

### Comments
- Use comments sparingly for complex logic
- Explain why, not what
- Remove commented-out code before committing

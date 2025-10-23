# Contributing to AutoSim Pro

Thank you for your interest in contributing to AutoSim Pro! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/Sherin-SEF-AI/car-simulation/issues) page
- Search existing issues before creating a new one
- Provide detailed information including:
  - Operating system and version
  - Python version
  - Steps to reproduce
  - Expected vs actual behavior
  - Screenshots if applicable

### Suggesting Features
- Open a [GitHub Discussion](https://github.com/Sherin-SEF-AI/car-simulation/discussions)
- Describe the feature and its benefits
- Provide use cases and examples
- Consider implementation complexity

### Code Contributions

#### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/car-simulation.git
cd car-simulation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Making Changes
1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Follow the coding standards below
3. **Add tests**: Ensure your changes are tested
4. **Run tests**: `python -m pytest tests/`
5. **Commit changes**: Use descriptive commit messages
6. **Push branch**: `git push origin feature/your-feature-name`
7. **Create Pull Request**: Submit your changes for review

## 📝 Coding Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where possible

### Code Quality
- Write clear, self-documenting code
- Add docstrings to all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Testing
- Write unit tests for new functionality
- Maintain test coverage above 80%
- Use pytest for testing framework
- Mock external dependencies

### Documentation
- Update README.md if needed
- Add docstrings to new functions/classes
- Update API documentation
- Include examples in docstrings

## 🏗️ Project Structure

```
AutoSim Pro/
├── src/                    # Source code
│   ├── core/              # Core simulation engine
│   ├── ui/                # User interface components
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── docs/                  # Documentation
├── examples/              # Example scripts
├── requirements*.txt      # Dependencies
└── setup.py              # Package setup
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_physics.py

# Run with verbose output
python -m pytest -v
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Test both success and failure cases
- Mock external dependencies

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] Branch is up to date with main

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Screenshots (if applicable)
Add screenshots for UI changes

## Additional Notes
Any additional information
```

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization** - Improve simulation speed
- **AI Algorithms** - New behavioral models
- **Testing** - Increase test coverage
- **Documentation** - Tutorials and guides

### Medium Priority
- **UI/UX Improvements** - Better user experience
- **New Features** - Additional simulation capabilities
- **Bug Fixes** - Resolve existing issues
- **Code Refactoring** - Improve code quality

### Low Priority
- **Translations** - Multi-language support
- **Themes** - Additional UI themes
- **Examples** - More example scenarios
- **Integrations** - Third-party tool support

## 🏷️ Commit Message Guidelines

### Format
```
type(scope): description

[optional body]

[optional footer]
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes
- **refactor**: Code refactoring
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

### Examples
```
feat(physics): add tire temperature simulation
fix(ui): resolve vehicle list refresh issue
docs(readme): update installation instructions
test(ai): add behavioral model unit tests
```

## 🔄 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git

## 📞 Getting Help

### Community Support
- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Email**: [connect@sherinjosephroy.link](mailto:connect@sherinjosephroy.link)

### Development Questions
- Check existing documentation
- Search GitHub issues and discussions
- Ask in the community channels
- Contact maintainers directly

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation
- Social media announcements

## 📄 License

By contributing to AutoSim Pro, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AutoSim Pro! Your efforts help make autonomous vehicle simulation accessible to everyone.
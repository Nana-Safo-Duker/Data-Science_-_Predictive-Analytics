# Contributing to Unicorn Companies Analysis Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce (if it's a bug)
   - Expected vs actual behavior
   - Screenshots (if applicable)

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the coding style
   - Add comments where necessary
   - Update documentation if needed
4. **Test your changes**:
   - Run existing tests
   - Test your new code
   - Ensure notebooks run without errors
5. **Commit your changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

## Code Style

### Python

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small
- Add comments for complex logic

### R

- Follow tidyverse style guide
- Use meaningful variable names
- Add comments for complex logic
- Keep code organized and readable

### Notebooks

- Use markdown cells for explanations
- Keep code cells focused
- Add clear section headers
- Include output cells with results

## Project Structure

Please maintain the existing project structure:

```
Unicorn_Companies/
├── data/          # Data files
├── notebooks/     # Jupyter/R Markdown notebooks
├── scripts/       # Reusable scripts
├── results/       # Generated results
└── docs/          # Documentation
```

## Documentation

- Update README.md if you add new features
- Add docstrings to new functions
- Update analysis guides if you change workflows
- Document any new dependencies

## Testing

- Test your code before submitting
- Ensure notebooks run from top to bottom
- Check that outputs are correct
- Verify visualizations render properly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions, please open an issue or contact the maintainers.

Thank you for contributing!



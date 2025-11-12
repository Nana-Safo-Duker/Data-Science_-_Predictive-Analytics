# Contributing to Position Salaries Project

Thank you for your interest in contributing to the Position Salaries project! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Position_Salaries.git
   cd Position_Salaries
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Python Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### R Environment

1. Install required R packages:
   ```r
   install.packages(c("tidyverse", "ggplot2", "dplyr", "readr", "corrplot", 
                      "moments", "randomForest", "e1071", "caret"))
   ```

## Code Style

### Python

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

### R

- Follow the tidyverse style guide
- Use meaningful variable names
- Add comments for complex logic
- Keep code modular and reusable

## Project Structure

Please maintain the existing project structure:

```
Position_Salaries/
├── data/
│   ├── raw/           # Raw datasets (do not modify)
│   └── processed/     # Processed datasets
├── notebooks/         # Jupyter notebooks and R Markdown files
├── scripts/           # Python and R scripts
├── results/           # Generated figures and models
└── ...
```

## Making Changes

1. **Make your changes** in your feature branch
2. **Test your changes** to ensure they work correctly
3. **Update documentation** if needed
4. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

## Commit Messages

Please write clear commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally

## Submitting Changes

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Provide a clear description of your changes
   - Reference any related issues
   - Ensure all tests pass (if applicable)

## Types of Contributions

We welcome various types of contributions:

### Bug Reports

- Use the GitHub issue tracker
- Include a clear description of the bug
- Provide steps to reproduce the issue
- Include expected vs actual behavior

### Feature Requests

- Use the GitHub issue tracker
- Clearly describe the feature
- Explain the use case and benefits
- Consider implementation approaches

### Code Contributions

- Fix bugs
- Add new features
- Improve documentation
- Optimize performance
- Add tests

### Documentation

- Improve README
- Add code comments
- Create tutorials
- Fix typos and errors

## Code Review Process

1. All submissions require review
2. Reviews will focus on:
   - Code quality and style
   - Functionality and correctness
   - Documentation and comments
   - Test coverage (if applicable)

3. Address review feedback promptly
4. Be open to suggestions and improvements

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Contact the project maintainers
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

Thank you for contributing to the Position Salaries project!



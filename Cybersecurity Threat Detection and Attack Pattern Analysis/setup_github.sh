#!/bin/bash
# Setup script for GitHub repository

echo "Setting up GitHub repository for Cybersecurity Attacks Analysis project..."

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p visualizations
mkdir -p results
mkdir -p docs

# Add all files to git
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Cybersecurity Attacks Analysis project

- Comprehensive EDA notebooks (Python and R)
- Statistical analysis notebooks
- Univariate, bivariate, multivariate analysis
- Machine learning analysis with multiple algorithms
- Comprehensive README and documentation
- Project structure and scripts"

echo "GitHub repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add the remote: git remote add origin <repository-url>"
echo "3. Push the code: git push -u origin main"




#!/bin/bash
# Setup script for Employee Dataset Analysis Project
# This script helps initialize the project structure

echo "========================================="
echo "Employee Dataset Analysis - Setup Script"
echo "========================================="

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks/python
mkdir -p notebooks/r
mkdir -p scripts/python
mkdir -p scripts/r
mkdir -p results/models
mkdir -p results/plots
mkdir -p results/tables
mkdir -p docs

# Create .gitkeep files to maintain directory structure in git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch results/models/.gitkeep
touch results/plots/.gitkeep
touch results/tables/.gitkeep

echo "Directories created successfully!"

# Check if Python is installed
if command -v python3 &> /dev/null; then
    echo "Python 3 is installed: $(python3 --version)"
else
    echo "Warning: Python 3 is not installed"
fi

# Check if R is installed
if command -v R &> /dev/null; then
    echo "R is installed: $(R --version | head -n 1)"
else
    echo "Warning: R is not installed"
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed: $(conda --version)"
    echo "You can create the conda environment with: conda env create -f environment.yml"
else
    echo "Warning: Conda is not installed"
fi

echo ""
echo "========================================="
echo "Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Place your dataset in data/raw/employees.csv"
echo "2. Install Python dependencies: pip install -r requirements.txt"
echo "3. Install R packages: Rscript scripts/r/install_packages.R"
echo "4. Run the notebooks in order:"
echo "   - notebooks/python/01_EDA.ipynb"
echo "   - notebooks/python/02_Statistical_Analysis.ipynb"
echo "   - notebooks/python/03_Univariate_Bivariate_Multivariate.ipynb"
echo "   - notebooks/python/04_ML_Analysis.ipynb"
echo ""
echo "Or run the R notebooks:"
echo "   - notebooks/r/01_EDA.ipynb"
echo "   - notebooks/r/02_Statistical_Analysis.ipynb"
echo "   - notebooks/r/03_Univariate_Bivariate_Multivariate.ipynb"
echo "   - notebooks/r/04_ML_Analysis.ipynb"
echo ""




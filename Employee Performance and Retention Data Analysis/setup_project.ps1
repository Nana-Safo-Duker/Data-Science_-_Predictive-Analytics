# Setup script for Employee Dataset Analysis Project (PowerShell)
# This script helps initialize the project structure on Windows

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Employee Dataset Analysis - Setup Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Create necessary directories
Write-Host "Creating project directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path "notebooks\python" | Out-Null
New-Item -ItemType Directory -Force -Path "notebooks\r" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts\python" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts\r" | Out-Null
New-Item -ItemType Directory -Force -Path "results\models" | Out-Null
New-Item -ItemType Directory -Force -Path "results\plots" | Out-Null
New-Item -ItemType Directory -Force -Path "results\tables" | Out-Null
New-Item -ItemType Directory -Force -Path "docs" | Out-Null

# Create .gitkeep files to maintain directory structure in git
New-Item -ItemType File -Force -Path "data\raw\.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "data\processed\.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "results\models\.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "results\plots\.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "results\tables\.gitkeep" | Out-Null

Write-Host "Directories created successfully!" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python is installed: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Warning: Python is not installed" -ForegroundColor Yellow
}

# Check if R is installed
try {
    $rVersion = R --version 2>&1 | Select-Object -First 1
    Write-Host "R is installed: $rVersion" -ForegroundColor Green
} catch {
    Write-Host "Warning: R is not installed" -ForegroundColor Yellow
}

# Check if conda is installed
try {
    $condaVersion = conda --version 2>&1
    Write-Host "Conda is installed: $condaVersion" -ForegroundColor Green
    Write-Host "You can create the conda environment with: conda env create -f environment.yml" -ForegroundColor Cyan
} catch {
    Write-Host "Warning: Conda is not installed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setup completed!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Place your dataset in data\raw\employees.csv"
Write-Host "2. Install Python dependencies: pip install -r requirements.txt"
Write-Host "3. Install R packages: Rscript scripts\r\install_packages.R"
Write-Host "4. Run the notebooks in order:"
Write-Host "   - notebooks\python\01_EDA.ipynb"
Write-Host "   - notebooks\python\02_Statistical_Analysis.ipynb"
Write-Host "   - notebooks\python\03_Univariate_Bivariate_Multivariate.ipynb"
Write-Host "   - notebooks\python\04_ML_Analysis.ipynb"
Write-Host ""
Write-Host "Or run the R notebooks:"
Write-Host "   - notebooks\r\01_EDA.ipynb"
Write-Host "   - notebooks\r\02_Statistical_Analysis.ipynb"
Write-Host "   - notebooks\r\03_Univariate_Bivariate_Multivariate.ipynb"
Write-Host "   - notebooks\r\04_ML_Analysis.ipynb"
Write-Host ""




# Quick Start Guide

This guide will help you get started with the Employee Dataset Analysis project quickly.

## Prerequisites

- Python 3.8+ or R 4.0+
- Jupyter Notebook (for notebook execution)
- Git (for version control)

## Quick Setup

### Python Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd emplyees
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda env create -f environment.yml
   conda activate employees-analysis
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis scripts:**
   ```bash
   # EDA
   python scripts/python/eda.py
   
   # Statistical Analysis
   python scripts/python/statistical_analysis.py
   
   # Univariate/Bivariate/Multivariate Analysis
   python scripts/python/univariate_bivariate_multivariate.py
   
   # ML Analysis
   python scripts/python/ml_analysis.py
   ```

5. **Or use Jupyter Notebooks:**
   ```bash
   jupyter notebook
   # Then open notebooks/python/01_EDA.ipynb, etc.
   ```

### R Setup

1. **Install R packages:**
   ```bash
   Rscript scripts/r/install_packages.R
   ```

2. **Run the analysis scripts:**
   ```bash
   # From project root directory
   Rscript scripts/r/eda.R
   Rscript scripts/r/statistical_analysis.R
   Rscript scripts/r/univariate_bivariate_multivariate.R
   Rscript scripts/r/ml_analysis.R
   ```

3. **Or use RStudio/Jupyter:**
   - Open RStudio and set working directory to project root
   - Open and run the R notebooks in `notebooks/r/`

## Expected Output

After running the scripts, you should see:

- **Cleaned dataset**: `data/processed/employees_cleaned.csv`
- **Visualizations**: `results/plots/*.png`
- **Statistical tables**: `results/tables/*.csv`
- **Trained models**: `results/models/*.pkl` (Python) or `*.rds` (R)

## Running Analysis in Order

It's recommended to run the analysis scripts in the following order:

1. **EDA** (`01_EDA` or `eda.py`/`eda.R`)
   - Creates cleaned dataset
   - Generates initial visualizations

2. **Statistical Analysis** (`02_Statistical_Analysis` or `statistical_analysis.py`/`statistical_analysis.R`)
   - Performs hypothesis tests
   - Generates statistical summaries

3. **Univariate/Bivariate/Multivariate Analysis** (`03_Univariate_Bivariate_Multivariate` or `univariate_bivariate_multivariate.py`/`univariate_bivariate_multivariate.R`)
   - Analyzes variable relationships
   - Creates detailed visualizations

4. **ML Analysis** (`04_ML_Analysis` or `ml_analysis.py`/`ml_analysis.R`)
   - Trains machine learning models
   - Generates predictions and model comparisons

## Troubleshooting

### Python Issues

- **Import errors**: Make sure all packages are installed: `pip install -r requirements.txt`
- **File not found**: Ensure you're running scripts from the project root directory
- **Memory errors**: Try reducing dataset size or using smaller models

### R Issues

- **Package installation errors**: Run `Rscript scripts/r/install_packages.R`
- **Working directory errors**: Make sure you're in the project root directory
- **Missing packages**: Install missing packages manually: `install.packages("package_name")`

## Next Steps

1. Review the generated plots in `results/plots/`
2. Check statistical tables in `results/tables/`
3. Examine trained models in `results/models/`
4. Read the comprehensive README.md for detailed documentation

## Getting Help

- Check the README.md for detailed documentation
- Review the script comments for implementation details
- Check the notebooks for interactive analysis

## License

Please respect the original dataset's license. See LICENSE file for more information.







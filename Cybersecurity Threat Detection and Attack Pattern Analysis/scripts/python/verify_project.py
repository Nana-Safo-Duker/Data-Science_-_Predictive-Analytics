"""
Script to verify project structure and files
"""
import os
import json

def check_file_exists(filepath):
    """Check if file exists"""
    if os.path.exists(filepath):
        return True
    else:
        print(f"Missing: {filepath}")
        return False

def check_notebook_valid(filepath):
    """Check if notebook is valid JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except:
        print(f"Invalid notebook: {filepath}")
        return False

# Required files and directories
required_files = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "PROJECT_SUMMARY.md",
    "data/Cybersecurity_attacks.csv",
    "scripts/python/eda.py",
    "scripts/python/ml_analysis.py",
    "scripts/python/univariate_bivariate_multivariate_analysis.py",
    "scripts/r/install.R",
    "scripts/r/eda.R",
    "scripts/r/statistical_analysis.R",
    "scripts/r/ml_analysis.R",
]

required_notebooks = [
    "notebooks/python/01_EDA_Cybersecurity_Attacks.ipynb",
    "notebooks/python/02_Statistical_Analysis.ipynb",
    "notebooks/python/03_Univariate_Bivariate_Multivariate_Analysis.ipynb",
    "notebooks/python/04_ML_Analysis.ipynb",
    "notebooks/r/01_EDA_Cybersecurity_Attacks.ipynb",
    "notebooks/r/02_Statistical_Analysis.ipynb",
    "notebooks/r/03_Univariate_Bivariate_Multivariate_Analysis.ipynb",
    "notebooks/r/04_ML_Analysis.ipynb",
]

required_directories = [
    "data",
    "notebooks",
    "notebooks/python",
    "notebooks/r",
    "scripts",
    "scripts/python",
    "scripts/r",
    "visualizations",
    "results",
    "docs",
]

print("=" * 60)
print("PROJECT VERIFICATION")
print("=" * 60)

# Check directories
print("\n1. Checking directories...")
all_dirs_ok = True
for dir_path in required_directories:
    if os.path.exists(dir_path):
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ✗ Missing: {dir_path}")
        all_dirs_ok = False

# Check files
print("\n2. Checking files...")
all_files_ok = True
for file_path in required_files:
    if check_file_exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        all_files_ok = False

# Check notebooks
print("\n3. Checking notebooks...")
all_notebooks_ok = True
for notebook_path in required_notebooks:
    if check_file_exists(notebook_path):
        if check_notebook_valid(notebook_path):
            print(f"  ✓ {notebook_path}")
        else:
            all_notebooks_ok = False
    else:
        all_notebooks_ok = False

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

if all_dirs_ok and all_files_ok and all_notebooks_ok:
    print("✓ All checks passed! Project structure is complete.")
else:
    print("✗ Some checks failed. Please review the output above.")

print("\nProject Structure:")
print("  - Directories: OK" if all_dirs_ok else "  - Directories: FAILED")
print("  - Files: OK" if all_files_ok else "  - Files: FAILED")
print("  - Notebooks: OK" if all_notebooks_ok else "  - Notebooks: FAILED")

print("\n" + "=" * 60)




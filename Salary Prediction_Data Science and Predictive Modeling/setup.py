"""
Setup script for Position Salaries Project
"""

from pathlib import Path
import os

def create_directories():
    """Create necessary directories if they don't exist."""
    project_root = Path(__file__).resolve().parent
    
    directories = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "results" / "figures",
        project_root / "results" / "models",
        project_root / "notebooks" / "python",
        project_root / "notebooks" / "r",
        project_root / "scripts" / "python",
        project_root / "scripts" / "r",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep file
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("Directories created successfully!")

def verify_dataset():
    """Verify that the dataset exists."""
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "data" / "raw" / "Position_Salaries.csv"
    
    if dataset_path.exists():
        print(f"[OK] Dataset found at: {dataset_path}")
        return True
    else:
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("Please ensure the dataset is placed in data/raw/Position_Salaries.csv")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("Position Salaries Project Setup")
    print("=" * 60)
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Verifying dataset...")
    dataset_exists = verify_dataset()
    
    print("\n" + "=" * 60)
    if dataset_exists:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Install Python dependencies: pip install -r requirements.txt")
        print("2. Install R packages (if using R)")
        print("3. Run the analysis notebooks or scripts")
    else:
        print("Setup completed with warnings!")
        print("Please ensure the dataset is in the correct location.")
    print("=" * 60)

if __name__ == "__main__":
    main()


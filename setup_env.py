#!/usr/bin/env python3
"""
Cross-platform virtual environment setup script.

This script creates and configures a virtual environment for the SEM/GDS
alignment tool with all required dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True, shell=None):
    """Run a command and return the result."""
    if shell is None:
        shell = platform.system() == "Windows"
    
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        sys.exit(1)
    
    return result


def create_venv(venv_path):
    """Create virtual environment."""
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return
    
    print(f"Creating virtual environment at {venv_path}")
    run_command([sys.executable, "-m", "venv", str(venv_path)])


def get_activation_script(venv_path):
    """Get the appropriate activation script for the platform."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "activate.bat"
    else:
        return venv_path / "bin" / "activate"


def get_python_executable(venv_path):
    """Get the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_requirements(venv_path, requirements_file):
    """Install requirements in the virtual environment."""
    python_exe = get_python_executable(venv_path)
    
    if not requirements_file.exists():
        print(f"Requirements file not found: {requirements_file}")
        return
    
    print("Installing requirements...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)])


def create_activation_scripts(venv_path, project_root):
    """Create platform-specific activation scripts."""
    
    # Windows batch file
    windows_script = project_root / "activate_env.bat"
    with open(windows_script, 'w') as f:
        f.write(f"""@echo off
REM Activate virtual environment for SEM/GDS alignment tool

if exist "{venv_path}\\Scripts\\activate.bat" (
    call "{venv_path}\\Scripts\\activate.bat"
    echo Virtual environment activated: {venv_path}
) else (
    echo Error: Virtual environment not found at {venv_path}
    echo Run setup_env.py first to create the environment
    pause
    exit /b 1
)

REM Optional: Change to project directory
cd /d "{project_root}"

echo.
echo Environment ready! You can now:
echo   - Run the application: python start_gui.py
echo   - Install dev dependencies: pip install -e ".[dev]"
echo   - Run tests: pytest
echo.
""")

    # Unix shell script
    unix_script = project_root / "activate_env.sh"
    with open(unix_script, 'w') as f:
        f.write(f"""#!/bin/bash
# Activate virtual environment for SEM/GDS alignment tool

VENV_PATH="{venv_path}"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated: $VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Run setup_env.py first to create the environment"
    exit 1
fi

# Optional: Change to project directory
cd "{project_root}"

echo
echo "Environment ready! You can now:"
echo "  - Run the application: python start_gui.py"
echo "  - Install dev dependencies: pip install -e \".[dev]\""
echo "  - Run tests: pytest"
echo
""")
    
    # Make Unix script executable
    if platform.system() != "Windows":
        os.chmod(unix_script, 0o755)

    print(f"Created activation scripts:")
    print(f"  Windows: {windows_script}")
    print(f"  Unix: {unix_script}")


def validate_environment(venv_path):
    """Validate that the environment is working correctly."""
    python_exe = get_python_executable(venv_path)
    
    print("Validating environment...")
    
    # Test Python version
    result = run_command([str(python_exe), "--version"])
    print(f"Python version: {result.stdout.strip()}")
    
    # Test key imports
    test_imports = [
        "PySide6",
        "numpy", 
        "cv2",
        "scipy",
        "skimage",
        "PIL",
        "matplotlib"
    ]
    
    for module in test_imports:
        result = run_command([
            str(python_exe), "-c", f"import {module}; print(f'{module}: OK')"
        ], check=False)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"{module}: FAILED - {result.stderr.strip()}")


def main():
    """Main setup function."""
    project_root = Path(__file__).parent.absolute()
    venv_path = project_root / ".venv"
    requirements_file = project_root / "requirements.txt"
    
    print("=== SEM/GDS Alignment Tool Environment Setup ===")
    print(f"Project root: {project_root}")
    print(f"Virtual environment: {venv_path}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    
    # Create virtual environment
    create_venv(venv_path)
    
    # Install requirements
    install_requirements(venv_path, requirements_file)
    
    # Create activation scripts
    create_activation_scripts(venv_path, project_root)
    
    # Validate environment
    validate_environment(venv_path)
    
    print("\n=== Setup Complete ===")
    print("To activate the environment:")
    if platform.system() == "Windows":
        print("  activate_env.bat")
    else:
        print("  source activate_env.sh")
    print()
    print("To run the application:")
    print("  python start_gui.py")


if __name__ == "__main__":
    main()

@echo off
REM ZKCNN Multi-Models Installation Script for Windows
REM Phase 3: Advanced Cryptographic ZKCNN with BLS12-381

echo 🚀 Installing ZKCNN Multi-Models (Phase 3)...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv zkcnn_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call zkcnn_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version for compatibility)
echo 🔥 Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
echo 📚 Installing other dependencies...
pip install numpy pandas cryptography

REM Install optional dependencies
echo 🎨 Installing optional dependencies...
pip install matplotlib seaborn jupyter notebook

REM Install development dependencies
echo 🛠️  Installing development dependencies...
pip install pytest black flake8

REM Verify installation
echo ✅ Verifying installation...
python -c "import torch; import numpy as np; import pandas as pd; import cryptography; print('✅ All core dependencies installed successfully!'); print(f'PyTorch version: {torch.__version__}'); print(f'NumPy version: {np.__version__}'); print(f'Pandas version: {pd.__version__}')"

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Activate the virtual environment:
echo    zkcnn_env\Scripts\activate.bat
echo.
echo 2. Run the ZKCNN implementation:
echo    python zkCNN_multi_models.py
echo.
echo 3. Deactivate when done:
echo    deactivate
echo.
echo 📖 For more information, see README.md
pause

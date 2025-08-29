#!/bin/bash

# ZKCNN Multi-Models Installation Script
# Phase 3: Advanced Cryptographic ZKCNN with BLS12-381

echo "🚀 Installing ZKCNN Multi-Models (Phase 3)..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version)"
else
    echo "❌ Python $python_version detected. Please install Python $required_version or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv zkcnn_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source zkcnn_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install numpy pandas cryptography

# Install optional dependencies
echo "🎨 Installing optional dependencies..."
pip install matplotlib seaborn jupyter notebook

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest black flake8

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import numpy as np
import pandas as pd
import cryptography
print('✅ All core dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment:"
echo "   source zkcnn_env/bin/activate"
echo ""
echo "2. Run the ZKCNN implementation:"
echo "   python zkCNN_multi_models.py"
echo ""
echo "3. Deactivate when done:"
echo "   deactivate"
echo ""
echo "📖 For more information, see README.md"

#!/bin/bash
#
# Ludwig ML Pipeline - Linux Installation Script
# Optimized for: 16 GB RAM, 100 GB HDD, 8 CPUs
#

set -e  # Exit on error

echo "========================================"
echo "  Ludwig ML Pipeline - Installation"
echo "========================================"
echo ""
echo "System Requirements:"
echo "  - 16 GB RAM (you have this ✓)"
echo "  - 100 GB HDD (you have this ✓)"
echo "  - 8 CPUs (you have this ✓)"
echo "  - Python 3.9+"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || {
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
}

# Check if venv module is available
python3 -m venv --help > /dev/null 2>&1 || {
    echo "❌ python3-venv not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
}

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "Installing dependencies (this may take 5-10 minutes)..."
echo "Ludwig is a large package with many dependencies..."

# Install PyTorch first (Ludwig dependency)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Ludwig
echo ""
echo "Installing Ludwig..."
pip install ludwig[serve]==0.8.5

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p artifacts
mkdir -p logs
mkdir -p data
mkdir -p results

# Test installation
echo ""
echo "Testing installation..."
python3 -c "import ludwig; print(f'Ludwig version: {ludwig.__version__}')" || {
    echo "❌ Ludwig installation failed"
    exit 1
}

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run example pipeline:"
echo "     python run_pipeline.py examples/du-cell-ludwig.yaml"
echo ""
echo "  3. View results:"
echo "     ls -lh artifacts/ludwig_model/"
echo ""
echo "========================================"

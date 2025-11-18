#!/bin/bash
# Setup script for RottenToRiches conda environment

echo "=========================================="
echo "Setting up RottenToRiches Conda Environment"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^rottentoriches "; then
    echo ""
    echo "⚠️  Environment 'rottentoriches' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n rottentoriches -y
    else
        echo "ℹ️  Using existing environment. To update it, run:"
        echo "   conda env update -f environment.yml --prune"
        echo ""
        echo "📝 Activate with: conda activate rottentoriches"
        exit 0
    fi
fi

# Create environment
echo ""
echo "📦 Creating conda environment 'rottentoriches'..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "❌ Error creating environment"
    exit 1
fi

# Activate environment
echo ""
echo "✅ Environment created successfully!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate rottentoriches"
echo ""
echo "2. Install Playwright browsers:"
echo "   playwright install chromium"
echo ""
echo "3. Download NLTK data (run once):"
echo "   python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('movie_reviews')\""
echo ""
echo "4. You're ready to go! 🚀"
echo ""


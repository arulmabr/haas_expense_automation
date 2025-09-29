#!/bin/bash

# AI Expense Report Generator - Deployment Script
# This script helps deploy the application to various platforms

echo "🚀 AI Expense Report Generator - Deployment Helper"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR not found!"
    echo "Please install Tesseract OCR:"
    echo "  macOS: brew install tesseract"
    echo "  Ubuntu: sudo apt-get install tesseract-ocr"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    exit 1
fi

# Check configuration
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "⚠️  Configuration missing!"
    echo "Please copy .streamlit/secrets.toml.example to .streamlit/secrets.toml"
    echo "and fill in your API keys and credentials."
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "  streamlit run streamlit_expense_app.py"
echo ""
echo "To deploy to Streamlit Cloud:"
echo "  1. Push your code to GitHub"
echo "  2. Go to https://share.streamlit.io"
echo "  3. Connect your repository"
echo "  4. Configure secrets in the Streamlit dashboard"
echo ""
echo "To deploy to Heroku:"
echo "  heroku create your-app-name"
echo "  heroku config:set OPENAI_API_KEY=your-key"
echo "  heroku config:set GOOGLE_SHEET_ID=your-sheet-id"
echo "  git push heroku main"

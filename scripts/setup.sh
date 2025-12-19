#!/bin/bash
# Setup script for the microplastic ML pipeline

set -e

# #region agent log
echo "{\"sessionId\":\"debug-session\",\"runId\":\"setup-check\",\"hypothesisId\":\"A\",\"location\":\"setup.sh:8\",\"message\":\"Checking venv status\",\"data\":{\"venv_exists\":$([ -d "venv" ] && echo "true" || echo "false"),\"activate_exists\":$([ -f "venv/bin/activate" ] && echo "true" || echo "false"),\"pwd\":\"$(pwd)\"},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
# #endregion

echo "Setting up Microplastic ML Pipeline..."

# Create virtual environment
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
        echo "Virtual environment exists but is corrupted. Removing and recreating..."
        rm -rf venv
    else
        echo "Creating virtual environment..."
    fi
    python3 -m venv venv
    
    # Verify venv was created correctly
    if [ ! -f "venv/bin/activate" ]; then
        echo "ERROR: Failed to create virtual environment properly"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed,labels}
mkdir -p models/{checkpoints,saved}
mkdir -p logs
mkdir -p notebooks

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys and configuration"
fi

# Download Kaggle credentials instructions
echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. For Kaggle datasets, set up credentials:"
echo "   - Create ~/.kaggle/kaggle.json with your API credentials"
echo "   - chmod 600 ~/.kaggle/kaggle.json"
echo "3. Run data ingestion: python -m src.ingestion.main"
echo "4. Run preprocessing: python -m src.preprocessing.main"
echo "5. Train models: python -m src.models.train"
echo "6. Start API: uvicorn src.api.main:app --reload"
echo "7. Start dashboard: streamlit run dashboard/app.py"


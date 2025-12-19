#!/bin/bash
# Fix script to install streamlit in the active environment or switch to correct venv

# #region agent log
echo "{\"sessionId\":\"debug-session\",\"runId\":\"fix-streamlit\",\"hypothesisId\":\"A\",\"location\":\"fix_streamlit.sh:5\",\"message\":\"Detecting active environment\",\"data\":{\"VIRTUAL_ENV\":\"${VIRTUAL_ENV:-none}\",\"which_python\":\"$(which python 2>/dev/null || echo 'not found')\"},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
# #endregion

CURRENT_VENV="${VIRTUAL_ENV:-none}"

if [ "$CURRENT_VENV" != "none" ]; then
    echo "Current virtual environment: $CURRENT_VENV"
    
    # Check if streamlit is installed
    if pip show streamlit >/dev/null 2>&1; then
        echo "✓ Streamlit is already installed in current environment"
        streamlit --version
    else
        echo "Installing streamlit in current environment..."
        # #region agent log
        echo "{\"sessionId\":\"debug-session\",\"runId\":\"fix-streamlit\",\"hypothesisId\":\"B\",\"location\":\"fix_streamlit.sh:18\",\"message\":\"Installing streamlit\",\"data\":{\"venv\":\"$CURRENT_VENV\"},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
        # #endregion
        pip install streamlit>=1.25.0
        
        # #region agent log
        echo "{\"sessionId\":\"debug-session\",\"runId\":\"fix-streamlit\",\"hypothesisId\":\"B\",\"location\":\"fix_streamlit.sh:22\",\"message\":\"Verifying streamlit installation\",\"data\":{\"installed\":$(pip show streamlit >/dev/null 2>&1 && echo "true" || echo "false")},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
        # #endregion
        
        if pip show streamlit >/dev/null 2>&1; then
            echo "✓ Streamlit installed successfully"
            streamlit --version
        else
            echo "✗ Failed to install streamlit"
            exit 1
        fi
    fi
else
    echo "No virtual environment active."
    echo "Checking available environments..."
    
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        echo "Found 'venv' with streamlit installed"
        echo "To use it, run: source venv/bin/activate"
    fi
    
    if [ -d "py_venv" ] && [ -f "py_venv/bin/activate" ]; then
        echo "Found 'py_venv' (streamlit not installed)"
        echo "To install streamlit in py_venv, run:"
        echo "  source py_venv/bin/activate"
        echo "  pip install streamlit>=1.25.0"
    fi
fi

# #region agent log
echo "{\"sessionId\":\"debug-session\",\"runId\":\"fix-streamlit\",\"hypothesisId\":\"C\",\"location\":\"fix_streamlit.sh:45\",\"message\":\"Fix script complete\",\"data\":{},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
# #endregion


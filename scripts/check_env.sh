#!/bin/bash
# Environment check script with instrumentation

# #region agent log
echo "{\"sessionId\":\"debug-session\",\"runId\":\"env-check\",\"hypothesisId\":\"A\",\"location\":\"check_env.sh:5\",\"message\":\"Checking active virtual environment\",\"data\":{\"VIRTUAL_ENV\":\"${VIRTUAL_ENV:-none}\",\"which_python\":\"$(which python 2>/dev/null || echo 'not found')\",\"python_version\":\"$(python --version 2>&1 || echo 'error')\"},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
# #endregion

echo "Current Virtual Environment: ${VIRTUAL_ENV:-none}"
echo "Python path: $(which python 2>/dev/null || echo 'not found')"
echo "Python version: $(python --version 2>&1 || echo 'error')"
echo ""

# #region agent log
echo "{\"sessionId\":\"debug-session\",\"runId\":\"env-check\",\"hypothesisId\":\"B\",\"location\":\"check_env.sh:12\",\"message\":\"Checking streamlit installation\",\"data\":{\"streamlit_installed\":$(pip show streamlit >/dev/null 2>&1 && echo "true" || echo "false"),\"streamlit_version\":\"$(pip show streamlit 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'not found')\"},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
# #endregion

if pip show streamlit >/dev/null 2>&1; then
    echo "✓ Streamlit is installed"
    pip show streamlit | grep Version
else
    echo "✗ Streamlit is NOT installed in current environment"
    echo ""
    echo "Available virtual environments:"
    if [ -d "venv" ]; then
        echo "  - venv (has streamlit: $(source venv/bin/activate 2>/dev/null && pip show streamlit >/dev/null 2>&1 && echo 'yes' || echo 'no'))"
    fi
    if [ -d "py_venv" ]; then
        echo "  - py_venv (has streamlit: $(source py_venv/bin/activate 2>/dev/null && pip show streamlit >/dev/null 2>&1 && echo 'yes' || echo 'no'))"
    fi
fi

# #region agent log
echo "{\"sessionId\":\"debug-session\",\"runId\":\"env-check\",\"hypothesisId\":\"C\",\"location\":\"check_env.sh:25\",\"message\":\"Environment check complete\",\"data\":{\"venv_exists\":$([ -d "venv" ] && echo "true" || echo "false"),\"py_venv_exists\":$([ -d "py_venv" ] && echo "true" || echo "false")},\"timestamp\":$(date +%s000)}" >> /home/arvind/Downloads/Projects/.cursor/debug.log 2>/dev/null || true
# #endregion


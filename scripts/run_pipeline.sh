#!/bin/bash
# Complete pipeline execution script

set -e

echo "Starting Microplastic ML Pipeline..."

# 1. Data Ingestion
echo "Step 1: Data Ingestion"
python -m src.ingestion.main --discover-only

# 2. Data Preprocessing
echo "Step 2: Data Preprocessing"
python -m src.preprocessing.main --data-type all

# 3. Model Training
echo "Step 3: Model Training"
python -m src.models.train --model-type all

# 4. Start API
echo "Step 4: Starting API server..."
python -m src.api.main &
API_PID=$!

# 5. Start Monitoring
echo "Step 5: Starting monitoring service..."
python -m src.monitoring.main --test &
MONITORING_PID=$!

# 6. Start Dashboard
echo "Step 6: Starting dashboard..."
streamlit run dashboard/app.py &
DASHBOARD_PID=$!

echo "Pipeline started!"
echo "API PID: $API_PID"
echo "Monitoring PID: $MONITORING_PID"
echo "Dashboard PID: $DASHBOARD_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $API_PID $MONITORING_PID $DASHBOARD_PID 2>/dev/null; exit" INT TERM
wait


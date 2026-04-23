#!/bin/bash

# Proactive AI Customer Churn Engine - Execution Script
# Run each command in a separate terminal/command prompt

echo "=========================================="
echo "Churn Prediction Engine - Startup Guide"
echo "=========================================="
echo ""
echo "IMPORTANT: Run these commands in SEPARATE terminals."
echo ""

echo "STEP 1: Install Dependencies"
echo "-----------------------------"
echo "pip install -r requirements.txt"
echo ""

echo "STEP 2: Train the ML Model"
echo "---------------------------"
echo "python train.py"
echo ""

echo "STEP 3: Start the FastAPI Server (in terminal 1)"
echo "-------------------------------------------------"
echo "python api.py"
echo "(Server runs at http://localhost:8000)"
echo ""

echo "STEP 4: Start the Data Simulator (in terminal 2)"
echo "-------------------------------------------------"
echo "python simulator.py"
echo "(Sends new user data every 3 seconds)"
echo ""

echo "STEP 5: Launch Streamlit Dashboard (in terminal 3)"
echo "---------------------------------------------------"
echo "streamlit run app.py"
echo "(Opens in your browser at http://localhost:8501)"
echo ""

echo "=========================================="
echo "Order of startup:"
echo "1. requirements.txt (one time)"
echo "2. train.py (one time)"
echo "3. api.py (keep running)"
echo "4. simulator.py (keep running)"
echo "5. app.py (keep running)"
echo "=========================================="
# Fraud Detection Model API

## Project Overview
This project implements a complete pipeline for fraud detection in an insurance dataset containing categorical features. The model uses Weight of Evidence (WOE) transformation and a logistic regression scorecard for prediction.

## Features
- Data preprocessing: duplicate removal, categorical validation, missing values handling.
- Model training with WOE transformation and logistic regression.
- Scorecard generation showing points per category.
- REST API to compute risk scores for clients.

## Requirements
All dependencies are listed in `requirements.txt`.  
Install them with:
```bash
pip install -r requirements.txt
```

## Running the API
To start the API server using Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Once started, open in your browser:
Locally: http://127.0.0.1:8000/docs
From another device on the same network: http://<your_local_ip>:8000/docs

## API Endpoints

### **POST** `/score`
Predicts the fraud risk score for a given client.

**Request body (JSON example):**
```json
{
    "Month": "Jan",
    "Make": "Pontiac",
    "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Monday",
    "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": "4",
    "MaritalStatus": "Married",
    "VehiclePrice": "less than 20000",
    "AgeOfVehicle": "more than 7",
    "AgeOfPolicyHolder": "41 to 50",
    "AgentType": "External",
    "AddressChange_Claim": "no change",
    "NumberOfCars": "1 vehicle"
}
```

**Example request with cURL:**
```bash
curl -X POST "http://0.0.0.0:8000/score" -H "Content-Type: application/json" -d '{
    "Month": "Jan",
    "Make": "Pontiac",
    "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Monday",
    "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": "4",
    "MaritalStatus": "Married",
    "VehiclePrice": "less than 20000",
    "AgeOfVehicle": "more than 7",
    "AgeOfPolicyHolder": "41 to 50",
    "AgentType": "External",
    "AddressChange_Claim": "no change",
    "NumberOfCars": "1 vehicle"
}'
```

**Example response:**
```json
{
  "score": 466,
  "probability_fraud": 0.676999,
  "log_odds": 0.740013,
  "used_variables": [
    "Month",
    "Make",
    "AccidentArea",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "WeekOfMonthClaimed",
    "MaritalStatus",
    "VehiclePrice",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "AgentType",
    "AddressChange_Claim",
    "NumberOfCars"
  ],
  "inputs_used": {
    "Month": "Jan",
    "Make": "Pontiac",
    "AccidentArea": "Urban",
    "DayOfWeekClaimed": "Monday",
    "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": "4",
    "MaritalStatus": "Married",
    "VehiclePrice": "less than 20000",
    "AgeOfVehicle": "more than 7",
    "AgeOfPolicyHolder": "41 to 50",
    "AgentType": "External",
    "AddressChange_Claim": "no change",
    "NumberOfCars": "1 vehicle"
  }
}
```

## Notes
- Ensure that the model artifacts (`selected_vars.json`, `woe_mappings.json`, `model.pkl`, etc.) are located in the `artifacts/` folder before starting the API.
- Modify `main.py` if the model or preprocessing changes.

## Results Presentation
The complete results analysis, including visualizations, metrics, and scorecard interpretation, is available in the Jupyter notebook `fraud_detection.ipynb`.
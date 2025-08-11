# Technical Documentation - Fraud Detection Model

## 1. Data Preprocessing

The dataset contains categorical information about insurance customers and policies. The target variable **FraudFound_P** indicates whether a case was fraudulent (1) or not (0), with fraud being a low-frequency event.

### Steps Performed:
1. **Duplicate Removal** – Remove duplicate rows to avoid data leakage and bias was not necessary.
2. **Categorical Consistency** – Verified and standardized category labels (e.g., fixing typos, aligning capitalization).
3. **Encoding using Weight of Evidence (WoE)** – Transformed categorical variables to numerical form while preserving predictive power and monotonic relationship with the target.
4. **Variable Selection** – Retained only variables with statistically significant predictive capacity after WoE transformation.

---

## 2. Model Development

The chosen baseline model was **Logistic Regression** trained on WoE-transformed features.

### Reasons for Logistic Regression:
- Interpretability through coefficients and odds ratios.
- Direct compatibility with scorecard generation.
- Robustness to categorical data when using WoE encoding.

### Training Steps:
- Split dataset into training and test sets.
- Fit logistic regression on selected WoE variables.
- Validate performance using metrics suitable for imbalanced classification.

---

## 3. Model Evaluation

The model was evaluated using the following metrics:

- **AUC (Area Under the ROC Curve)** – Measures the model's ability to discriminate between fraud and non-fraud cases.
- **KS Statistic** – Captures the separation between the cumulative distributions of fraud and non-fraud predictions.
- **Confusion Matrix** – Shows classification distribution across true/false positives and negatives.

The logistic model achieved a **good trade-off between interpretability and performance**.

---

## 4. Scorecard Generation

A scorecard was generated from the logistic regression model using the following scaling parameters:

- **PDO (Points to Double the Odds)**: 20
- **Base Score**: 600
- **Base Odds**: 50:1

Each variable category receives a number of points based on its WoE value and model coefficient.  
Example excerpt from the scorecard:

| Variable            | Category         | WOE      | Coef     | Points |
|---------------------|------------------|----------|----------|--------|
| AccidentArea        | Rural            | 0.3927   | 0.8383   | -10    |
| AccidentArea        | Urban            | -0.0608  | 0.8383   | 1      |
| AddressChange_Claim | under 6 months   | 8.4976   | 1.1183   | -274   |

The total score for a client is obtained by **summing the points for each applicable category plus the intercept points**.

---

## 5. API Usage
The model is deployed as a REST API using **FastAPI**.
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

---

## 6. Reproducibility & Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the notebook `fraud_detection.ipynb` to reproduce preprocessing, training, and scorecard generation.

---

**Author:** Ezau Faridh Torres Torres  
**Institution:** Centro de Investigación en Matemáticas (CIMAT)  

## 7. Results Presentation
For detailed evaluation results, plots, and scorecard insights, refer to the Jupyter notebook `fraud_detection.ipynb`.
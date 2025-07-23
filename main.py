from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field


# Load model and preprocessing objects
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # Dictionary of encoders per categorical column
feature_columns = joblib.load("feature_columns.pkl")  # List of final feature columns used in model

app = FastAPI()


class ClaimInput(BaseModel):
    gender: str = Field(..., alias="Gender")
    provider_specialty: str = Field(..., alias="Provider Specialty")
    provider_location: str = Field(..., alias="Provider Location")
    provider_npi: str = Field(..., alias="Provider NPI")
    diagnosis_code: str = Field(..., alias="Diagnosis Code")
    diagnosis_description: str = Field(..., alias="Diagnosis Description")
    procedure_code: str = Field(..., alias="Procedure Code")
    procedure_description: str = Field(..., alias="Procedure Description")
    claim_amount: float = Field(..., alias="Claim Amount")
    billed_amount: float = Field(..., alias="Billed Amount")
    allowed_amount: float = Field(..., alias="Allowed Amount")
    paid_amount: float = Field(..., alias="Paid Amount")
    patient_responsibility: float = Field(..., alias="Patient Responsibility")
    claim_type: str = Field(..., alias="Claim Type")
    claim_status: str = Field(..., alias="Claim Status")
    billing_code: str = Field(..., alias="Billing Code")
    place_of_service_code: str = Field(..., alias="Place of Service Code")
    billing_frequency: str = Field(..., alias="Billing Frequency")
    referring_physician_id: str = Field(..., alias="Referring Physician ID")
    referring_physician_specialty: str = Field(..., alias="Referring Physician Specialty")
    audit_notes: str = Field(..., alias="Audit Notes")
    prior_authorization: str = Field(..., alias="Prior Authorization")
    claim_rejection_reason: str = Field(..., alias="Claim Rejection Reason")
    appeal_information: str = Field(..., alias="Appeal Information")

@app.post("/predict")
def predict_fraud(claim: ClaimInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([claim.model_dump(by_alias=True)])



    # Feature engineering
    input_df['Submission Delay'] = 5  # fixed as per your requirement
    input_df['Claim_Billed_Diff'] = input_df['Claim Amount'] - input_df['Billed Amount']
    input_df['Billed_Allowed_Diff'] = input_df['Billed Amount'] - input_df['Allowed Amount']
    input_df['Allowed_Paid_Diff'] = input_df['Allowed Amount'] - input_df['Paid Amount']
    input_df['Patient_Resp_Percent'] = input_df['Patient Responsibility'] / input_df['Claim Amount']
    input_df['Claim Count per Patient'] = 1

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            if val not in le.classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value '{val}' for '{col}'. Allowed: {list(le.classes_)}"
                )
            input_df[col] = le.transform([val])

    # Select and scale final features
    try:
        input_df = input_df[feature_columns]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    return {"prediction": "Fraud" if prediction == 1 else "Not Fraud"}


@app.get("/")
def read_root():
    return {"message": "Fraud detection API is running"}

@app.get("/predict")
def predict_info():
    return {"message": "Use POST request to submit data for fraud prediction."}


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import joblib
import os
import json
from datetime import datetime

app = FastAPI(title="Churn Prediction API", version="1.0.0")

model = None

class UserFeatures(BaseModel):
    user_id: str
    days_active: int
    usage_minutes_last_7_days: int
    support_tickets: int
    payment_failures: int

def generate_mock_llm_email(user_data: dict, churn_prob: float) -> str:
    """Generate a mock retention email using simple template logic."""
    tickets = user_data.get('support_tickets', 0)
    usage = user_data.get('usage_minutes_last_7_days', 0)
    failures = user_data.get('payment_failures', 0)
    
    email_parts = [f"Hi {user_data['user_id']},"]
    
    if tickets > 3:
        email_parts.append(f"We noticed you've had {tickets} support tickets recently.")
    if usage < 50:
        email_parts.append("Your recent activity has been lower than usual.")
    if failures > 1:
        email_parts.append(f"We've noticed {failures} payment issues.")
    
    if churn_prob > 0.85:
        email_parts.append("To thank you for your loyalty, we're offering you a 30% discount on your next billing cycle!")
    elif churn_prob > 0.75:
        email_parts.append("We'd love to offer you a 20% discount to keep you satisfied!")
    else:
        email_parts.append("We value your membership and hope to continue serving you.")
    
    email_parts.append("\nBest regards,\nThe Customer Success Team")
    
    return " ".join(email_parts)

def generate_openai_email(user_data: dict, churn_prob: float) -> str:
    """
    OpenAI integration - commented out by default.
    Uncomment and add your API key to use real GenAI.
    """
    # import openai
    # openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    # 
    # prompt = f"""Write a personalized customer retention email for a user who has:
    # - User ID: {user_data['user_id']}
    # - Days Active: {user_data['days_active']}
    # - Usage (mins/week): {user_data['usage_minutes_last_7_days']}
    # - Support Tickets: {user_data['support_tickets']}
    # - Payment Failures: {user_data['payment_failures']}
    # - Churn Probability: {churn_prob:.2%}
    # 
    # Write a brief, personalized email offering help and a discount if appropriate.
    # """
    # 
    # response = openai.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=200
    # )
    # return response.choices[0].message.content
    
    return generate_mock_llm_email(user_data, churn_prob)

def identify_top_risk_factor(user_data: dict) -> str:
    """Identify the highest risk factor based on feature values."""
    features = {
        'support_tickets': user_data.get('support_tickets', 0),
        'payment_failures': user_data.get('payment_failures', 0),
        'low_usage': 500 - user_data.get('usage_minutes_last_7_days', 0),
        'low_engagement': 730 - user_data.get('days_active', 0)
    }
    
    max_feature = max(features, key=features.get)
    
    if max_feature == 'support_tickets':
        return f"High support tickets ({user_data.get('support_tickets', 0)})"
    elif max_feature == 'payment_failures':
        return f"Payment failures ({user_data.get('payment_failures', 0)})"
    elif max_feature == 'low_usage':
        return "Low weekly usage"
    else:
        return "Low engagement (new/inactive account)"

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists('churn_model.pkl'):
        model = joblib.load('churn_model.pkl')
        print("Model loaded successfully!")
    else:
        print("Warning: Model not found. Please run train.py first.")

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running", "status": "healthy"}

@app.post("/predict")
def predict_churn(user: UserFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    user_dict = user.dict()
    features = [[
        user.days_active,
        user.usage_minutes_last_7_days,
        user.support_tickets,
        user.payment_failures
    ]]
    
    churn_prob = model.predict_proba(features)[0][1]
    
    top_risk = identify_top_risk_factor(user_dict)
    
    retention_email = None
    if churn_prob > 0.75:
        retention_email = generate_openai_email(user_dict, churn_prob)
    
    response = {
        "user_id": user.user_id,
        "churn_probability": round(churn_prob, 4),
        "top_risk_factor": top_risk,
        "retention_email": retention_email,
        "timestamp": datetime.now().isoformat()
    }
    
    return response

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
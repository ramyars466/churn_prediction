import requests
import json
import time
import random
import os
from datetime import datetime

API_URL = "http://localhost:8000/predict"
ALERTS_FILE = "live_alerts.json"
MAX_ALERTS = 20

def generate_random_user():
    """Generate random user data for simulation."""
    user_id = f"user_{random.randint(1, 99999):05d}"
    
    user_data = {
        "user_id": user_id,
        "days_active": random.randint(1, 730),
        "usage_minutes_last_7_days": random.randint(0, 500),
        "support_tickets": random.randint(0, 10),
        "payment_failures": random.randint(0, 5)
    }
    
    return user_data

def load_alerts():
    """Load existing alerts from JSON file."""
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_alerts(alerts):
    """Save alerts to JSON file, keeping only the latest MAX_ALERTS."""
    alerts = alerts[-MAX_ALERTS:]
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2)

def send_prediction_request(user_data):
    """Send user data to the API and get prediction."""
    try:
        response = requests.post(API_URL, json=user_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: API returned status {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure api.py is running on port 8000.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    print("=" * 60)
    print("Starting Data Simulator...")
    print(f"Target API: {API_URL}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    initial_alerts = load_alerts()
    if initial_alerts:
        print(f"Loaded {len(initial_alerts)} existing alerts")
    
    while True:
        user_data = generate_random_user()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending data for {user_data['user_id']}...")
        
        result = send_prediction_request(user_data)
        
        if result:
            print(f"  → Churn Probability: {result.get('churn_probability', 0):.2%}")
            print(f"  → Top Risk Factor: {result.get('top_risk_factor', 'N/A')}")
            
            if result.get('retention_email'):
                print(f"  → Retention Email: GENERATED")
                alerts = load_alerts()
                alerts.append(result)
                save_alerts(alerts)
                print(f"  → Alert saved to {ALERTS_FILE}")
            else:
                print(f"  → Retention Email: Not generated (low risk)")
        else:
            print("  → Failed to get prediction")
        
        time.sleep(3)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulator stopped.")
        print(f"Alerts saved to {ALERTS_FILE}")
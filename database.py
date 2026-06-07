import os
import json
from supabase import create_client, Client

SUPABASE_URL = "https://gkdmxcxluhkzvcdefbju.supabase.co"
SUPABASE_KEY = "sb_secret_i6b-WB9yjaVHakpD2n39Pw_CHCH2yKt"

class SupabaseManager:
    def __init__(self):
        self.client: Client = None
        self.connected = False
        try:
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self.connected = True
        except Exception as e:
            print(f"Failed to connect to Supabase: {e}")

    def register_user(self, name: str, phone: str = "", email: str = "") -> dict:
        if not self.connected:
            return {"error": "Not connected to database", "id": "local_user"}
            
        try:
            data = {"name": name, "phone": phone, "email": email}
            response = self.client.table("users").insert(data).execute()
            if len(response.data) > 0:
                return response.data[0]
            return {"error": "Failed to insert user"}
        except Exception as e:
            print(f"Registration error: {e}")
            # If the table doesn't exist, we fallback gracefully for local testing
            return {"error": str(e), "id": "local_user"}

    def store_reading(self, user_id: str, sensor_data: dict, predictions: dict):
        if not self.connected:
            return
            
        try:
            data = {
                "user_id": user_id,
                "gsr": sensor_data.get("gsr", 0),
                "ppg": sensor_data.get("ppg", 0),
                "delta": sensor_data.get("delta", 0),
                "theta": sensor_data.get("theta", 0),
                "low_alpha": sensor_data.get("low_alpha", 0),
                "high_alpha": sensor_data.get("high_alpha", 0),
                "low_beta": sensor_data.get("low_beta", 0),
                "high_beta": sensor_data.get("high_beta", 0),
                "low_gamma": sensor_data.get("low_gamma", 0),
                "mid_gamma": sensor_data.get("mid_gamma", 0),
                "mental_state": predictions.get("state", "Unknown"),
                "state_confidence": predictions.get("state_confidence", 0.0),
                "emotion": predictions.get("emotion", "Unknown"),
                "emotion_confidence": predictions.get("emotion_confidence", 0.0)
            }
            self.client.table("sensor_readings").insert(data).execute()
        except Exception as e:
            print(f"Store reading error: {e}")

# Global instance
db = SupabaseManager()

import customtkinter as ctk
import threading
import time

from database import db
from inference import engine
from sensors import ArduinoReader, EEGReader, SensorData

from screens.welcome import WelcomeScreen
from screens.register import RegisterScreen
from screens.dashboard import DashboardScreen

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class MindSenseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("MindSense AI Desktop")
        self.geometry("1100x700")
        self.minsize(900, 600)
        
        # Internal State
        self.user_data = {}
        self.sensor_data = SensorData()
        self.latest_frame = None
        
        # Readers
        self.arduino = ArduinoReader()
        self.arduino.data_ref = self.sensor_data
        
        self.eeg = EEGReader()
        self.eeg.data_ref = self.sensor_data
        
        # Screens
        self.welcome_screen = WelcomeScreen(self, self)
        self.register_screen = RegisterScreen(self, self)
        self.dashboard_screen = DashboardScreen(self, self)
        
        # Setup Inference Thread
        self.running = True
        engine.load_models()
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self.show_welcome()
        
    def show_welcome(self):
        self.register_screen.pack_forget()
        self.dashboard_screen.pack_forget()
        self.welcome_screen.pack(fill="both", expand=True)
        
    def show_register(self):
        self.welcome_screen.pack_forget()
        self.register_screen.pack(fill="both", expand=True)
        
    def show_dashboard(self):
        self.register_screen.pack_forget()
        self.dashboard_screen.pack(fill="both", expand=True)
        self.dashboard_screen.start(self.user_data)
        
        # Start sensors
        self.arduino.start()
        self.eeg.start()
        self.inference_thread.start()
        
    def register_user(self, name, phone, email):
        def _register():
            user = db.register_user(name, phone, email)
            self.user_data = user
            # Proceed to dashboard
            self.after(0, self.show_dashboard)
            
        threading.Thread(target=_register, daemon=True).start()
        
    def _inference_loop(self):
        while self.running:
            # 1. Get latest sensor data
            vals = self.sensor_data.to_list()
            data_dict = self.sensor_data.to_dict()
            
            # 2. Model Inference
            state_pred = engine.predict_sensor(vals)
            emo_pred = engine.predict_facial(self.latest_frame)
            
            # (Optional) Future predictor
            # future_pred = engine.predict_future(vals)
            
            # 3. Recommendations
            recs = engine.get_recommendations(state_pred.get("state", "NORMAL"))
            
            # 4. Store in DB
            if self.user_data.get("id"):
                predictions = {
                    "state": state_pred.get("state"),
                    "state_confidence": state_pred.get("confidence"),
                    "emotion": emo_pred.get("emotion"),
                    "emotion_confidence": emo_pred.get("confidence")
                }
                db.store_reading(self.user_data["id"], data_dict, predictions)
            
            # 5. Update UI
            if self.dashboard_screen.winfo_ismapped():
                self.after(0, self.dashboard_screen.update_sensors, data_dict)
                self.after(0, self.dashboard_screen.update_predictions, state_pred, emo_pred, recs)
                
            time.sleep(1.0) # Run inference every 1 second
            
    def on_closing(self):
        self.running = False
        self.arduino.stop()
        self.eeg.stop()
        self.dashboard_screen.stop()
        self.destroy()

if __name__ == "__main__":
    app = MindSenseApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

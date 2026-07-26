import customtkinter as ctk
import cv2
from PIL import Image
import threading

class DashboardScreen(ctk.CTkFrame):
    def __init__(self, master, app_controller, **kwargs):
        super().__init__(master, **kwargs)
        self.app_controller = app_controller
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # --- Header ---
        self.header = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        self.title = ctk.CTkLabel(self.header, text="🧠 MindSense AI Dashboard", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00e5cc")
        self.title.pack(side="left", padx=20, pady=15)
        
        self.user_label = ctk.CTkLabel(self.header, text="User: None", font=ctk.CTkFont(size=14))
        self.user_label.pack(side="right", padx=20, pady=15)
        
        # --- Left Column (Sensors & Camera) ---
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.left_panel.grid_rowconfigure(2, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Arduino Sensors
        self.arduino_frame = ctk.CTkFrame(self.left_panel)
        self.arduino_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(self.arduino_frame, text="Biometric Signals", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.gsr_label = ctk.CTkLabel(self.arduino_frame, text="GSR: 0", font=ctk.CTkFont(family="monospace", size=18, weight="bold"), text_color="#ff5e7d")
        self.gsr_label.pack(side="left", expand=True, pady=10)
        
        self.ppg_label = ctk.CTkLabel(self.arduino_frame, text="HR: 0 BPM", font=ctk.CTkFont(family="monospace", size=18, weight="bold"), text_color="#ff5e7d")
        self.ppg_label.pack(side="right", expand=True, pady=10)
        
        # EEG Sensors
        self.eeg_frame = ctk.CTkFrame(self.left_panel)
        self.eeg_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(self.eeg_frame, text="EEG Bands", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, pady=5)
        
        self.eeg_labels = {}
        eeg_names = ["Delta", "Theta", "LowAlpha", "HighAlpha", "LowBeta", "HighBeta", "LowGamma", "MidGamma"]
        for i, name in enumerate(eeg_names):
            lbl = ctk.CTkLabel(self.eeg_frame, text=f"{name[:4]}: 0.00", font=ctk.CTkFont(family="monospace", size=12))
            lbl.grid(row=1 + i//4, column=i%4, padx=5, pady=5)
            self.eeg_labels[name.lower()] = lbl
            
        # Camera Feed
        self.cam_frame = ctk.CTkFrame(self.left_panel)
        self.cam_frame.grid(row=2, column=0, sticky="nsew")
        ctk.CTkLabel(self.cam_frame, text="Live Camera", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="Initializing Camera...")
        self.cam_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # --- Right Column (Predictions & Recs) ---
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Mental State
        self.state_frame = ctk.CTkFrame(self.right_panel)
        self.state_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(self.state_frame, text="Mental State Classification", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.state_lbl = ctk.CTkLabel(self.state_frame, text="ANALYZING...", font=ctk.CTkFont(size=28, weight="bold"), text_color="#00e5cc")
        self.state_lbl.pack(pady=10)
        self.state_conf = ctk.CTkLabel(self.state_frame, text="Confidence: 0%")
        self.state_conf.pack(pady=(0, 10))
        
        # Facial Emotion
        self.emotion_frame = ctk.CTkFrame(self.right_panel)
        self.emotion_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(self.emotion_frame, text="Facial Emotion", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.emotion_lbl = ctk.CTkLabel(self.emotion_frame, text="Neutral", font=ctk.CTkFont(size=24, weight="bold"))
        self.emotion_lbl.pack(pady=10)
        
        # Recommendations
        self.rec_frame = ctk.CTkFrame(self.right_panel)
        self.rec_frame.grid(row=2, column=0, sticky="nsew")
        ctk.CTkLabel(self.rec_frame, text="Health Recommendations", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.rec_textbox = ctk.CTkTextbox(self.rec_frame, wrap="word", font=ctk.CTkFont(size=14))
        self.rec_textbox.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Camera Capture
        self.cap = None
        self.running = False
        
    def start(self, user_data):
        self.user_label.configure(text=f"Patient: {user_data.get('name', 'Unknown')}")
        self.running = True
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            
    def update_camera(self):
        if not self.running: return
        
        ret, frame = self.cap.read()
        if ret:
            # Update app controller with latest frame for emotion detection
            self.app_controller.latest_frame = frame.copy()
            
            # Display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320, 240))
            img = Image.fromarray(frame)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 240))
            self.cam_label.configure(image=ctk_img, text="")
            
        self.after(30, self.update_camera)

    def update_sensors(self, data: dict):
        self.gsr_label.configure(text=f"GSR: {int(data.get('gsr', 0))}")
        self.ppg_label.configure(text=f"HR: {int(data.get('ppg', 0))} BPM")
        
        self.eeg_labels["delta"].configure(text=f"Delt: {data.get('delta', 0):.2f}")
        self.eeg_labels["theta"].configure(text=f"Thet: {data.get('theta', 0):.2f}")
        self.eeg_labels["low_alpha"].configure(text=f"L-Al: {data.get('low_alpha', 0):.2f}")
        self.eeg_labels["high_alpha"].configure(text=f"H-Al: {data.get('high_alpha', 0):.2f}")
        self.eeg_labels["low_beta"].configure(text=f"L-Be: {data.get('low_beta', 0):.2f}")
        self.eeg_labels["high_beta"].configure(text=f"H-Be: {data.get('high_beta', 0):.2f}")
        self.eeg_labels["low_gamma"].configure(text=f"L-Ga: {data.get('low_gamma', 0):.2f}")
        self.eeg_labels["mid_gamma"].configure(text=f"M-Ga: {data.get('mid_gamma', 0):.2f}")
        
    def update_predictions(self, state_dict, emotion_dict, recs):
        state = state_dict.get("state", "Unknown").replace("_", " ")
        conf = state_dict.get("confidence", 0) * 100
        
        color = "#00e5cc" # Greenish/Teal for normal
        if "PANIC" in state: color = "#ff2d55"
        elif "ANXIETY" in state: color = "#ff5e7d"
        elif "STRESS" in state: color = "#ffb547"
        elif "DEPRESSION" in state: color = "#9f6ef5"
        
        self.state_lbl.configure(text=state, text_color=color)
        self.state_conf.configure(text=f"Confidence: {conf:.1f}%")
        
        emo = emotion_dict.get("emotion", "Neutral")
        emo_conf = emotion_dict.get("confidence", 0) * 100
        self.emotion_lbl.configure(text=f"{emo} ({emo_conf:.1f}%)")
        
        self.rec_textbox.configure(state="normal")
        self.rec_textbox.delete("1.0", "end")
        for idx, rec in enumerate(recs):
            self.rec_textbox.insert("end", f"• {rec}\n\n")
        self.rec_textbox.configure(state="disabled")

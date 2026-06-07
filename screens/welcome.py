import customtkinter as ctk
from PIL import Image
import os

class WelcomeScreen(ctk.CTkFrame):
    def __init__(self, master, app_controller, **kwargs):
        super().__init__(master, **kwargs)
        self.app_controller = app_controller
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Center container
        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.grid(row=1, column=0)
        
        # Logo/Title
        self.title_label = ctk.CTkLabel(
            self.center_frame, 
            text="🧠 MindSense AI", 
            font=ctk.CTkFont(family="Space Grotesk", size=48, weight="bold"),
            text_color="#00e5cc"
        )
        self.title_label.pack(pady=(0, 10))
        
        self.subtitle_label = ctk.CTkLabel(
            self.center_frame,
            text="Advanced Mental Health Assessment System",
            font=ctk.CTkFont(family="Space Grotesk", size=20)
        )
        self.subtitle_label.pack(pady=(0, 40))
        
        # Start button
        self.start_btn = ctk.CTkButton(
            self.center_frame,
            text="Get Started",
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#00e5cc",
            text_color="#070b14",
            hover_color="#00b4a2",
            height=50,
            width=200,
            command=self.app_controller.show_register
        )
        self.start_btn.pack()
        
        # Footer
        self.footer = ctk.CTkLabel(
            self,
            text="v4.0 Desktop Edition",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.footer.grid(row=2, column=0, side="bottom", pady=20)

import customtkinter as ctk

class RegisterScreen(ctk.CTkFrame):
    def __init__(self, master, app_controller, **kwargs):
        super().__init__(master, **kwargs)
        self.app_controller = app_controller
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.card = ctk.CTkFrame(self, corner_radius=15)
        self.card.grid(row=1, column=0, padx=20, pady=20)
        
        # Header
        self.header = ctk.CTkLabel(
            self.card, 
            text="Patient Registration", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header.pack(pady=(30, 20), padx=40)
        
        # Form fields
        self.name_entry = ctk.CTkEntry(self.card, placeholder_text="Full Name (Required)", width=300, height=40)
        self.name_entry.pack(pady=10, padx=40)
        
        self.phone_entry = ctk.CTkEntry(self.card, placeholder_text="Phone Number (Optional)", width=300, height=40)
        self.phone_entry.pack(pady=10, padx=40)
        
        self.email_entry = ctk.CTkEntry(self.card, placeholder_text="Email Address (Optional)", width=300, height=40)
        self.email_entry.pack(pady=10, padx=40)
        
        self.error_label = ctk.CTkLabel(self.card, text="", text_color="red")
        self.error_label.pack(pady=5)
        
        # Buttons
        self.btn_frame = ctk.CTkFrame(self.card, fg_color="transparent")
        self.btn_frame.pack(pady=(10, 30))
        
        self.back_btn = ctk.CTkButton(
            self.btn_frame, text="Back", width=120, height=40,
            fg_color="transparent", border_width=1,
            command=self.app_controller.show_welcome
        )
        self.back_btn.pack(side="left", padx=10)
        
        self.submit_btn = ctk.CTkButton(
            self.btn_frame, text="Continue", width=120, height=40,
            fg_color="#00e5cc", text_color="#070b14", hover_color="#00b4a2",
            command=self.submit
        )
        self.submit_btn.pack(side="left", padx=10)
        
    def submit(self):
        name = self.name_entry.get().strip()
        if not name:
            self.error_label.configure(text="Name is required")
            return
            
        self.error_label.configure(text="Connecting to database...", text_color="#00e5cc")
        self.update()
        
        # Register with Supabase via controller
        self.app_controller.register_user(name, self.phone_entry.get().strip(), self.email_entry.get().strip())

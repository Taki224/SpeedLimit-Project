import tkinter as tk
from tkinter import messagebox
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:8080")

class SpeedLimitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SpeedLimit Operator Interface (MVP)")
        self.root.geometry("600x700") # Made slightly wider for sliders
        self.root.configure(bg="#f0f0f0")

        # --- Header ---
        header_frame = tk.Frame(root, bg="#333", pady=15)
        header_frame.pack(fill="x")
        
        title_label = tk.Label(
            header_frame, 
            text="SpeedLimit Control System", 
            font=("Helvetica", 18, "bold"), 
            fg="white", 
            bg="#333"
        )
        title_label.pack()

        # --- Input Section ---
        input_frame = tk.Frame(root, bg="#f0f0f0", pady=10, padx=20)
        input_frame.pack(fill="x")

        # Helper to create synchronized Slider + Entry rows
        def create_slider_row(parent, label_text, row, min_val, max_val, default_val, is_int=False):
            # 1. Label
            lbl = tk.Label(parent, text=label_text, font=("Arial", 10, "bold"), bg="#f0f0f0", anchor="w")
            lbl.grid(row=row, column=0, sticky="w", pady=(15, 0))
            
            # 2. Variable (This links the Slider and Entry)
            if is_int:
                var = tk.IntVar(value=default_val)
            else:
                var = tk.DoubleVar(value=default_val)

            # 3. Slider (Scale)
            # We put the slider in the next row, spanning columns
            scale = tk.Scale(
                parent, 
                from_=min_val, 
                to=max_val, 
                orient=tk.HORIZONTAL, 
                variable=var, 
                bg="#f0f0f0",
                length=300,
                highlightthickness=0
            )
            scale.grid(row=row+1, column=0, columnspan=2, sticky="w", padx=(0, 10))

            # 4. Entry Box (Manual Override)
            entry = tk.Entry(parent, textvariable=var, font=("Arial", 11), width=8)
            entry.grid(row=row+1, column=2, sticky="e", pady=5)
            
            return var

        # Create inputs with ranges based on Case Study Page 5
        self.var_temp = create_slider_row(
            input_frame, "Temperature (°C) [-30 to 70]", 
            row=0, min_val=-30, max_val=70, default_val=20
        )
        
        self.var_water = create_slider_row(
            input_frame, "Water Level (micrometers) [0 to 2000]", 
            row=2, min_val=0, max_val=2000, default_val=0
        )
        # Note: Threshold for Black Ice is 1000
        
        self.var_lux = create_slider_row(
            input_frame, "Illuminance (Lux) [0 to 5000]", 
            row=4, min_val=0, max_val=5000, default_val=5000
        ) 
        # Note: Darkness threshold is 500. Slider focuses on the low range for precision.
        
        self.var_aqi = create_slider_row(
            input_frame, "Air Quality Index (AQI) [0 to 500]", 
            row=6, min_val=0, max_val=500, default_val=30, is_int=True
        )

        # --- Action Button ---
        btn_frame = tk.Frame(root, bg="#f0f0f0", pady=20)
        btn_frame.pack()
        
        calc_btn = tk.Button(
            btn_frame, 
            text="CALCULATE LIMIT", 
            font=("Arial", 12, "bold"), 
            bg="#007bff", 
            fg="black",
            padx=30, 
            pady=10,
            command=self.calculate_limit
        )
        calc_btn.pack()

        # --- Results Section ---
        result_frame = tk.LabelFrame(root, text="System Decision", font=("Arial", 12, "bold"), bg="#fff", padx=20, pady=10)
        result_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Speed Limit Sign
        self.limit_label = tk.Label(
            result_frame, 
            text="--", 
            font=("Helvetica", 48, "bold"), 
            fg="#d9534f", 
            bg="#fff",
            bd=5,
            relief="solid",
            width=4
        )
        self.limit_label.pack(pady=10)
        
        self.lbl_kmh = tk.Label(result_frame, text="km/h", font=("Arial", 14), bg="#fff")
        self.lbl_kmh.pack()

        self.lbl_reason = tk.Label(result_frame, text="Ready", wraplength=450, justify="center", bg="#fff", fg="#555")
        self.lbl_reason.pack(pady=10)
        
        self.lbl_source = tk.Label(result_frame, text="", font=("Arial", 9, "italic"), bg="#fff", fg="#888")
        self.lbl_source.pack(side="bottom")

    def calculate_limit(self):
        try:
            # Get values directly from the synced variables
            temp = self.var_temp.get()
            water = self.var_water.get()
            lux = self.var_lux.get()
            aqi = self.var_aqi.get()

            payload = {
                "temperature_celsius": temp,
                "water_level_micrometers": water,
                "illuminance_lux": lux,
                "aqi": aqi
            }

            # UI Feedback
            self.limit_label.config(text="...", fg="gray")
            self.root.update()

            # API Call
            response = requests.post(f"{CONTROLLER_URL}/decide_speed_limit", json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Display
            speed_val = data.get("final_speed_limit", 0)
            reason_txt = data.get("reason", "Unknown")
            source_txt = data.get("source_service", "Unknown")

            self.limit_label.config(text=str(speed_val), fg="#d9534f" if speed_val < 80 else "#28a745")
            self.lbl_reason.config(text=f"Reason: {reason_txt}")
            self.lbl_source.config(text=f"Source: {source_txt}")

        except requests.exceptions.ConnectionError:
            self.limit_label.config(text="ERR", fg="red")
            messagebox.showerror("Connection Error", f"Cannot reach Controller at\n{CONTROLLER_URL}")
        except Exception as e:
            self.limit_label.config(text="ERR", fg="red")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedLimitApp(root)
    root.mainloop()
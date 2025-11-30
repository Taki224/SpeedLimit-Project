import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import tensorflow as tf

app = FastAPI(title="SpeedLimit AI Inference Service")

# --- Custom Metric Definition (REQUIRED for loading) ---
def r_squared(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# --- Global Artifacts ---
model = None
scaler = None
feature_cols = None
imputer = None 

# File paths
MODEL_FILE = 'speed_limit_model.keras'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'feature_cols.pkl'
IMPUTER_FILE = 'imputer.pkl' 

@app.on_event("startup")
def load_artifacts():
    """Load the Keras model and Joblib support files separately."""
    global model, scaler, feature_cols, imputer
    
    # 1. Load Neural Network (With Custom Object)
    if os.path.exists(MODEL_FILE):
        try:
            # FIXED: Pass custom_objects map so Keras knows what 'r_squared' is
            model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'r_squared': r_squared})
            print(f"SUCCESS: Loaded {MODEL_FILE}")
        except Exception as e:
            print(f"ERROR: Failed to load Keras model with metrics: {e}")
            # Fallback: Try loading without compiling if metric fails (Inference only mode)
            try:
                print("Attempting to load with compile=False...")
                model = tf.keras.models.load_model(MODEL_FILE, compile=False)
                print(f"SUCCESS: Loaded {MODEL_FILE} (Uncompiled)")
            except Exception as e2:
                print(f"FATAL: Could not load model even uncompiled: {e2}")
    else:
        print(f"CRITICAL: {MODEL_FILE} not found.")

    # 2. Load Scaler
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print(f"SUCCESS: Loaded {SCALER_FILE}")
    else:
        print(f"CRITICAL: {SCALER_FILE} not found.")

    # 3. Load Feature Names
    if os.path.exists(FEATURES_FILE):
        feature_cols = joblib.load(FEATURES_FILE)
        print(f"SUCCESS: Loaded {FEATURES_FILE}")
        print(f"Features expected: {feature_cols}")
    else:
        print(f"CRITICAL: {FEATURES_FILE} not found.")

    # 4. Load Imputer
    if os.path.exists(IMPUTER_FILE):
        imputer = joblib.load(IMPUTER_FILE)
        print(f"SUCCESS: Loaded {IMPUTER_FILE}")
    else:
        print(f"WARNING: {IMPUTER_FILE} not found. Defaulting to 0-fill.")


# --- Data Models ---
class SensorData(BaseModel):
    # Map of SensorID to Value. (e.g., ID 1 is Water)
    readings: Dict[int, float] = Field(..., description="Map of SensorID to Value. E.g., {1: 1500.0, 7: -2.0}")

class SpeedLimitResponse(BaseModel):
    recommended_speed: int
    predicted_risk: float
    safety_status: str

# --- Logic ---

@app.post("/optimize_speed_limit", response_model=SpeedLimitResponse)
def find_safe_speed_limit(data: SensorData):
    """
    Tests speed limits from 130 km/h down to 60 km/h.
    Returns the highest speed where predicted accidents < 1.0.
    """
    if model is None or scaler is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Service not ready. Artifacts missing.")

    # 1. Map Sensor IDs to Column Names
    # Based on PDF/CSV Analysis
    id_to_type = {
        1: 'W', 8: 'W', 11: 'W',            # Water
        2: 'L', 9: 'L', 12: 'L',            # Light
        7: 'T', 15: 'T', 16: 'T', 18: 'T',  # Temperature
        3: 'N', 10: 'N',                    # Noise
        4: 'H',                             # Humidity
        5: 'WD', 13: 'WD',                  # Wind Dir
        6: 'WS', 14: 'WS',                  # Wind Str
        17: 'AP'                            # Air Pressure
    }

    # Initialize input row
    # We must ensure we have a value for every feature column
    input_values = {col: np.nan for col in feature_cols} 
    
    # Fill Input Row from Request
    temp_values = {col: [] for col in feature_cols}

    for sensor_id, value in data.readings.items():
        if sensor_id in id_to_type:
            sensor_type = id_to_type[sensor_id]
            if sensor_type in temp_values:
                temp_values[sensor_type].append(value)
    
    # Calculate means for the row
    for col, vals in temp_values.items():
        if vals:
            input_values[col] = sum(vals) / len(vals)

    # 2. Optimization Loop
    candidate_speeds = [130, 120, 110, 100, 90, 80, 70, 60]
    best_speed = 30 
    lowest_risk = 999.0
    found_safe = False

    # Check if we can run
    if 'SPEED_LIMIT' not in feature_cols:
         return SpeedLimitResponse(recommended_speed=80, predicted_risk=0.0, safety_status="Error: SPEED_LIMIT missing in model")

    for speed in candidate_speeds:
        # Update speed in our input dict
        input_values['SPEED_LIMIT'] = float(speed)
        
        # Convert to DataFrame (Correct order is critical)
        df_trial = pd.DataFrame([input_values], columns=feature_cols)
        
        # Impute (if available, otherwise fillna 0)
        if imputer:
            X_imputed = imputer.transform(df_trial)
        else:
            X_imputed = df_trial.fillna(0).values

        # Scale
        X_scaled = scaler.transform(X_imputed)
        
        # Predict
        risk = float(model.predict(X_scaled, verbose=0)[0][0])
        
        if risk < lowest_risk:
            lowest_risk = risk

        # REQ1: Max 1 accident/hour
        if risk < 1.0:
            best_speed = speed
            found_safe = True
            break 

    # 3. Response
    if found_safe:
        status = "Safe"
    else:
        status = "Risk High - Lowest Limit Selected"
        best_speed = 60 

    return SpeedLimitResponse(
        recommended_speed=best_speed,
        predicted_risk=lowest_risk,
        safety_status=status
    )
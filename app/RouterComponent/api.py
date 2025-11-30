from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="SpeedLimit Router Component")

# --- Data Models ---
class WeatherInput(BaseModel):
    illuminance_lux: float = Field(..., description="Sensor reading for Light (Lux)")
    water_level_micrometers: float = Field(..., description="Sensor reading for Water film depth")
    temperature_celsius: float = Field(..., description="Sensor reading for Temperature")

class RouterDecision(BaseModel):
    requires_neural_network: bool
    reason: str

# --- Constants from Requirements ---
# SPEC1: Dark if < 500 lux
THRESHOLD_LUX = 500
# SPEC2: Black Ice if Water > 1000 AND Temp < 0
THRESHOLD_WATER = 1000
THRESHOLD_TEMP = 0

@app.post("/check_conditions", response_model=RouterDecision)
def check_weather_conditions(data: WeatherInput):
    """
    Decides if the Neural Network needs to be consulted based on 
    hard-coded safety thresholds (SPEC1 & SPEC2).
    """
    reasons = []
    
    # Check SPEC1: Darkness
    is_dark = data.illuminance_lux < THRESHOLD_LUX
    if is_dark:
        reasons.append(f"Darkness detected ({data.illuminance_lux} lux < {THRESHOLD_LUX})")

    # Check SPEC2: Black Ice
    is_black_ice = (data.water_level_micrometers > THRESHOLD_WATER) and (data.temperature_celsius < THRESHOLD_TEMP)
    if is_black_ice:
        reasons.append("Black Ice danger detected")

    # Decision Logic
    if is_dark or is_black_ice:
        return RouterDecision(
            requires_neural_network=True,
            reason=" & ".join(reasons)
        )
    else:
        return RouterDecision(
            requires_neural_network=False,
            reason="Conditions are safe (sufficient light, no black ice)."
        )

# Command to run: uvicorn router_service:app --port 8001 --reload
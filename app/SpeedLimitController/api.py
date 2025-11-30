import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI(title="SpeedLimit Main Controller")

# --- Configuration (Docker Ready) ---
# We use environment variables for service URLs so they can be changed 
# in docker-compose.yml later.
ROUTER_SERVICE_URL = os.getenv("ROUTER_SERVICE_URL", "http://localhost:8001")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8002")
AIR_QUALITY_SERVICE_URL = os.getenv("AIR_QUALITY_SERVICE_URL", "http://localhost:8000")

# --- Data Models ---
class UserInput(BaseModel):
    """
    The simplified input form from the MVP GUI.
    """
    temperature_celsius: float = Field(..., description="Current Temperature")
    water_level_micrometers: float = Field(..., description="Water film height on road")
    illuminance_lux: float = Field(..., description="Light level")
    aqi: int = Field(..., description="Air Quality Index")

class FinalSystemDecision(BaseModel):
    final_speed_limit: int
    reason: str
    source_service: str # Which microservice made the decision?

# --- Helper: Sensor Mapping ---
def map_input_to_sensors(data: UserInput) -> Dict[int, float]:
    """
    Maps user inputs to the specific Sensor IDs used during training.
    Based on Case Study Page 5:
    - Sensor 1: Water (W)
    - Sensor 2: Light (L)
    - Sensor 7: Temperature (T)
    """
    return {
        1: data.water_level_micrometers,
        2: data.illuminance_lux,
        7: data.temperature_celsius
        # The Neural Network will impute (fill) the missing sensors (3-6, 8-18) with averages.
    }

# --- Main Endpoint ---

@app.post("/decide_speed_limit", response_model=FinalSystemDecision)
async def decide_speed_limit(user_input: UserInput):
    """
    Orchestrates the decision process:
    1. Check Router (Weather conditions).
    2. If Dangerous -> Use Inference Service (Neural Network).
    3. If Safe -> Use Air Quality Service (LLM).
    4. Return final decision.
    """
    async with httpx.AsyncClient() as client:
        
        # --- STEP 1: Call Router Service ---
        try:
            router_payload = {
                "illuminance_lux": user_input.illuminance_lux,
                "water_level_micrometers": user_input.water_level_micrometers,
                "temperature_celsius": user_input.temperature_celsius
            }
            router_response = await client.post(
                f"{ROUTER_SERVICE_URL}/check_conditions", 
                json=router_payload
            )
            router_data = router_response.json()
            
            # Use .get() for safety
            requires_nn = router_data.get("requires_neural_network", False)
            router_reason = router_data.get("reason", "Unknown")

        except Exception as e:
            # Fail safe if router is down
            raise HTTPException(status_code=503, detail=f"Failed to contact Router Service: {str(e)}")

        # --- STEP 2: Logic Branching ---
        
        # BRANCH A: Dangerous Weather -> Neural Network
        if requires_nn:
            try:
                # Map simple inputs to the dictionary structure the NN expects
                sensor_payload = {"readings": map_input_to_sensors(user_input)}
                
                inference_response = await client.post(
                    f"{INFERENCE_SERVICE_URL}/optimize_speed_limit", 
                    json=sensor_payload
                )
                inference_data = inference_response.json()
                
                return FinalSystemDecision(
                    final_speed_limit=inference_data["recommended_speed"],
                    reason=f"Weather Hazard Detected ({router_reason}). Risk Analysis: {inference_data['safety_status']}",
                    source_service="Neural Network Inference"
                )
            except Exception as e:
                 raise HTTPException(status_code=503, detail=f"Failed to contact Inference Service: {str(e)}")

        # BRANCH B: Safe Weather -> Check Air Quality
        else:
            try:
                aq_payload = {
                    "aqi": user_input.aqi,
                    "description": "User Input from Controller"
                }
                
                aq_response = await client.post(
                    f"{AIR_QUALITY_SERVICE_URL}/calculate_reduction",
                    json=aq_payload
                )
                aq_data = aq_response.json()
                
                # The AQ service calculates max(30, 80 - reduction)
                return FinalSystemDecision(
                    final_speed_limit=aq_data["recommended_speed_limit"],
                    reason=f"Weather is safe. Adjustment based on Air Quality: {aq_data['reason']}",
                    source_service="Air Quality Agent (LLM)"
                )
            except Exception as e:
                 raise HTTPException(status_code=503, detail=f"Failed to contact Air Quality Service: {str(e)}")

# --- How to Run ---
# Command: uvicorn controller_service:app --port 8080 --reload
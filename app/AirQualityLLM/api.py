from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import AirQualityAgent

# Initialize App and Agent
app = FastAPI(title="SpeedLimit Air Quality Service")
agent = AirQualityAgent()

# --- Data Models ---
class AirQualityRequest(BaseModel):
    aqi: int
    description: str = "Unknown"  # Optional text description (e.g., "Hazardous")

class ReductionResponse(BaseModel):
    aqi_received: int
    reduction_kmh: int
    reason: str
    recommended_speed_limit: int

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "SpeedLimit Air Quality Agent"}

@app.post("/calculate_reduction", response_model=ReductionResponse)
def calculate_reduction(request: AirQualityRequest):
    """
    Accepts an AQI value and uses GPT-4o-mini to determine 
    speed limit reduction.
    """
    try:
        # 1. Query the Agent
        result = agent.get_speed_reduction(request.aqi)
        
        reduction = result.get("reduction_kmh", 0)
        reason = result.get("reason", "No reason provided")

        # 2. Calculate Final Speed Limit (Baseline is 80 km/h per REQ3)
        BASELINE_SPEED = 80
        final_limit = max(30, BASELINE_SPEED - reduction) # Ensure we don't go below 30 km/h sanity check

        return ReductionResponse(
            aqi_received=request.aqi,
            reduction_kmh=reduction,
            reason=reason,
            recommended_speed_limit=final_limit
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- How to Run ---
# Command: uvicorn api:app --reload
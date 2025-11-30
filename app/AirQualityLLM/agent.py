import os
from openai import OpenAI
from pydantic import BaseModel, Field

# --- Define the Output Structure using Pydantic ---
class SpeedReductionResponse(BaseModel):
    reduction_kmh: int = Field(
        ..., 
        description="The integer amount to reduce the speed limit by (e.g., 0, 10, 20)."
    )
    reason: str = Field(
        ..., 
        description="A concise reason for the decision based on the AQI rules."
    )

class AirQualityAgent:
    def __init__(self):
        # Initialize OpenAI Client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def get_speed_reduction(self, aqi: int) -> dict:
        """
        Queries the LLM to decide speed limit reduction based on Air Quality Index (AQI).
        Enforces strict Pydantic output structure.
        
        Args:
            aqi (int): The current Air Quality Index.
            
        Returns:
            dict: {"reduction_kmh": int, "reason": str}
        """
        
        # 1. Define the System Persona and Rules
        system_prompt = (
            "You are an automated Traffic Control System for a highway. "
            "Your job is to determine if the speed limit (normally 80 km/h) needs to be reduced "
            "due to poor Air Quality (AQI).\n\n"
            "RULES:\n"
            "1. If AQI is 0-50 (Good): Reduction is ALWAYS 0 km/h.\n"
            "2. If AQI is 51-100 (Moderate): Reduction is usually 0, unless specifically requested, but typically 0.\n"
            "3. If AQI is 101-150 (Unhealthy for Sensitive): Suggest a mild reduction (e.g., 10-20 km/h).\n"
            "4. If AQI is > 150 (Unhealthy/Hazardous): Suggest significant reduction (e.g., 30-40 km/h) to minimize emissions."
        )

        user_prompt = f"The current Air Quality Index (AQI) is {aqi}. Calculate the reduction."

        try:
            # 2. Call OpenAI API using the Structured Outputs (parse) method
            # This enforces the Pydantic schema on the model level.
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=SpeedReductionResponse, # <--- Enforces Pydantic Structure
                temperature=0.0,
            )

            # 3. Extract the Pydantic object directly
            parsed_result = completion.choices[0].message.parsed
            
            # Convert back to dict for the API
            return parsed_result.model_dump()

        except Exception as e:
            print(f"LLM Error: {e}")
            # Fallback
            return {
                "reduction_kmh": 0,
                "reason": "Error connecting to AI Agent. Defaulting to safe baseline."
            }

# --- Quick Test Block ---
if __name__ == "__main__":
    agent = AirQualityAgent()
    
    # Test Scenarios
    test_aqis = [30, 120, 160]
    for test in test_aqis:
        print(f"Testing AQI: {test}")
        result = agent.get_speed_reduction(test)
        print(result)
        print("-" * 20)
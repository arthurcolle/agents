"""
Sample Dynamic Tool - Example of how to create a dynamic tool
This file is not meant to be imported directly, but to be used as a reference
or copy-pasted when creating a dynamic tool through the agent.
"""

import json
import datetime
import random
from typing import Dict, List, Any, Optional

def weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Generate a mock weather forecast for a location
    
    Args:
        location: City or location to generate forecast for
        days: Number of days to forecast (1-7)
        
    Returns:
        Dictionary with forecast data
    """
    # Cap days between 1 and 7
    days = max(1, min(7, days))
    
    # Weather conditions
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy", "Snowy", "Foggy", "Windy"]
    
    # Generate base temperature based on location (just a mock example)
    # In a real tool, this would use actual weather APIs
    location_hash = sum(ord(c) for c in location) % 100
    base_temp = 15 + (location_hash % 25)  # Between 15°C and 40°C
    
    # Generate forecast
    forecast = []
    today = datetime.datetime.now()
    
    for i in range(days):
        day_date = today + datetime.timedelta(days=i)
        condition = random.choice(conditions)
        
        # Temperature varies by day
        temp_variation = random.randint(-5, 5)
        temp = base_temp + temp_variation
        
        # Precipitation chance
        precipitation = 0
        if condition in ["Rainy", "Stormy", "Snowy"]:
            precipitation = random.randint(30, 90)
        elif condition in ["Partly Cloudy", "Cloudy"]:
            precipitation = random.randint(0, 30)
            
        # Wind speed
        wind_speed = random.randint(0, 30)
        if condition == "Windy":
            wind_speed = random.randint(20, 50)
            
        forecast.append({
            "date": day_date.strftime("%Y-%m-%d"),
            "day_of_week": day_date.strftime("%A"),
            "condition": condition,
            "temperature": {
                "celsius": round(temp, 1),
                "fahrenheit": round((temp * 9/5) + 32, 1)
            },
            "precipitation": {
                "chance": precipitation,
                "mm": round(precipitation / 10, 1) if precipitation > 0 else 0
            },
            "wind": {
                "speed_kmh": wind_speed,
                "speed_mph": round(wind_speed * 0.621371, 1)
            }
        })
    
    return {
        "location": location,
        "forecast_generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "days": days,
        "forecast": forecast,
        "note": "This is a mock forecast for demonstration purposes only"
    }

# Example of calling the function: 
# result = weather_forecast("New York", 5)
# print(json.dumps(result, indent=2))
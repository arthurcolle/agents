#!/usr/bin/env python3
"""
Simple test script for the weather functionality in the CLI agent
"""

import os
import sys
import json
from weather_script import get_current_weather, get_forecast
from tool_templates import create_weather_tool

def test_weather_script():
    """Test the weather_script functionality"""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        print("OPENWEATHERMAP_API_KEY environment variable not set.")
        print("Please set it with: export OPENWEATHERMAP_API_KEY=your_api_key_here")
        return False

    print("Testing weather_script.py...")
    try:
        # Test current weather
        print("Getting current weather for San Francisco...")
        weather_data = get_current_weather(api_key, "San Francisco")
        if weather_data:
            print("Success! Got weather data:")
            summary = weather_data.get_summary()
            print(f"Location: {summary['location']}")
            print(f"Temperature: {summary['temperature']}°C")
            print(f"Conditions: {summary['condition']}")
        else:
            print("Failed to get weather data.")
            return False

        # Test forecast
        print("\nGetting forecast for San Francisco...")
        forecast_data = get_forecast(api_key, "San Francisco", 3)
        if forecast_data:
            print("Success! Got forecast data:")
            summary = forecast_data.get_summary()
            print(f"Location: {summary['location']}")
            print(f"Days: {len(summary['days'])}")
            for day in summary['days']:
                print(f"- {day['date']}: {day['min_temp']}°C to {day['max_temp']}°C, {day['condition']}")
        else:
            print("Failed to get forecast data.")
            return False

        return True
    except Exception as e:
        print(f"Error testing weather script: {e}")
        return False

def test_weather_tool():
    """Test the weather tool template"""
    print("\nTesting tool_templates.py weather tool...")
    try:
        weather_tool = create_weather_tool()
        
        print(f"Tool name: {weather_tool['name']}")
        print(f"Description: {weather_tool['description']}")
        print(f"Category: {weather_tool['category']}")
        print(f"Parameters: {json.dumps(weather_tool['parameters'], indent=2)}")
        
        # Check the code
        code_sample = weather_tool['code'][:100] + "..." if len(weather_tool['code']) > 100 else weather_tool['code']
        print(f"Code sample: {code_sample}")
        
        return True
    except Exception as e:
        print(f"Error testing weather tool template: {e}")
        return False

if __name__ == "__main__":
    success_script = test_weather_script()
    success_tool = test_weather_tool()
    
    if success_script and success_tool:
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed. Check the errors above.")
        sys.exit(1)
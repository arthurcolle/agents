#!/usr/bin/env python3
"""
Simple weather script to get current weather conditions for a location
"""

import os
import sys
import requests
import json
from datetime import datetime

def get_current_weather(api_key, city):
    """
    Get current weather for a city using OpenWeatherMap API
    
    Args:
        api_key: OpenWeatherMap API key
        city: City name to get weather for
        
    Returns:
        Dictionary with weather information or None if request failed
    """
    base_url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # Use metric units (Celsius)
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def parse_weather_data(weather_data):
    """
    Parse weather data from OpenWeatherMap API response
    
    Args:
        weather_data: JSON response from OpenWeatherMap API
        
    Returns:
        Tuple of (weather condition, description, temperature, humidity, wind speed)
        or None if parsing failed
    """
    if not weather_data:
        return None
    
    try:
        # Extract basic weather information
        condition = weather_data['weather'][0]['main']
        description = weather_data['weather'][0]['description']
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data.get('wind', {}).get('speed', 0)
        
        return (condition, description, temperature, humidity, wind_speed)
    except (KeyError, IndexError) as e:
        print(f"Error parsing weather data: {e}")
        return None

def display_weather(city, weather_info):
    """
    Display weather information in a formatted way
    
    Args:
        city: City name
        weather_info: Tuple of weather information from parse_weather_data
    """
    if not weather_info:
        print(f"Could not retrieve weather information for {city}")
        return
    
    condition, description, temperature, humidity, wind_speed = weather_info
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*50)
    print(f"Current Weather for {city} (as of {current_time})")
    print("="*50)
    print(f"Condition: {condition} ({description})")
    print(f"Temperature: {temperature}°C")
    print(f"Humidity: {humidity}%")
    print(f"Wind Speed: {wind_speed} m/s")
    print("="*50 + "\n")

def main():
    # Get API key from environment variable
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    
    if not api_key:
        print("Error: OPENWEATHERMAP_API_KEY environment variable not set")
        print("Please set your OpenWeatherMap API key as an environment variable")
        print("Example: export OPENWEATHERMAP_API_KEY=your_api_key_here")
        return 1
    
    # Get city from command line argument or use default
    if len(sys.argv) > 1:
        city = sys.argv[1]
    else:
        city = "Bermuda"  # Default city
    
    # Get and display weather
    weather_data = get_current_weather(api_key, city)
    weather_info = parse_weather_data(weather_data)
    display_weather(city, weather_info)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

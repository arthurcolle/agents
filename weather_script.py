#!/usr/bin/env python3
"""
Advanced Weather Script

A comprehensive tool for retrieving and analyzing weather data from multiple sources,
with support for current conditions, forecasts, historical data, and data visualization.
"""

import os
import sys
import requests
import json
import argparse
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import textwrap
import math

# Optional imports for enhanced features
try:
    from colorama import init, Fore, Style
    COLORS_AVAILABLE = True
    init()  # Initialize colorama
except ImportError:
    COLORS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False

# Weather condition emoji mappings
WEATHER_EMOJIS = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Drizzle": "🌦️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️",
    "Mist": "🌫️",
    "Fog": "🌫️",
    "Haze": "🌫️",
    "Smoke": "🌫️",
    "Dust": "🌫️",
    "Sand": "🌫️",
    "Ash": "🌫️",
    "Squall": "💨",
    "Tornado": "🌪️"
}

# Weather condition color mappings
WEATHER_COLORS = {
    "Clear": Fore.YELLOW if COLORS_AVAILABLE else "",
    "Clouds": Fore.CYAN if COLORS_AVAILABLE else "",
    "Rain": Fore.BLUE if COLORS_AVAILABLE else "",
    "Drizzle": Fore.BLUE if COLORS_AVAILABLE else "",
    "Thunderstorm": Fore.MAGENTA if COLORS_AVAILABLE else "",
    "Snow": Fore.WHITE if COLORS_AVAILABLE else "",
    "Mist": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Fog": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Haze": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Smoke": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Dust": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Sand": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Ash": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Squall": Fore.LIGHTBLACK_EX if COLORS_AVAILABLE else "",
    "Tornado": Fore.RED if COLORS_AVAILABLE else ""
}

class WeatherAPI:
    """Base class for weather API implementations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_current_weather(self, location: str, units: str = "metric") -> Dict:
        """Get current weather for a location"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Dict:
        """Get weather forecast for a location"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_historical_weather(self, location: str, date: datetime, units: str = "metric") -> Dict:
        """Get historical weather for a location on a specific date"""
        raise NotImplementedError("Subclasses must implement this method")

class OpenWeatherMapAPI(WeatherAPI):
    """OpenWeatherMap API implementation"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, location: str, units: str = "metric") -> Dict:
        """
        Get current weather for a location using OpenWeatherMap API
        
        Args:
            location: Location name or coordinates (lat,lon)
            units: Units of measurement (metric, imperial, standard)
            
        Returns:
            Dictionary with weather information or empty dict if request failed
        """
        # Check if location is coordinates (lat,lon)
        if "," in location and all(part.replace('.', '', 1).replace('-', '', 1).isdigit() 
                                  for part in location.split(',')):
            lat, lon = location.split(',')
            params = {
                "lat": lat.strip(),
                "lon": lon.strip(),
                "appid": self.api_key,
                "units": units
            }
        else:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
        
        try:
            response = requests.get(f"{self.base_url}/weather", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {}
            
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Dict:
        """
        Get weather forecast for a location
        
        Args:
            location: Location name or coordinates (lat,lon)
            days: Number of days for forecast (max 5 for free tier)
            units: Units of measurement (metric, imperial, standard)
            
        Returns:
            Dictionary with forecast information or empty dict if request failed
        """
        # Check if location is coordinates (lat,lon)
        if "," in location and all(part.replace('.', '', 1).replace('-', '', 1).isdigit() 
                                  for part in location.split(',')):
            lat, lon = location.split(',')
            params = {
                "lat": lat.strip(),
                "lon": lon.strip(),
                "appid": self.api_key,
                "units": units,
                "cnt": min(days * 8, 40)  # 8 forecasts per day, max 40 (5 days)
            }
        else:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units,
                "cnt": min(days * 8, 40)  # 8 forecasts per day, max 40 (5 days)
            }
        
        try:
            response = requests.get(f"{self.base_url}/forecast", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast data: {e}")
            return {}
            
    def get_historical_weather(self, location: str, date: datetime, units: str = "metric") -> Dict:
        """
        Get historical weather for a location on a specific date
        
        Args:
            location: Location name or coordinates (lat,lon)
            date: Date for historical weather
            units: Units of measurement (metric, imperial, standard)
            
        Returns:
            Dictionary with historical weather information or empty dict if request failed
        """
        # For historical data, we need coordinates
        if "," in location and all(part.replace('.', '', 1).replace('-', '', 1).isdigit() 
                                  for part in location.split(',')):
            lat, lon = location.split(',')
        else:
            # Try to geocode the location
            if GEOCODING_AVAILABLE:
                geolocator = Nominatim(user_agent="weather_script")
                try:
                    location_data = geolocator.geocode(location)
                    if location_data:
                        lat, lon = location_data.latitude, location_data.longitude
                    else:
                        print(f"Could not geocode location: {location}")
                        return {}
                except Exception as e:
                    print(f"Error geocoding location: {e}")
                    return {}
            else:
                print("Geocoding not available. Please install geopy or provide coordinates.")
                return {}
        
        # Convert date to Unix timestamp
        timestamp = int(date.timestamp())
        
        params = {
            "lat": lat,
            "lon": lon,
            "dt": timestamp,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = requests.get(f"{self.base_url}/onecall/timemachine", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical data: {e}")
            return {}

class WeatherData:
    """Class for parsing and storing weather data"""
    
    def __init__(self, data: Dict, data_type: str = "current"):
        """
        Initialize weather data
        
        Args:
            data: Weather data dictionary from API
            data_type: Type of data (current, forecast, historical)
        """
        self.raw_data = data
        self.data_type = data_type
        self.parsed_data = self._parse_data()
        
    def _parse_data(self) -> Dict:
        """Parse raw weather data based on data type"""
        if not self.raw_data:
            return {}
            
        if self.data_type == "current":
            return self._parse_current_weather()
        elif self.data_type == "forecast":
            return self._parse_forecast()
        elif self.data_type == "historical":
            return self._parse_historical()
        else:
            return {}
            
    def _parse_current_weather(self) -> Dict:
        """Parse current weather data"""
        try:
            data = self.raw_data
            
            # Basic weather information
            weather = {
                "location": {
                    "name": data.get("name", "Unknown"),
                    "country": data.get("sys", {}).get("country", ""),
                    "coordinates": {
                        "lat": data.get("coord", {}).get("lat"),
                        "lon": data.get("coord", {}).get("lon")
                    },
                    "timezone": data.get("timezone", 0)  # Timezone offset in seconds
                },
                "weather": {
                    "condition": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"],
                    "icon": data["weather"][0]["icon"],
                    "id": data["weather"][0]["id"]
                },
                "measurements": {
                    "temperature": {
                        "current": data["main"]["temp"],
                        "feels_like": data["main"]["feels_like"],
                        "min": data["main"]["temp_min"],
                        "max": data["main"]["temp_max"]
                    },
                    "pressure": data["main"]["pressure"],  # hPa
                    "humidity": data["main"]["humidity"],  # %
                    "visibility": data.get("visibility", 0),  # meters
                    "wind": {
                        "speed": data.get("wind", {}).get("speed", 0),
                        "direction": data.get("wind", {}).get("deg", 0),
                        "gust": data.get("wind", {}).get("gust", 0)
                    },
                    "clouds": data.get("clouds", {}).get("all", 0),  # % cloud cover
                    "rain": {
                        "1h": data.get("rain", {}).get("1h", 0),  # mm in last hour
                        "3h": data.get("rain", {}).get("3h", 0)   # mm in last 3 hours
                    },
                    "snow": {
                        "1h": data.get("snow", {}).get("1h", 0),  # mm in last hour
                        "3h": data.get("snow", {}).get("3h", 0)   # mm in last 3 hours
                    }
                },
                "sun": {
                    "sunrise": data.get("sys", {}).get("sunrise", 0),  # Unix timestamp
                    "sunset": data.get("sys", {}).get("sunset", 0)     # Unix timestamp
                },
                "timestamp": data.get("dt", 0)  # Unix timestamp of data calculation
            }
            
            return weather
        except (KeyError, IndexError) as e:
            print(f"Error parsing current weather data: {e}")
            return {}
            
    def _parse_forecast(self) -> Dict:
        """Parse forecast data"""
        try:
            data = self.raw_data
            
            # Basic location information
            location = {
                "name": data.get("city", {}).get("name", "Unknown"),
                "country": data.get("city", {}).get("country", ""),
                "coordinates": {
                    "lat": data.get("city", {}).get("coord", {}).get("lat"),
                    "lon": data.get("city", {}).get("coord", {}).get("lon")
                },
                "timezone": data.get("city", {}).get("timezone", 0)  # Timezone offset in seconds
            }
            
            # Parse forecast list
            forecasts = []
            for item in data.get("list", []):
                forecast = {
                    "timestamp": item.get("dt", 0),  # Unix timestamp
                    "weather": {
                        "condition": item["weather"][0]["main"],
                        "description": item["weather"][0]["description"],
                        "icon": item["weather"][0]["icon"],
                        "id": item["weather"][0]["id"]
                    },
                    "measurements": {
                        "temperature": {
                            "current": item["main"]["temp"],
                            "feels_like": item["main"]["feels_like"],
                            "min": item["main"]["temp_min"],
                            "max": item["main"]["temp_max"]
                        },
                        "pressure": item["main"]["pressure"],  # hPa
                        "humidity": item["main"]["humidity"],  # %
                        "visibility": item.get("visibility", 0),  # meters
                        "wind": {
                            "speed": item.get("wind", {}).get("speed", 0),
                            "direction": item.get("wind", {}).get("deg", 0),
                            "gust": item.get("wind", {}).get("gust", 0)
                        },
                        "clouds": item.get("clouds", {}).get("all", 0),  # % cloud cover
                        "rain": {
                            "3h": item.get("rain", {}).get("3h", 0)   # mm in 3 hours
                        },
                        "snow": {
                            "3h": item.get("snow", {}).get("3h", 0)   # mm in 3 hours
                        }
                    },
                    "datetime": item.get("dt_txt", "")  # Datetime text
                }
                forecasts.append(forecast)
            
            return {
                "location": location,
                "forecasts": forecasts,
                "count": len(forecasts)
            }
        except (KeyError, IndexError) as e:
            print(f"Error parsing forecast data: {e}")
            return {}
            
    def _parse_historical(self) -> Dict:
        """Parse historical data"""
        try:
            data = self.raw_data
            
            # Basic location information
            location = {
                "coordinates": {
                    "lat": data.get("lat"),
                    "lon": data.get("lon")
                },
                "timezone": data.get("timezone", 0)  # Timezone offset in seconds
            }
            
            # Parse historical data
            historical_data = {
                "timestamp": data.get("current", {}).get("dt", 0),  # Unix timestamp
                "weather": {
                    "condition": data.get("current", {}).get("weather", [{}])[0].get("main", "Unknown"),
                    "description": data.get("current", {}).get("weather", [{}])[0].get("description", ""),
                    "icon": data.get("current", {}).get("weather", [{}])[0].get("icon", ""),
                    "id": data.get("current", {}).get("weather", [{}])[0].get("id", 0)
                },
                "measurements": {
                    "temperature": data.get("current", {}).get("temp"),
                    "feels_like": data.get("current", {}).get("feels_like"),
                    "pressure": data.get("current", {}).get("pressure"),  # hPa
                    "humidity": data.get("current", {}).get("humidity"),  # %
                    "dew_point": data.get("current", {}).get("dew_point"),
                    "clouds": data.get("current", {}).get("clouds"),  # % cloud cover
                    "uvi": data.get("current", {}).get("uvi"),  # UV index
                    "visibility": data.get("current", {}).get("visibility"),  # meters
                    "wind": {
                        "speed": data.get("current", {}).get("wind_speed", 0),
                        "direction": data.get("current", {}).get("wind_deg", 0),
                        "gust": data.get("current", {}).get("wind_gust", 0)
                    },
                    "rain": {
                        "1h": data.get("current", {}).get("rain", {}).get("1h", 0)  # mm in last hour
                    },
                    "snow": {
                        "1h": data.get("current", {}).get("snow", {}).get("1h", 0)  # mm in last hour
                    }
                }
            }
            
            return {
                "location": location,
                "historical": historical_data
            }
        except (KeyError, IndexError) as e:
            print(f"Error parsing historical data: {e}")
            return {}
    
    def get_summary(self) -> Dict:
        """Get a summary of the weather data"""
        if not self.parsed_data:
            return {}
            
        if self.data_type == "current":
            return self._get_current_summary()
        elif self.data_type == "forecast":
            return self._get_forecast_summary()
        elif self.data_type == "historical":
            return self._get_historical_summary()
        else:
            return {}
            
    def _get_current_summary(self) -> Dict:
        """Get a summary of current weather data"""
        if not self.parsed_data:
            return {}
            
        data = self.parsed_data
        
        return {
            "location": f"{data['location']['name']}, {data['location']['country']}",
            "condition": data['weather']['condition'],
            "description": data['weather']['description'],
            "temperature": data['measurements']['temperature']['current'],
            "feels_like": data['measurements']['temperature']['feels_like'],
            "humidity": data['measurements']['humidity'],
            "wind_speed": data['measurements']['wind']['speed'],
            "wind_direction": data['measurements']['wind']['direction'],
            "pressure": data['measurements']['pressure'],
            "visibility": data['measurements']['visibility'],
            "clouds": data['measurements']['clouds'],
            "rain_1h": data['measurements']['rain']['1h'],
            "snow_1h": data['measurements']['snow']['1h'],
            "sunrise": data['sun']['sunrise'],
            "sunset": data['sun']['sunset'],
            "timestamp": data['timestamp']
        }
        
    def _get_forecast_summary(self) -> Dict:
        """Get a summary of forecast data"""
        if not self.parsed_data:
            return {}
            
        data = self.parsed_data
        
        # Group forecasts by day
        forecasts_by_day = {}
        for forecast in data['forecasts']:
            date = forecast['datetime'].split()[0]  # Extract date part
            if date not in forecasts_by_day:
                forecasts_by_day[date] = []
            forecasts_by_day[date].append(forecast)
        
        # Create daily summaries
        daily_summaries = []
        for date, forecasts in forecasts_by_day.items():
            # Calculate min/max temperatures
            min_temp = min(f['measurements']['temperature']['min'] for f in forecasts)
            max_temp = max(f['measurements']['temperature']['max'] for f in forecasts)
            
            # Count weather conditions
            conditions = {}
            for f in forecasts:
                condition = f['weather']['condition']
                conditions[condition] = conditions.get(condition, 0) + 1
            
            # Get most common condition
            most_common_condition = max(conditions.items(), key=lambda x: x[1])[0]
            
            # Calculate average values
            avg_humidity = sum(f['measurements']['humidity'] for f in forecasts) / len(forecasts)
            avg_wind_speed = sum(f['measurements']['wind']['speed'] for f in forecasts) / len(forecasts)
            
            # Calculate precipitation probability
            rain_forecasts = sum(1 for f in forecasts if f['measurements']['rain']['3h'] > 0)
            snow_forecasts = sum(1 for f in forecasts if f['measurements']['snow']['3h'] > 0)
            precipitation_probability = (rain_forecasts + snow_forecasts) / len(forecasts)
            
            daily_summaries.append({
                "date": date,
                "condition": most_common_condition,
                "min_temp": min_temp,
                "max_temp": max_temp,
                "avg_humidity": avg_humidity,
                "avg_wind_speed": avg_wind_speed,
                "precipitation_probability": precipitation_probability,
                "forecasts_count": len(forecasts)
            })
        
        return {
            "location": f"{data['location']['name']}, {data['location']['country']}",
            "days": daily_summaries,
            "days_count": len(daily_summaries)
        }
        
    def _get_historical_summary(self) -> Dict:
        """Get a summary of historical data"""
        if not self.parsed_data:
            return {}
            
        data = self.parsed_data
        historical = data.get('historical', {})
        
        return {
            "coordinates": f"{data['location']['coordinates']['lat']}, {data['location']['coordinates']['lon']}",
            "condition": historical.get('weather', {}).get('condition', 'Unknown'),
            "description": historical.get('weather', {}).get('description', ''),
            "temperature": historical.get('measurements', {}).get('temperature'),
            "feels_like": historical.get('measurements', {}).get('feels_like'),
            "humidity": historical.get('measurements', {}).get('humidity'),
            "wind_speed": historical.get('measurements', {}).get('wind', {}).get('speed'),
            "wind_direction": historical.get('measurements', {}).get('wind', {}).get('direction'),
            "pressure": historical.get('measurements', {}).get('pressure'),
            "visibility": historical.get('measurements', {}).get('visibility'),
            "clouds": historical.get('measurements', {}).get('clouds'),
            "timestamp": historical.get('timestamp')
        }

class WeatherFormatter:
    """Class for formatting weather data for display"""
    
    @staticmethod
    def format_current_weather(weather_data: WeatherData, use_colors: bool = True, use_emoji: bool = True) -> str:
        """
        Format current weather data for display
        
        Args:
            weather_data: WeatherData object with current weather
            use_colors: Whether to use colors in output
            use_emoji: Whether to use emoji in output
            
        Returns:
            Formatted string with weather information
        """
        summary = weather_data.get_summary()
        if not summary:
            return "No weather data available"
        
        # Get emoji and color for condition
        condition = summary['condition']
        emoji = WEATHER_EMOJIS.get(condition, "🌡️") if use_emoji else ""
        color = WEATHER_COLORS.get(condition, "") if use_colors and COLORS_AVAILABLE else ""
        reset = Style.RESET_ALL if use_colors and COLORS_AVAILABLE else ""
        
        # Format sunrise and sunset times
        sunrise = datetime.fromtimestamp(summary['sunrise']).strftime("%H:%M")
        sunset = datetime.fromtimestamp(summary['sunset']).strftime("%H:%M")
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(summary['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format wind direction as cardinal direction
        wind_direction = summary['wind_direction']
        cardinal_direction = WeatherFormatter._degrees_to_cardinal(wind_direction)
        
        # Format visibility in km
        visibility_km = summary['visibility'] / 1000
        
        # Format precipitation
        precipitation = ""
        if summary['rain_1h'] > 0:
            precipitation += f"Rain (1h): {summary['rain_1h']} mm\n"
        if summary['snow_1h'] > 0:
            precipitation += f"Snow (1h): {summary['snow_1h']} mm\n"
        
        # Build output
        output = []
        output.append("=" * 60)
        output.append(f"{color}{emoji} Current Weather for {summary['location']} (as of {timestamp}){reset}")
        output.append("=" * 60)
        output.append(f"{color}Condition: {condition} ({summary['description']}){reset}")
        output.append(f"Temperature: {summary['temperature']}°C (Feels like: {summary['feels_like']}°C)")
        output.append(f"Humidity: {summary['humidity']}%")
        output.append(f"Wind: {summary['wind_speed']} m/s from {cardinal_direction} ({wind_direction}°)")
        output.append(f"Pressure: {summary['pressure']} hPa")
        output.append(f"Visibility: {visibility_km:.1f} km")
        output.append(f"Cloud Cover: {summary['clouds']}%")
        if precipitation:
            output.append(precipitation.rstrip())
        output.append(f"Sunrise: {sunrise} | Sunset: {sunset}")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    @staticmethod
    def format_forecast(weather_data: WeatherData, use_colors: bool = True, use_emoji: bool = True) -> str:
        """
        Format forecast data for display
        
        Args:
            weather_data: WeatherData object with forecast
            use_colors: Whether to use colors in output
            use_emoji: Whether to use emoji in output
            
        Returns:
            Formatted string with forecast information
        """
        summary = weather_data.get_summary()
        if not summary:
            return "No forecast data available"
        
        reset = Style.RESET_ALL if use_colors and COLORS_AVAILABLE else ""
        
        # Build output
        output = []
        output.append("=" * 60)
        output.append(f"Weather Forecast for {summary['location']}")
        output.append("=" * 60)
        
        for day in summary['days']:
            # Get emoji and color for condition
            condition = day['condition']
            emoji = WEATHER_EMOJIS.get(condition, "🌡️") if use_emoji else ""
            color = WEATHER_COLORS.get(condition, "") if use_colors and COLORS_AVAILABLE else ""
            
            # Format date
            date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
            date_str = date_obj.strftime("%A, %b %d")  # e.g., "Monday, Jan 01"
            
            # Format precipitation probability as percentage
            precip_percent = int(day['precipitation_probability'] * 100)
            
            output.append(f"\n{color}{emoji} {date_str}{reset}")
            output.append(f"{color}Condition: {condition}{reset}")
            output.append(f"Temperature: {day['min_temp']}°C to {day['max_temp']}°C")
            output.append(f"Humidity: {day['avg_humidity']:.0f}%")
            output.append(f"Wind: {day['avg_wind_speed']:.1f} m/s")
            output.append(f"Precipitation Chance: {precip_percent}%")
        
        output.append("\n" + "=" * 60)
        
        return "\n".join(output)
    
    @staticmethod
    def format_historical(weather_data: WeatherData, use_colors: bool = True, use_emoji: bool = True) -> str:
        """
        Format historical weather data for display
        
        Args:
            weather_data: WeatherData object with historical data
            use_colors: Whether to use colors in output
            use_emoji: Whether to use emoji in output
            
        Returns:
            Formatted string with historical weather information
        """
        summary = weather_data.get_summary()
        if not summary:
            return "No historical weather data available"
        
        # Get emoji and color for condition
        condition = summary['condition']
        emoji = WEATHER_EMOJIS.get(condition, "🌡️") if use_emoji else ""
        color = WEATHER_COLORS.get(condition, "") if use_colors and COLORS_AVAILABLE else ""
        reset = Style.RESET_ALL if use_colors and COLORS_AVAILABLE else ""
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(summary['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format wind direction as cardinal direction
        wind_direction = summary.get('wind_direction', 0)
        cardinal_direction = WeatherFormatter._degrees_to_cardinal(wind_direction)
        
        # Format visibility in km
        visibility_km = summary.get('visibility', 0) / 1000 if summary.get('visibility') else 0
        
        # Build output
        output = []
        output.append("=" * 60)
        output.append(f"{color}{emoji} Historical Weather for {summary['coordinates']} (as of {timestamp}){reset}")
        output.append("=" * 60)
        output.append(f"{color}Condition: {condition} ({summary.get('description', '')}){reset}")
        output.append(f"Temperature: {summary.get('temperature', 'N/A')}°C (Feels like: {summary.get('feels_like', 'N/A')}°C)")
        output.append(f"Humidity: {summary.get('humidity', 'N/A')}%")
        output.append(f"Wind: {summary.get('wind_speed', 'N/A')} m/s from {cardinal_direction} ({wind_direction}°)")
        output.append(f"Pressure: {summary.get('pressure', 'N/A')} hPa")
        output.append(f"Visibility: {visibility_km:.1f} km")
        output.append(f"Cloud Cover: {summary.get('clouds', 'N/A')}%")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    @staticmethod
    def _degrees_to_cardinal(degrees: float) -> str:
        """Convert degrees to cardinal direction"""
        directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]
        index = round(degrees / 22.5) % 16
        return directions[index]

class WeatherVisualizer:
    """Class for visualizing weather data"""
    
    @staticmethod
    def plot_forecast(weather_data: WeatherData, output_file: str = None) -> bool:
        """
        Plot forecast data
        
        Args:
            weather_data: WeatherData object with forecast
            output_file: Path to save the plot (if None, display instead)
            
        Returns:
            True if successful, False otherwise
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Please install matplotlib and numpy.")
            return False
        
        summary = weather_data.get_summary()
        if not summary or 'days' not in summary:
            print("No forecast data available for plotting")
            return False
        
        # Extract data for plotting
        dates = []
        min_temps = []
        max_temps = []
        conditions = []
        
        for day in summary['days']:
            date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
            dates.append(date_obj.strftime("%m-%d"))
            min_temps.append(day['min_temp'])
            max_temps.append(day['max_temp'])
            conditions.append(day['condition'])
        
        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot temperature range
        x = np.arange(len(dates))
        width = 0.35
        
        # Plot min and max temperatures as a range
        for i in range(len(dates)):
            ax1.plot([i, i], [min_temps[i], max_temps[i]], 'o-', linewidth=2, 
                    color=WeatherVisualizer._condition_to_color(conditions[i]))
        
        # Plot average temperature
        avg_temps = [(min_temp + max_temp) / 2 for min_temp, max_temp in zip(min_temps, max_temps)]
        ax1.plot(x, avg_temps, 'o-', linewidth=2, color='black', label='Avg Temp')
        
        # Add condition icons or text
        for i, condition in enumerate(conditions):
            ax1.annotate(WeatherVisualizer._condition_to_symbol(condition), 
                        (i, avg_temps[i]), 
                        textcoords="offset points",
                        xytext=(0, 10), 
                        ha='center')
        
        # Set labels and title
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title(f'Weather Forecast for {summary["location"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(dates)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax1.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display
        if output_file:
            plt.savefig(output_file)
            print(f"Forecast plot saved to {output_file}")
        else:
            plt.show()
        
        return True
    
    @staticmethod
    def _condition_to_color(condition: str) -> str:
        """Convert weather condition to color for plotting"""
        color_map = {
            "Clear": "gold",
            "Clouds": "lightblue",
            "Rain": "blue",
            "Drizzle": "skyblue",
            "Thunderstorm": "purple",
            "Snow": "lightgray",
            "Mist": "gray",
            "Fog": "gray",
            "Haze": "gray",
            "Smoke": "darkgray",
            "Dust": "tan",
            "Sand": "tan",
            "Ash": "darkgray",
            "Squall": "gray",
            "Tornado": "red"
        }
        return color_map.get(condition, "black")
    
    @staticmethod
    def _condition_to_symbol(condition: str) -> str:
        """Convert weather condition to symbol for plotting"""
        return WEATHER_EMOJIS.get(condition, "?")

def get_current_weather(api_key, location, units="metric"):
    """
    Get current weather for a location using OpenWeatherMap API
    
    Args:
        api_key: OpenWeatherMap API key
        location: Location name or coordinates (lat,lon)
        units: Units of measurement (metric, imperial, standard)
        
    Returns:
        Dictionary with weather information or None if request failed
    """
    api = OpenWeatherMapAPI(api_key)
    data = api.get_current_weather(location, units)
    if not data:
        return None
    
    weather_data = WeatherData(data, "current")
    return weather_data

def get_forecast(api_key, location, days=5, units="metric"):
    """
    Get weather forecast for a location
    
    Args:
        api_key: OpenWeatherMap API key
        location: Location name or coordinates (lat,lon)
        days: Number of days for forecast (max 5 for free tier)
        units: Units of measurement (metric, imperial, standard)
        
    Returns:
        Dictionary with forecast information or None if request failed
    """
    api = OpenWeatherMapAPI(api_key)
    data = api.get_forecast(location, days, units)
    if not data:
        return None
    
    weather_data = WeatherData(data, "forecast")
    return weather_data

def get_historical_weather(api_key, location, date, units="metric"):
    """
    Get historical weather for a location on a specific date
    
    Args:
        api_key: OpenWeatherMap API key
        location: Location name or coordinates (lat,lon)
        date: Date for historical weather (datetime object)
        units: Units of measurement (metric, imperial, standard)
        
    Returns:
        Dictionary with historical weather information or None if request failed
    """
    api = OpenWeatherMapAPI(api_key)
    data = api.get_historical_weather(location, date, units)
    if not data:
        return None
    
    weather_data = WeatherData(data, "historical")
    return weather_data

def display_weather(weather_data, use_colors=True, use_emoji=True):
    """
    Display weather information in a formatted way
    
    Args:
        weather_data: WeatherData object
        use_colors: Whether to use colors in output
        use_emoji: Whether to use emoji in output
    """
    if not weather_data or not weather_data.parsed_data:
        print("No weather data available to display")
        return
    
    if weather_data.data_type == "current":
        formatted = WeatherFormatter.format_current_weather(weather_data, use_colors, use_emoji)
    elif weather_data.data_type == "forecast":
        formatted = WeatherFormatter.format_forecast(weather_data, use_colors, use_emoji)
    elif weather_data.data_type == "historical":
        formatted = WeatherFormatter.format_historical(weather_data, use_colors, use_emoji)
    else:
        formatted = "Unknown weather data type"
    
    print(formatted)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Advanced Weather Script - Get current, forecast, and historical weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python weather_script.py current "New York"
          python weather_script.py forecast "London" --days 3
          python weather_script.py historical "Paris" --date 2023-01-01
          python weather_script.py current "Tokyo" --units imperial --no-color --no-emoji
          python weather_script.py forecast "Berlin" --plot forecast.png
        ''')
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Current weather command
    current_parser = subparsers.add_parser("current", help="Get current weather")
    current_parser.add_argument("location", help="Location name or coordinates (lat,lon)")
    current_parser.add_argument("--units", choices=["metric", "imperial", "standard"], 
                              default="metric", help="Units of measurement")
    
    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Get weather forecast")
    forecast_parser.add_argument("location", help="Location name or coordinates (lat,lon)")
    forecast_parser.add_argument("--days", type=int, default=5, 
                               help="Number of days for forecast (max 5 for free tier)")
    forecast_parser.add_argument("--units", choices=["metric", "imperial", "standard"], 
                               default="metric", help="Units of measurement")
    forecast_parser.add_argument("--plot", metavar="FILE", help="Save forecast plot to file")
    
    # Historical weather command
    historical_parser = subparsers.add_parser("historical", help="Get historical weather")
    historical_parser.add_argument("location", help="Location name or coordinates (lat,lon)")
    historical_parser.add_argument("--date", required=True, help="Date for historical weather (YYYY-MM-DD)")
    historical_parser.add_argument("--units", choices=["metric", "imperial", "standard"], 
                                 default="metric", help="Units of measurement")
    
    # Global options
    for subparser in [current_parser, forecast_parser, historical_parser]:
        subparser.add_argument("--no-color", action="store_true", help="Disable colored output")
        subparser.add_argument("--no-emoji", action="store_true", help="Disable emoji in output")
    
    # Add default command (for backward compatibility)
    parser.add_argument("--location", help="Location for backward compatibility")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    
    if not api_key:
        print("Error: OPENWEATHERMAP_API_KEY environment variable not set")
        print("Please set your OpenWeatherMap API key as an environment variable")
        print("Example: export OPENWEATHERMAP_API_KEY=your_api_key_here")
        return 1
    
    # Handle backward compatibility
    if not args.command and args.location:
        args.command = "current"
        args.location = args.location
        args.units = "metric"
        args.no_color = False
        args.no_emoji = False
    elif not args.command and not args.location:
        # Default behavior
        args.command = "current"
        args.location = "Bermuda"  # Default city
        args.units = "metric"
        args.no_color = False
        args.no_emoji = False
    
    # Execute command
    use_colors = not args.no_color
    use_emoji = not args.no_emoji
    
    if args.command == "current":
        weather_data = get_current_weather(api_key, args.location, args.units)
        if weather_data:
            display_weather(weather_data, use_colors, use_emoji)
        else:
            print(f"Could not retrieve weather information for {args.location}")
            return 1
    
    elif args.command == "forecast":
        weather_data = get_forecast(api_key, args.location, args.days, args.units)
        if weather_data:
            display_weather(weather_data, use_colors, use_emoji)
            
            # Plot forecast if requested
            if args.plot and PLOTTING_AVAILABLE:
                WeatherVisualizer.plot_forecast(weather_data, args.plot)
            elif args.plot and not PLOTTING_AVAILABLE:
                print("Warning: Plotting not available. Please install matplotlib and numpy.")
        else:
            print(f"Could not retrieve forecast information for {args.location}")
            return 1
    
    elif args.command == "historical":
        try:
            date = datetime.strptime(args.date, "%Y-%m-%d")
            weather_data = get_historical_weather(api_key, args.location, date, args.units)
            if weather_data:
                display_weather(weather_data, use_colors, use_emoji)
            else:
                print(f"Could not retrieve historical weather information for {args.location} on {args.date}")
                return 1
        except ValueError:
            print("Error: Invalid date format. Please use YYYY-MM-DD")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

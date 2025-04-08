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
    def plot_forecast(weather_data: WeatherData, output_file: str = None, plot_type: str = "temperature") -> bool:
        """
        Plot forecast data with multiple visualization options
        
        Args:
            weather_data: WeatherData object with forecast
            output_file: Path to save the plot (if None, display instead)
            plot_type: Type of plot to generate ("temperature", "precipitation", "humidity", "wind", "combined")
            
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
        humidities = []
        wind_speeds = []
        precip_probs = []
        
        for day in summary['days']:
            date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
            dates.append(date_obj.strftime("%m-%d"))
            min_temps.append(day['min_temp'])
            max_temps.append(day['max_temp'])
            conditions.append(day['condition'])
            humidities.append(day['avg_humidity'])
            wind_speeds.append(day['avg_wind_speed'])
            precip_probs.append(day['precipitation_probability'] * 100)  # Convert to percentage
        
        # Create figure with appropriate size
        if plot_type == "combined":
            fig = plt.figure(figsize=(14, 10))
            grid = plt.GridSpec(3, 2, hspace=0.3, wspace=0.3)
            
            # Temperature subplot
            ax1 = fig.add_subplot(grid[0, :])
            WeatherVisualizer._plot_temperature(ax1, dates, min_temps, max_temps, conditions, summary["location"])
            
            # Precipitation subplot
            ax2 = fig.add_subplot(grid[1, 0])
            WeatherVisualizer._plot_precipitation(ax2, dates, precip_probs, conditions)
            
            # Humidity subplot
            ax3 = fig.add_subplot(grid[1, 1])
            WeatherVisualizer._plot_humidity(ax3, dates, humidities)
            
            # Wind subplot
            ax4 = fig.add_subplot(grid[2, 0])
            WeatherVisualizer._plot_wind(ax4, dates, wind_speeds)
            
            # Conditions summary subplot
            ax5 = fig.add_subplot(grid[2, 1])
            WeatherVisualizer._plot_conditions_summary(ax5, conditions)
            
            plt.suptitle(f'Weather Forecast for {summary["location"]}', fontsize=16)
            
        elif plot_type == "precipitation":
            fig, ax = plt.subplots(figsize=(10, 6))
            WeatherVisualizer._plot_precipitation(ax, dates, precip_probs, conditions)
            ax.set_title(f'Precipitation Forecast for {summary["location"]}')
            
        elif plot_type == "humidity":
            fig, ax = plt.subplots(figsize=(10, 6))
            WeatherVisualizer._plot_humidity(ax, dates, humidities)
            ax.set_title(f'Humidity Forecast for {summary["location"]}')
            
        elif plot_type == "wind":
            fig, ax = plt.subplots(figsize=(10, 6))
            WeatherVisualizer._plot_wind(ax, dates, wind_speeds)
            ax.set_title(f'Wind Speed Forecast for {summary["location"]}')
            
        else:  # Default to temperature
            fig, ax = plt.subplots(figsize=(10, 6))
            WeatherVisualizer._plot_temperature(ax, dates, min_temps, max_temps, conditions, summary["location"])
        
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
    def _plot_temperature(ax, dates, min_temps, max_temps, conditions, location_name):
        """Helper method to plot temperature data"""
        x = np.arange(len(dates))
        
        # Plot min and max temperatures as a range
        for i in range(len(dates)):
            ax.plot([i, i], [min_temps[i], max_temps[i]], 'o-', linewidth=2, 
                   color=WeatherVisualizer._condition_to_color(conditions[i]))
        
        # Plot average temperature
        avg_temps = [(min_temp + max_temp) / 2 for min_temp, max_temp in zip(min_temps, max_temps)]
        ax.plot(x, avg_temps, 'o-', linewidth=2, color='black', label='Avg Temp')
        
        # Add condition icons or text
        for i, condition in enumerate(conditions):
            ax.annotate(WeatherVisualizer._condition_to_symbol(condition), 
                       (i, avg_temps[i]), 
                       textcoords="offset points",
                       xytext=(0, 10), 
                       ha='center',
                       fontsize=12)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature Forecast for {location_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(dates)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Add temperature range annotation
        min_all = min(min_temps)
        max_all = max(max_temps)
        ax.annotate(f"Range: {min_all:.1f}°C - {max_all:.1f}°C", 
                   xy=(0.02, 0.02), 
                   xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    @staticmethod
    def _plot_precipitation(ax, dates, precip_probs, conditions):
        """Helper method to plot precipitation probability"""
        x = np.arange(len(dates))
        
        # Create bars with colors based on conditions
        bars = ax.bar(x, precip_probs, width=0.6, alpha=0.7)
        
        # Color bars based on probability
        for i, bar in enumerate(bars):
            if precip_probs[i] > 70:
                bar.set_color('darkblue')
            elif precip_probs[i] > 40:
                bar.set_color('royalblue')
            elif precip_probs[i] > 20:
                bar.set_color('skyblue')
            else:
                bar.set_color('lightblue')
        
        # Add condition symbols
        for i, condition in enumerate(conditions):
            if "Rain" in condition or "Drizzle" in condition or "Thunderstorm" in condition:
                ax.annotate(WeatherVisualizer._condition_to_symbol(condition), 
                           (i, precip_probs[i] + 5), 
                           ha='center',
                           fontsize=12)
        
        # Set labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Precipitation Probability (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(dates)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add threshold lines
        ax.axhline(y=20, color='lightblue', linestyle='--', alpha=0.5)
        ax.axhline(y=50, color='royalblue', linestyle='--', alpha=0.5)
        ax.axhline(y=80, color='darkblue', linestyle='--', alpha=0.5)
    
    @staticmethod
    def _plot_humidity(ax, dates, humidities):
        """Helper method to plot humidity data"""
        x = np.arange(len(dates))
        
        # Create line plot with gradient fill
        line = ax.plot(x, humidities, 'o-', linewidth=2, color='teal')[0]
        
        # Add gradient fill
        ax.fill_between(x, 0, humidities, alpha=0.3, color='teal')
        
        # Add data labels
        for i, humidity in enumerate(humidities):
            ax.annotate(f"{humidity:.0f}%", 
                       (i, humidity + 2), 
                       ha='center',
                       fontsize=9)
        
        # Set labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Relative Humidity (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(dates)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add comfort zones
        ax.axhspan(40, 60, alpha=0.2, color='green', label='Comfort Zone')
        ax.legend()
    
    @staticmethod
    def _plot_wind(ax, dates, wind_speeds):
        """Helper method to plot wind speed data"""
        x = np.arange(len(dates))
        
        # Create wind speed visualization with arrows
        for i, speed in enumerate(wind_speeds):
            # Scale arrow size based on wind speed
            scale = min(1.0, speed / 10.0) * 0.8
            ax.arrow(i, 0, 0, speed, head_width=0.3, head_length=min(1.0, speed * 0.2), 
                    fc='darkblue', ec='darkblue', alpha=0.7,
                    length_includes_head=True, width=0.1)
        
        # Add data labels
        for i, speed in enumerate(wind_speeds):
            ax.annotate(f"{speed:.1f} m/s", 
                       (i, speed + 0.5), 
                       ha='center',
                       fontsize=9)
        
        # Set labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Wind Speed (m/s)')
        ax.set_xticks(x)
        ax.set_xticklabels(dates)
        ax.set_ylim(0, max(wind_speeds) * 1.3)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add wind speed categories
        categories = [
            (0, 0.5, "Calm", "lightgray"),
            (0.5, 3.3, "Light", "lightblue"),
            (3.3, 7.9, "Moderate", "skyblue"),
            (7.9, 13.8, "Fresh", "royalblue"),
            (13.8, 20.7, "Strong", "darkblue")
        ]
        
        # Add colored bands for wind categories
        for start, end, label, color in categories:
            if start < ax.get_ylim()[1]:
                end_val = min(end, ax.get_ylim()[1])
                ax.axhspan(start, end_val, alpha=0.1, color=color)
                # Add label at the right side
                if end_val > start:
                    ax.annotate(label, 
                               xy=(len(dates) - 0.5, (start + end_val) / 2),
                               ha='right',
                               fontsize=8,
                               color='darkblue')
    
    @staticmethod
    def _plot_conditions_summary(ax, conditions):
        """Helper method to plot conditions summary"""
        # Count occurrences of each condition
        condition_counts = {}
        for condition in conditions:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        # Sort by count
        sorted_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Extract data for pie chart
        labels = [cond for cond, count in sorted_conditions]
        sizes = [count for cond, count in sorted_conditions]
        colors = [WeatherVisualizer._condition_to_color(cond) for cond, count in sorted_conditions]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Customize text properties
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
        
        # Add legend with condition symbols
        legend_labels = [f"{cond} {WeatherVisualizer._condition_to_symbol(cond)}" for cond in labels]
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax.set_title('Weather Conditions Distribution')
        ax.set_aspect('equal')
    
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
        
    @staticmethod
    def generate_weather_report(weather_data: WeatherData, output_file: str = None, include_plots: bool = True) -> str:
        """
        Generate a comprehensive weather report with text and optional plots
        
        Args:
            weather_data: WeatherData object
            output_file: Path to save the report (if None, returns as string)
            include_plots: Whether to include plots in the report
            
        Returns:
            Report content as string if output_file is None, otherwise path to saved file
        """
        if not weather_data or not weather_data.parsed_data:
            return "No weather data available for report generation"
        
        report = []
        
        # Add header
        report.append("=" * 80)
        report.append("COMPREHENSIVE WEATHER REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Format based on data type
        if weather_data.data_type == "current":
            summary = weather_data.get_summary()
            
            # Current conditions section
            report.append("CURRENT CONDITIONS")
            report.append("-" * 80)
            report.append(f"Location: {summary['location']}")
            report.append(f"Time of Observation: {datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Weather: {summary['condition']} ({summary['description']})")
            report.append(f"Temperature: {summary['temperature']}°C (Feels like: {summary['feels_like']}°C)")
            report.append("")
            
            # Detailed measurements section
            report.append("DETAILED MEASUREMENTS")
            report.append("-" * 80)
            report.append(f"Humidity: {summary['humidity']}%")
            report.append(f"Wind: {summary['wind_speed']} m/s from {WeatherFormatter._degrees_to_cardinal(summary['wind_direction'])} ({summary['wind_direction']}°)")
            report.append(f"Pressure: {summary['pressure']} hPa")
            report.append(f"Visibility: {summary['visibility'] / 1000:.1f} km")
            report.append(f"Cloud Cover: {summary['clouds']}%")
            
            # Precipitation data if available
            if summary['rain_1h'] > 0 or summary['snow_1h'] > 0:
                report.append("")
                report.append("PRECIPITATION")
                report.append("-" * 80)
                if summary['rain_1h'] > 0:
                    report.append(f"Rain (last hour): {summary['rain_1h']} mm")
                if summary['snow_1h'] > 0:
                    report.append(f"Snow (last hour): {summary['snow_1h']} mm")
            
            # Sun information
            report.append("")
            report.append("SUN INFORMATION")
            report.append("-" * 80)
            sunrise = datetime.fromtimestamp(summary['sunrise']).strftime("%H:%M:%S")
            sunset = datetime.fromtimestamp(summary['sunset']).strftime("%H:%M:%S")
            day_length = datetime.fromtimestamp(summary['sunset']) - datetime.fromtimestamp(summary['sunrise'])
            day_length_hours = day_length.total_seconds() / 3600
            
            report.append(f"Sunrise: {sunrise}")
            report.append(f"Sunset: {sunset}")
            report.append(f"Day Length: {day_length_hours:.1f} hours")
            
        elif weather_data.data_type == "forecast":
            summary = weather_data.get_summary()
            
            # Forecast overview
            report.append("FORECAST OVERVIEW")
            report.append("-" * 80)
            report.append(f"Location: {summary['location']}")
            report.append(f"Forecast Period: {summary['days'][0]['date']} to {summary['days'][-1]['date']}")
            report.append(f"Number of Days: {summary['days_count']}")
            report.append("")
            
            # Daily forecasts
            report.append("DAILY FORECASTS")
            report.append("-" * 80)
            
            for day in summary['days']:
                date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
                date_str = date_obj.strftime("%A, %B %d, %Y")  # e.g., "Monday, January 01, 2023"
                
                report.append(f"\n{date_str}")
                report.append(f"Weather: {day['condition']}")
                report.append(f"Temperature: {day['min_temp']:.1f}°C to {day['max_temp']:.1f}°C")
                report.append(f"Humidity: {day['avg_humidity']:.0f}%")
                report.append(f"Wind: {day['avg_wind_speed']:.1f} m/s")
                report.append(f"Precipitation Chance: {day['precipitation_probability'] * 100:.0f}%")
            
            # Weather trends analysis
            report.append("\nWEATHER TRENDS ANALYSIS")
            report.append("-" * 80)
            
            # Temperature trend
            temp_trend = []
            for i in range(1, len(summary['days'])):
                prev_avg = (summary['days'][i-1]['min_temp'] + summary['days'][i-1]['max_temp']) / 2
                curr_avg = (summary['days'][i]['min_temp'] + summary['days'][i]['max_temp']) / 2
                diff = curr_avg - prev_avg
                if diff > 2:
                    temp_trend.append(f"Significant warming on {summary['days'][i]['date']} (+{diff:.1f}°C)")
                elif diff < -2:
                    temp_trend.append(f"Significant cooling on {summary['days'][i]['date']} ({diff:.1f}°C)")
            
            if temp_trend:
                report.append("Temperature Changes:")
                for trend in temp_trend:
                    report.append(f"- {trend}")
            else:
                report.append("Temperature: Relatively stable throughout the forecast period")
            
            # Precipitation trend
            precip_days = [day for day in summary['days'] if day['precipitation_probability'] > 0.3]
            if precip_days:
                report.append("\nPrecipitation Expected:")
                for day in precip_days:
                    date_obj = datetime.strptime(day['date'], "%Y-%m-%d")
                    date_str = date_obj.strftime("%A, %B %d")
                    report.append(f"- {date_str}: {day['precipitation_probability'] * 100:.0f}% chance")
            else:
                report.append("\nPrecipitation: No significant precipitation expected during the forecast period")
            
        elif weather_data.data_type == "historical":
            summary = weather_data.get_summary()
            
            # Historical data overview
            report.append("HISTORICAL WEATHER DATA")
            report.append("-" * 80)
            report.append(f"Location Coordinates: {summary['coordinates']}")
            report.append(f"Date: {datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d')}")
            report.append(f"Time of Observation: {datetime.fromtimestamp(summary['timestamp']).strftime('%H:%M:%S')}")
            report.append("")
            
            # Weather conditions
            report.append("WEATHER CONDITIONS")
            report.append("-" * 80)
            report.append(f"Weather: {summary['condition']} ({summary.get('description', '')})")
            report.append(f"Temperature: {summary.get('temperature', 'N/A')}°C (Feels like: {summary.get('feels_like', 'N/A')}°C)")
            report.append(f"Humidity: {summary.get('humidity', 'N/A')}%")
            report.append(f"Wind: {summary.get('wind_speed', 'N/A')} m/s from {WeatherFormatter._degrees_to_cardinal(summary.get('wind_direction', 0))} ({summary.get('wind_direction', 'N/A')}°)")
            report.append(f"Pressure: {summary.get('pressure', 'N/A')} hPa")
            report.append(f"Visibility: {summary.get('visibility', 0) / 1000:.1f} km")
            report.append(f"Cloud Cover: {summary.get('clouds', 'N/A')}%")
        
        # Add plots if requested and available
        if include_plots and PLOTTING_AVAILABLE:
            report.append("\nPLOTS")
            report.append("-" * 80)
            report.append("Plots are available but not included in text report.")
            report.append("Use the --plot option to generate visual representations of the data.")
        
        # Add footer
        report.append("")
        report.append("=" * 80)
        report.append(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # Compile the report
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            return output_file
        
        return report_text

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
          python weather_script.py forecast "Miami" --plot-type combined
          python weather_script.py report "Chicago" --output chicago_weather.txt
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
    forecast_parser.add_argument("--plot-type", choices=["temperature", "precipitation", "humidity", "wind", "combined"],
                               default="temperature", help="Type of plot to generate")
    
    # Historical weather command
    historical_parser = subparsers.add_parser("historical", help="Get historical weather")
    historical_parser.add_argument("location", help="Location name or coordinates (lat,lon)")
    historical_parser.add_argument("--date", required=True, help="Date for historical weather (YYYY-MM-DD)")
    historical_parser.add_argument("--units", choices=["metric", "imperial", "standard"], 
                                 default="metric", help="Units of measurement")
    
    # Comprehensive report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive weather report")
    report_parser.add_argument("location", help="Location name or coordinates (lat,lon)")
    report_parser.add_argument("--type", choices=["current", "forecast", "both"], 
                             default="both", help="Type of data to include in report")
    report_parser.add_argument("--days", type=int, default=5, 
                             help="Number of days for forecast (max 5 for free tier)")
    report_parser.add_argument("--units", choices=["metric", "imperial", "standard"], 
                             default="metric", help="Units of measurement")
    report_parser.add_argument("--output", metavar="FILE", help="Save report to file")
    report_parser.add_argument("--include-plots", action="store_true", help="Include plots in the report")
    
    # Compare locations command
    compare_parser = subparsers.add_parser("compare", help="Compare weather between multiple locations")
    compare_parser.add_argument("locations", nargs="+", help="List of locations to compare")
    compare_parser.add_argument("--units", choices=["metric", "imperial", "standard"], 
                              default="metric", help="Units of measurement")
    compare_parser.add_argument("--plot", metavar="FILE", help="Save comparison plot to file")
    
    # Global options
    for subparser in [current_parser, forecast_parser, historical_parser, report_parser, compare_parser]:
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
                WeatherVisualizer.plot_forecast(weather_data, args.plot, args.plot_type)
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
            
    elif args.command == "report":
        # Generate comprehensive report
        if args.type == "both" or args.type == "current":
            current_data = get_current_weather(api_key, args.location, args.units)
        else:
            current_data = None
            
        if args.type == "both" or args.type == "forecast":
            forecast_data = get_forecast(api_key, args.location, args.days, args.units)
        else:
            forecast_data = None
            
        if not current_data and not forecast_data:
            print(f"Could not retrieve weather information for {args.location}")
            return 1
            
        # Use the data that's available
        report_data = current_data if current_data else forecast_data
        
        # Generate the report
        report = WeatherVisualizer.generate_weather_report(report_data, args.output, args.include_plots)
        
        if args.output:
            print(f"Weather report saved to {args.output}")
        else:
            print(report)
            
    elif args.command == "compare":
        # Compare weather between multiple locations
        if len(args.locations) < 2:
            print("Error: At least two locations are required for comparison")
            return 1
            
        print(f"Comparing weather for {len(args.locations)} locations...")
        
        # Get current weather for all locations
        weather_data_list = []
        for location in args.locations:
            data = get_current_weather(api_key, location, args.units)
            if data:
                weather_data_list.append(data)
            else:
                print(f"Warning: Could not retrieve weather for {location}")
                
        if not weather_data_list:
            print("Error: Could not retrieve weather for any of the specified locations")
            return 1
            
        # Display comparison table
        print("\n" + "=" * 80)
        print("WEATHER COMPARISON")
        print("=" * 80)
        
        # Print header
        print(f"{'Location':<20} {'Condition':<15} {'Temp (°C)':<10} {'Feels Like':<10} {'Humidity':<10} {'Wind (m/s)':<10}")
        print("-" * 80)
        
        # Print data rows
        for data in weather_data_list:
            summary = data.get_summary()
            location_name = summary['location'].split(',')[0]  # Just city name
            print(f"{location_name:<20} {summary['condition']:<15} {summary['temperature']:<10.1f} {summary['feels_like']:<10.1f} {summary['humidity']:<10} {summary['wind_speed']:<10.1f}")
            
        print("=" * 80)
        
        # Generate comparison plot if requested
        if args.plot and PLOTTING_AVAILABLE:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Extract data for plotting
            locations = [data.get_summary()['location'].split(',')[0] for data in weather_data_list]
            temps = [data.get_summary()['temperature'] for data in weather_data_list]
            feels_like = [data.get_summary()['feels_like'] for data in weather_data_list]
            humidities = [data.get_summary()['humidity'] for data in weather_data_list]
            
            # Temperature comparison
            x = np.arange(len(locations))
            width = 0.35
            
            ax1.bar(x - width/2, temps, width, label='Actual Temp')
            ax1.bar(x + width/2, feels_like, width, label='Feels Like')
            
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title('Temperature Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(locations)
            ax1.legend()
            
            # Humidity comparison
            ax2.bar(x, humidities, color='skyblue')
            
            ax2.set_ylabel('Humidity (%)')
            ax2.set_title('Humidity Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(locations)
            
            plt.tight_layout()
            plt.savefig(args.plot)
            print(f"Comparison plot saved to {args.plot}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

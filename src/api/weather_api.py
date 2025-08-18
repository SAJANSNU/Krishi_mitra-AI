"""
weather_api.py

Dynamic Weather API Module for Agri Assistant

Supports OpenWeatherMap as primary provider with easy extensibility,
graceful error handling, clear structure and no city/state presets.

"""

import os
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.settings import OPENWEATHER_API_KEY
from src.utils.logger import log

class WeatherAPI:
    def __init__(self):
        # You may add support for more providers here
        self.api_key = os.getenv("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY)
        self.owm_url_current = "https://api.openweathermap.org/data/2.5/weather"
        self.owm_url_forecast = "https://api.openweathermap.org/data/2.5/forecast"
        self.session = requests.Session()
        log.info("WeatherAPI initialized")

    def get_current_weather(self, city: str, country: str = "IN") -> Dict[str, Any]:
        """
        Fetches current weather for a city/country using OpenWeatherMap.
        Returns canonical structure with keys:
        ['temperature', 'humidity', 'description', 'wind_speed', 'pressure', ...]
        """
        result = {"city": city, "country": country, "source": "openweathermap"}
        try:
            params = {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "metric"
            }
            response = self.session.get(self.owm_url_current, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            weather = data.get("weather", [{}])[0]
            main = data.get("main", {})
            wind = data.get("wind", {})
            result.update({
                "temperature": main.get("temp"),
                "humidity": main.get("humidity"),
                "description": weather.get("description"),
                "wind_speed": wind.get("speed"),
                "pressure": main.get("pressure"),
                "feels_like": main.get("feels_like"),
                "icon": weather.get("icon"),
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
        except Exception as e:
            log.error(f"get_current_weather failed for {city},{country}: {e}")
            result.update({
                "error": str(e),
                "success": False
            })
        return result

    def get_forecast(self, city: str, country: str = "IN", days: int = 5) -> Dict[str, Any]:
        """
        Fetches weather forecast for a city for specified number of days.
        Returns list of forecasts (up to 40 3-hour slots from OWM).
        """
        result = {"city": city, "country": country, "source": "openweathermap"}
        try:
            params = {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "metric"
            }
            response = self.session.get(self.owm_url_forecast, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            slots = data.get("list", [])
            forecast: List[Dict[str, Any]] = []
            for entry in slots:
                dt = entry.get("dt", 0)
                dt_iso = datetime.fromtimestamp(dt).isoformat() if dt else None
                main = entry.get("main", {})
                weather = entry.get("weather", [{}])[0]
                wind = entry.get("wind", {})
                forecast.append({
                    "datetime": dt_iso,
                    "temperature": main.get("temp"),
                    "humidity": main.get("humidity"),
                    "description": weather.get("description"),
                    "icon": weather.get("icon"),
                    "wind_speed": wind.get("speed"),
                    "pressure": main.get("pressure"),
                })
                if len(forecast) >= days * 8:
                    # OWM gives 8 slots per day (3-hourly), so days*8 slots
                    break
            result.update({
                "forecast": forecast,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            log.error(f"get_forecast failed for {city},{country}: {e}")
            result.update({
                "forecast": [],
                "error": str(e),
                "success": False
            })
        return result

    def get_weather_bundle(self, city: str, country: str = "IN", forecast_days: int = 5) -> Dict[str, Any]:
        """
        Convenience: Returns both current weather and forecast in one call.
        """
        return {
            "current": self.get_current_weather(city, country),
            "forecast": self.get_forecast(city, country, forecast_days)
        }

    def healthcheck(self) -> Dict[str, Any]:
        """
        Check if API is working using a simple ping for a major city.
        """
        check_city = "Delhi"
        status = {"openweathermap": False}
        try:
            data = self.get_current_weather(check_city)
            if data.get("success"):
                status["openweathermap"] = True
        except Exception as e:
            status["error"] = str(e)
        return status

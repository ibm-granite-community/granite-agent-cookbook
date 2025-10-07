from datetime import datetime
from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from ibm_granite_community.notebook_utils import get_env_var
from langchain_core.tools import tool
from langchain_core.utils.utils import convert_to_secret_str
from tools import ROOT_DIR
from typing_extensions import List

WEATHER_API_KEY = convert_to_secret_str(get_env_var("WEATHER_API_KEY", "unset"))


@tool(parse_docstring=True)
def get_current_weather(location: str) -> dict:
    """
    Fetches the current weather for a given location (default: San Francisco).

    Args:
        location: The name of the city for which to retrieve the weather information.

    Returns:
        A dictionary containing weather information such as temperature in celsius, weather description, and humidity.
    """
    print(f"Getting current weather for {location}")
    apikey = WEATHER_API_KEY.get_secret_value()
    if apikey == "unset":
        print("No API key present; using a fixed, predetermined value for demonstration purposes")
        return {"description": "thunderstorms", "temperature": 25.3, "humidity": 94}

    try:
        # API request to fetch weather data
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={apikey}&units=metric"
        weather_data = requests.get(weather_url)
        data = weather_data.json()
        # Extracting relevant weather details
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        # Returning weather details
        return {"description": weather_description, "temperature": temperature, "humidity": humidity}
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {"description": "none", "temperature": "none", "humidity": "none"}


def _get_geo_coordinates_internal(city_name: str, state_code: str, country: str) -> Tuple[float, float]:
    """
    Internal function to get geographic coordinates (not a LangChain tool).
    """
    print(f"Getting geo coordinates data for {city_name} {state_code} {country}")
    apikey = WEATHER_API_KEY.get_secret_value()
    if apikey == "unset":
        print("No API key present; using a fixed, predetermined value for demonstration purposes")
        return 37.7790262, -122.419906

    try:
        # API request to fetch geo data
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},{state_code},{country}&limit=5&appid={apikey}"
        geo_data = requests.get(geo_url)
        data = geo_data.json()

        return data[0].get("lat", 37.7790262), data[0].get("lon", -122.419906)
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return 37.7790262, -122.419906


@tool(parse_docstring=True)
def get_geo_coordinates(city_name: str, state_code: str, country: str) -> Tuple[float, float]:
    """
    Retrieves geographic coordinates (latitude and longitude) for a specified city.

    This function converts city names into precise geographic coordinates that can be used
    for weather API calls and other location-based services. It uses the OpenWeatherMap
    Geocoding API to resolve city names to coordinates.

    Args:
        city_name (str): The name of the city. Examples "New York", "Montréal", "London"
        state_code (str): The state or province code. Examples "NY", "CA", "Québec", "ON"
        country (str): The two-letter country code. Examples "US", "CA", "GB", "FR"

    Returns:
        Tuple[float, float]: A tuple containing (latitude, longitude) coordinates.
            - Latitude: Decimal degrees between -90 and 90 (negative = South)
            - Longitude: Decimal degrees between -180 and 180 (negative = West)

        Example returns:
            (40.7128, -74.0060)  # New York City
            (45.5017, -73.5673)  # Montréal
            (37.7749, -122.4194) # San Francisco (fallback)

    Raises:
        Exception: If API request fails, returns fallback coordinates for San Francisco

    Note:
        - Returns the first matching result from the geocoding API
        - If no API key is configured, returns San Francisco coordinates as fallback
        - State codes should use standard abbreviations (e.g., "CA" for California)
        - Country codes should use ISO 3166-1 alpha-2 format
    """
    return _get_geo_coordinates_internal(city_name, state_code, country)


def _get_weather_forecast_internal(lat: float, lon: float) -> list:
    """
    Internal function to get weather forecast (not a LangChain tool).
    """
    print(f"Getting weather forecast for {lat} {lon} ")
    apikey = WEATHER_API_KEY.get_secret_value()
    if apikey == "unset":
        print("No API key present; using a fixed, predetermined value for demonstration purposes")
        return [{"2025-10-04 12:00:00": 25.3}]

    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={apikey}&units=metric"
        weather_data = requests.get(weather_url)
        data = weather_data.json()

        # Transform the data to the required format
        weather_list = data.get("list", [])
        formatted_data = []

        for item in weather_list:
            dt = item.get("dt", "")
            temperature = item.get("main", {}).get("temp", 0)

            if dt and temperature is not None:
                dt_datetime = datetime.fromtimestamp(dt)
                dt_string = dt_datetime.strftime("%Y-%m-%d %H:%M:%S")
                formatted_data.append({dt_string: temperature})
        print(f"{len(formatted_data)} forecast datapoints fetched:")
        return formatted_data
    except Exception as e:
        raise e


@tool(parse_docstring=True)
def get_weather_forecast(lat: float, lon: float) -> list:
    """
    Retrieves a 5-day weather forecast for a specific location and date.

    This function fetches detailed weather forecast data for the next 5 days starting from
    the specified datetime string. The forecast includes 3-hourly intervals with temperature data
    formatted for easy consumption by AI agents and applications.

    Args:
        lat: Latitude coordinate in decimal degrees. Range: -90 to 90 Examples 40.7128 (New York), 45.5017 (Montréal)
        lon: Longitude coordinate in decimal degrees. Range: -180 to 180 Examples -74.0060 (New York), -73.5673 (Montréal)

    Returns:
        list: A list of dictionaries, each containing a datetime-temperature pair.
              Format: [{"YYYY-MM-DD HH:MM:SS": temperature_in_celsius}]

        Example return:
            [
                {"2025-10-04 00:00:00": 9.93},
                {"2025-10-04 03:00:00": 10.42},
                {"2025-10-04 06:00:00": 11.96},
                {"2025-10-04 09:00:00": 12.63},
                # ... continues for 5 days at 3-hour intervals
            ]

    Raises:
        ValueError: If start_datetime is not in the correct format
        Exception: If API request fails, raises the original exception

    Note:
        - Temperature values are in Celsius
        - Datetime strings are in UTC timezone
        - Forecast covers 5 days with 3-hourly intervals (40 data points total)
        - If no API key is configured, returns demo data for demonstration
        - The start_datetime parameter determines the starting point of the forecast
        - Use get_geo_coordinates() to convert city names to lat/lon coordinates
    """
    return _get_weather_forecast_internal(lat, lon)


def _plot_weather_timeseries_internal(
    weather_data: List[Dict[str, float]], title: str = "Weather Forecast", save_path: str = None
) -> None:
    """
    Internal function to plot weather timeseries (not a LangChain tool).
    """
    if not weather_data:
        raise ValueError("Weather data cannot be empty")

    dataframes = []
    all_series_labels = []

    for series_name, series_list in weather_data.items():
        if not series_list:
            continue

        datetimes = []
        temperatures = []

        for item in series_list:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError("Each item in weather_data must be a dictionary with exactly one key-value pair")

            dt_str = list(item.keys())[0]
            temp = list(item.values())[0]

            try:
                dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                datetimes.append(dt_obj)
                temperatures.append(temp)
            except ValueError as e:
                raise ValueError(f"Invalid datetime format in data: {dt_str}. Expected 'YYYY-MM-DD HH:MM:SS'") from e

        if datetimes:
            df = pd.DataFrame({"datetime": datetimes, series_name: temperatures})
            df.set_index("datetime", inplace=True)
            dataframes.append(df)
            all_series_labels.append(series_name)

    if not dataframes:
        raise ValueError("No valid datetime-temperature pairs found in weather_data")

    if len(dataframes) == 1:
        merged_df = dataframes[0]
    else:
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = merged_df.join(df, how="outer")

    plt.figure(figsize=(14, 8))

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#7209B7", "#2D5016"]
    markers = ["o", "s", "^", "D", "v", "p"]

    for i, (series_name, series_data) in enumerate(merged_df.items()):
        if series_data.dropna().empty:
            continue

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(
            series_data.index,
            series_data.values,
            marker=marker,
            linewidth=2,
            markersize=6,
            color=color,
            label=series_name,
            alpha=0.8,
        )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Date and Time", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=45)

    if len(merged_df.columns) > 1:
        plt.legend(loc="best", framealpha=0.9)

    all_temps = merged_df.values.flatten()
    all_temps = all_temps[~np.isnan(all_temps)]
    if len(all_temps) > 0:
        min_temp = np.min(all_temps)
        max_temp = np.max(all_temps)
        plt.text(
            0.02,
            0.98,
            f"Range: {min_temp:.1f}°C - {max_temp:.1f}°C",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


@tool(parse_docstring=True)
def plot_weather_timeseries(weather_data: Dict[str, List[Dict[str, float]]], title: str = "Weather Forecast") -> str:
    """
    Creates a time series plot from weather forecast data, supporting multiple series.

    This function can multiple forecasts on the same plot.
    It uses pandas to merge multiple time series by datetime, allowing comparison of different
    forecasts or locations.

    Args:
        weather_data: Dict[str, List[Dict[str, float]]]: Dictionary with city names as keys and lists of datetime-temperature pairs as values. Format: {'CityName': [{'YYYY-MM-DD HH:MM:SS': temperature}, ...]} Example: {'City1': [{'2025-10-05 17:00:00': 29.73}, {'2025-10-05 20:00:00': 27.28},.....],  'City2': [{'2025-10-05 20:00:00': 26.09}, {'2025-10-05 23:00:00': 21.3},....]}
        title: Title for the plot. Defaults to "Weather Forecast"

    Returns:
        str: Completion message

    Raises:
        ValueError: If weather_data is empty or has invalid format
        Exception: If plotting fails
    """  # noqa: E501
    _plot_weather_timeseries_internal(weather_data=weather_data, title=title, save_path=f"{ROOT_DIR}/images/plot.png")
    return f"Plot {title} was saved to f'{ROOT_DIR}/images/plot.png"


if __name__ == "__main__":
    cities = [("Montréal", "Québec", "CA"), ("Toronto", "Ontario", "CA"), ("Armonk", "NY", "US")]

    all_forecasts = {}
    for city, state, country_code in cities:
        lat, lon = _get_geo_coordinates_internal(city, state, country_code)
        forecast = _get_weather_forecast_internal(lat, lon)
        all_forecasts[city] = forecast

    _plot_weather_timeseries_internal(
        all_forecasts, title="Weather Forecast Comparison", save_path="cities_weather_comparison.png"
    )

import pandas as pd

from model.room_model import RaumModell
from data_module.weather_api import fetch_weather
from data_module.sunpos import sunpos
from data_module.RasPiDummy import TemperaturSensor, fetch_sensor_data
from data_module.data_module_static import DataModuleStatic

DataModule = DataModuleStatic()

DataModule.load_csv(r"data\training\240331_Dataset.csv")

# Zeitbereich abrufen
start, end = data_module.get_time_range()

print(f"Startzeit: {start}, Endzeit: {end}")
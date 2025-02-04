import pandas as pd

from model.room_model import RaumModell
from data_module.weather_api import fetch_weather
from data_module.sunpos import sunpos
from data_module.RasPiDummy import TemperaturSensor, fetch_sensor_data
from data_module.data_module import DatenModul

# Initialisiere Sensoren
sensoren = [TemperaturSensor(ad_bit=i) for i in range(2)]

def full_system_test():
    # Teste Wetterdatenabruf    
    api_key = "385eb69ad269f567adbebbee5b7c015d"
    lat, lon = 48.491576, 9.680011  # Koordinaten für München
    weather_data = fetch_weather(api_key, lat, lon)
    print("Wetterdaten:")
    print(weather_data.head())
    
    # Teste Sonnenpositionsberechnung
    timestamp = pd.Timestamp.utcnow()
    print(timestamp)
    azimuth, elevation = sunpos(timestamp, [lat, lon])
    print(f"Sonnenposition: Azimut={azimuth}, Elevation={elevation}")
    
    # Teste Sensor-Datenabruf
    sensor_data = fetch_sensor_data(sensoren)
    print("Sensordaten:")
    print(sensor_data.head())
    
    # Teste Datenmodul
    data_module = DatenModul()
    data_module.add_source("weather", lambda: fetch_weather(api_key, lat, lon))
    data_module.add_source("sensors", lambda: fetch_sensor_data(sensoren))
    data_module.add_source("sunpos", lambda: sunpos(pd.Timestamp.utcnow(), [lat, lon]))
    data_module.collect_data()
    print("Gesammelte Daten:")
    print(data_module.data.head())
    
    # Teste Raum-Modell
    raum = RaumModell()
    raum.raumtemperatur_model(data_module.data)
    print("Raumtemperatur-Simulation abgeschlossen.")

full_system_test()

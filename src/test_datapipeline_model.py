import pandas as pd
from room_model import RaumModell
from fetch_weather import fetch_weather
from sunpos import sunpos
from raspi_dummy import TemperaturSensor, fetch_sensor_data
from data_module import DatenModul

# Initialisiere Sensoren
sensoren = [TemperaturSensor(ad_bit=i) for i in range(2)]

def full_system_test():
    # Teste Wetterdatenabruf
    api_key = "your_api_key_here"
    lat, lon = 48.1351, 11.5820  # Koordinaten für München
    weather_data = fetch_weather(api_key, lat, lon)
    print("Wetterdaten:")
    print(weather_data.head())
    
    # Teste Sonnenpositionsberechnung
    timestamp = pd.Timestamp.utcnow()
    azimuth, elevation = sunpos(timestamp, lat, lon)
    print(f"Sonnenposition: Azimut={azimuth}, Elevation={elevation}")
    
    # Teste Sensor-Datenabruf
    sensor_data = fetch_sensor_data(sensoren)
    print("Sensordaten:")
    print(sensor_data.head())
    
    # Teste Datenmodul
    data_module = DatenModul()
    data_module.add_source("weather", lambda: fetch_weather(api_key, lat, lon))
    data_module.add_source("sensors", lambda: fetch_sensor_data(sensoren))
    data_module.add_source("sunpos", lambda: sunpos(pd.Timestamp.utcnow(), lat, lon))
    data_module.collect_data()
    print("Gesammelte Daten:")
    print(data_module.data.head())
    
    # Teste Raum-Modell
    raum = RaumModell()
    raum.raumtemperatur_model(data_module.data)
    print("Raumtemperatur-Simulation abgeschlossen.")

full_system_test()

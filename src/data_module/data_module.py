import pandas as pd
import json
from datetime import datetime

class DatenModul:
    def __init__(self, fetch_weather=None, fetch_sensors=None, sonnenwinkel_funktion=None, aufloesung='1T'):
        self.fetch_weather = fetch_weather
        self.fetch_sensors = fetch_sensors
        self.aufloesung = aufloesung
        self.df = pd.DataFrame()
    
    def lade_wetterdaten(self):
        wetter_df = self.fetch_weather()
        wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
        return wetter_df
    
    def lade_sensordaten(self):
        if not self.fetch_sensors:
            raise ValueError("Keine Sensordatei angegeben.")
        sensor_df = pd.read_csv(self.fetch_sensors)
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        return sensor_df
    
    def berechne_sonnenwinkel(self, timestamps, lat, lon):
        sonnenwinkel_daten = [self.sonnenwinkel_funktion(ts, lat, lon) for ts in timestamps]
        return pd.DataFrame(sonnenwinkel_daten, columns=['azimuth', 'elevation'], index=timestamps)
    
    def synchronisiere_daten(self, lat, lon):
        wetter_df = self.lade_wetterdaten()
        sensor_df = self.lade_sensordaten()
        
        gemeinsame_zeitachse = pd.date_range(
            start=min(wetter_df['timestamp'].min(), sensor_df['timestamp'].min()),
            end=max(wetter_df['timestamp'].max(), sensor_df['timestamp'].max()),
            freq=self.aufloesung
        )
        
        wetter_df = wetter_df.set_index('timestamp').reindex(gemeinsame_zeitachse).interpolate()
        sensor_df = sensor_df.set_index('timestamp').reindex(gemeinsame_zeitachse).interpolate()
        sonnenwinkel_df = self.berechne_sonnenwinkel(gemeinsame_zeitachse, lat, lon)
        
        self.df = pd.concat([wetter_df, sensor_df, sonnenwinkel_df], axis=1).reset_index()
        self.df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    def speichere_daten(self, dateipfad):
        self.df.to_csv(dateipfad, index=False)
    
    def lade_daten(self, dateipfad):
        self.df = pd.read_csv(dateipfad)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

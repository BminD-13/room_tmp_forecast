import pandas as pd
import json
import sunpos
from datetime import datetime

class DatenModul:
    def __init__(self, wetter_datei=None, sensor_datei=None, sonnenwinkel_funktion=None, aufloesung='1T'):
        self.wetter_datei = wetter_datei
        self.sensor_datei = sensor_datei
        self.sonnenwinkel_funktion = sonnenwinkel_funktion if sonnenwinkel_funktion else sunpos.calculate_sun_position
        self.aufloesung = aufloesung
        self.df = pd.DataFrame()
    
    def lade_wetterdaten(self):
        if not self.wetter_datei:
            raise ValueError("Keine Wetterdatei angegeben.")
        with open(self.wetter_datei, 'r') as file:
            daten = json.load(file)
        wetter_df = pd.DataFrame(daten)
        wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
        return wetter_df
    
    def lade_sensordaten(self):
        if not self.sensor_datei:
            raise ValueError("Keine Sensordatei angegeben.")
        sensor_df = pd.read_csv(self.sensor_datei)
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

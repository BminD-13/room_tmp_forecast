import pandas as pd
import json
from datetime import datetime

class DataModuleStatic:

    def __init__(self, aufloesung='1T'):
        self.aufloesung = aufloesung
        self.df = pd.DataFrame()
    
    def load_dataframe(self, df:pd.DataFrame):
        self.df = df
        self.check_df()

    def load_csv(self, path):
        if not path:
            raise ValueError("Keine .csv Datei angegeben.")
        self.df = pd.read_csv(path)
        self.check_df()

    def save_csv(self, dateipfad):
        self.df.to_csv(dateipfad, index=False)
    
    def check_df(self):
        pass

    def get_df(self):
        return self.df

    def synchronisiere_daten(self, lat, lon):

        gemeinsame_zeitachse = pd.date_range(
            start=self.df['timestamp'].min(),
            end=self.df['timestamp'].max(),
            freq=self.aufloesung
        )
        
        self.df = self.df.set_index('timestamp').reindex(gemeinsame_zeitachse).interpolate()
        sonnenwinkel_df = self.berechne_sonnenwinkel(gemeinsame_zeitachse, lat, lon)
        
        #self.df = pd.concat([wetter_df, sensor_df, sonnenwinkel_df], axis=1).reset_index()
        #self.df.rename(columns={'index': 'timestamp'}, inplace=True)
    
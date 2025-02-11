import pandas as pd
from datetime import datetime

class DataModuleStatic:

    def __init__(self, aufloesung='1T', soll_keys=None):
        self.aufloesung = aufloesung
        self.df = pd.DataFrame()
        
        # Soll-Keys für die Zuordnung (falls keine angegeben werden, Standardwerte)
        self.soll_keys = soll_keys if soll_keys else {
            'timestamp'     : 'timestamp',
            'SunPow'        : 'sunPower',
            'tmpAmbient'    : 'tmpAmbient',
            'tmpAmbient1'   : 'tmpAmbient1',
            'tmpAmbient2'   : 'tmpAmbient2',
            'presence'      : 'presence',
            'hd'            : 'hd',
            'tmpAmbientFc'  : 'tmpAmbientFc',
            'clouds'        : 'clouds',
            'sunOrtho'      : 'sunOrtho',
            'xHeating'      : 'xHeating',
            'tmpRoom'       : 'tmpRoom',
            'tmpInventory'  : 'tmpInventory',
            'tmpDoorRoom'   : 'tmpDoorRoom',
            'windSpeed'     : 'windSpeed',
            'visibility'    : 'visibility',
            'azimuth'       : 'azimuth',
            'elevation'     : 'elevation',
        }

    def load_dataframe(self, df: pd.DataFrame):
        self.df = df
        self.check_df()
    
    def load_csv(self, path: str):
        if not path:
            raise ValueError("Keine .csv Datei angegeben.")
        self.df = pd.read_csv(path)
        self.check_df()

    def save_csv(self, dateipfad: str):
        self.df.to_csv(dateipfad, index=False)

    def check_df(self):
        """
        Überprüft, ob die benötigten Spalten vorhanden sind und benennt sie gemäß den Soll-Keys um.
        """
        # Umbenennen der Spalten nach den Soll-Keys
        self.df = self.df.rename(columns=self.soll_keys)

        # Überprüfen, ob die 'timestamp' Spalte vorhanden ist
        if 'timestamp' not in self.df.columns:
            raise ValueError("Die 'timestamp'-Spalte fehlt im DataFrame.")

        # Sicherstellen, dass der timestamp als datetime formatiert ist
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
    
    def get_df(self):
        return self.df
    
    def get_timespan(self, start_time: str = None, end_time: str = None):
        """
        Filtert den DataFrame nach einer angegebenen Zeitspanne.
        Beide Parameter können als Strings im Format 'YYYY-MM-DD HH:MM:SS' angegeben werden.
        """
        if start_time:
            start_time = pd.to_datetime(start_time)
        if end_time:
            end_time = pd.to_datetime(end_time)

        # Filter anwenden, wenn Zeitrahmen angegeben wurde
        if start_time and end_time:
            return self.df[(self.df['timestamp'] >= start_time) & (self.df['timestamp'] <= end_time)] 
        elif start_time:
            return self.df[self.df['timestamp'] >= start_time]
        elif end_time:
            return self.df[self.df['timestamp'] <= end_time]
        
    def get_time_range(self):
        """
        Gibt das früheste und späteste Datum im DataFrame zurück.
        Rückgabe: Tupel (frühestes Datum, spätestes Datum) als Strings im Format 'YYYY-MM-DD HH:MM:SS'.
        Falls kein gültiger Zeitstempel vorhanden ist, wird None zurückgegeben.
        """
        if 'timestamp' not in self.df.columns or self.df.empty:
            print("Keine gültige Zeitinformation im Datensatz enthalten")
            return None, None  # Falls keine Daten vorhanden sind
        
        min_time = self.df['timestamp'].min()
        max_time = self.df['timestamp'].max()
        
        return min_time.strftime('%Y-%m-%d %H:%M:%S'), max_time.strftime('%Y-%m-%d %H:%M:%S')

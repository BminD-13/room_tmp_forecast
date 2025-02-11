import os
import sys
import pandas as pd

sys.path.append(os.path.abspath('./src'))

from data_sources.sunpos import sunpos

def berechne_sonnenwinkel(timestamps, lat=48.49159, lon=9.679936):
    """
    Berechnet die Sonnenwinkel (Azimut und Elevation) für eine Liste von Zeitstempeln.
    
    Parameter:
    timestamps: Pandas Series oder Liste von Zeitstempeln
    lat: Breitengrad des Standorts (Standard: 48.49159)
    lon: Längengrad des Standorts (Standard: 9.679936)
    
    Rückgabe:
    DataFrame mit Azimut und Elevation für jeden Zeitstempel
    """
    # Stellen Sie sicher, dass die Zeitstempel als datetime-Objekte vorliegen
    timestamps = pd.to_datetime(timestamps)  # Wandeln Sie die Zeitstempel in datetime um
    
    sonnenwinkel_daten = [sunpos(ts, [lat, lon]) for ts in timestamps]
    return pd.DataFrame(sonnenwinkel_daten, columns=['azimuth', 'elevation'], index=timestamps)

# Lade den ursprünglichen DataFrame
df = pd.read_csv(r"data\training\240331_Dataset.csv")

# Berechne Sonnenwinkel und füge diese zum DataFrame hinzu
df_sunpos = berechne_sonnenwinkel(df["timestamp"], lat=48.49159, lon=9.679936)

# Kombiniere die ursprünglichen Daten mit den neuen Sonnenwinkel-Daten
df_with_sunpos = pd.concat([df, df_sunpos], axis=1)

# Sicherstellen, dass die Zeitstempel in beiden DataFrames im richtigen Format sind
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_sunpos.index = pd.to_datetime(df_sunpos.index)

# Führe den Merge basierend auf der exakten Übereinstimmung der Zeitstempel durch
df_merged = df.merge(df_sunpos, left_on='timestamp', right_index=True, how='left')

# Speichere das kombinierte Dataset
df_merged.to_csv(r"data\training\240331_Dataset_with_sunpos.csv", index=False)

print(df_merged.head())


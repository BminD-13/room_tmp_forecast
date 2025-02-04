import requests
import pandas as pd
from datetime import datetime

def fetch_weather(api_key, lat, lon):
    """
    Ruft Wetterdaten von einer API ab und gibt ein Pandas-DataFrame mit relevanten Werten zurück.
    
    Parameter:
    - api_key: API-Schlüssel für den Wetterdienst
    - lat: Breitengrad
    - lon: Längengrad
    
    Rückgabe:
    - Pandas DataFrame mit Spalten: timestamp, t_aussen, bewölkung, windgeschwindigkeit
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Fehler beim Abrufen der Wetterdaten: {response.status_code}")
    
    data = response.json()
    
    # Wetterwerte extrahieren
    timestamp = datetime.utcfromtimestamp(data['dt']).isoformat() + 'Z'
    t_aussen = data['main']['temp']
    bewölkung = data['clouds']['all']  # % der Wolkenbedeckung
    windgeschwindigkeit = data['wind']['speed']  # m/s
    
    # Sonnenstrahlung abschätzen (Vereinfacht!)
    max_sonnenstrahlung = 1000  # Max. Leistung bei klarem Himmel
    sonnenleistung = max_sonnenstrahlung * (1 - (bewölkung / 100))  # Annäherung
    
    df = pd.DataFrame([{  
        "timestamp": timestamp,
        "t_aussen": t_aussen,
        "bewölkung": bewölkung,
        "windgeschwindigkeit": windgeschwindigkeit,
        "sonnenleistung": sonnenleistung
    }])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


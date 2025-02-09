import pandas as pd
import matplotlib.pyplot as plt

# Daten einlesen
data = pd.read_csv(r"C:\Users\benja\Desktop\Projekte\RegelungFussbodenheizung\RegelungFussbodenheizung_v015\WrittenData\merged_output.csv")

# Überprüfe die Spaltennamen und gib sie aus
print("Spaltennamen:", data.columns)

# Angenommen, die Zeitstempelspalte heißt 'index' (falls sie anders heißt, ersetze 'index' durch den richtigen Namen)
if 'index' in data.columns:
    data['timestamp'] = pd.to_datetime(data['index'])
else:
    print("Keine Zeitstempelspalte gefunden!")

# Plot 1: Temperaturen (tmpAmbient, tmpAmbient1, tmpAmbient2, tmpRoom)
plt.figure(figsize=(10,6))
plt.plot(data['timestamp'], data['tmpAmbient'], label='tmpAmbient', color='blue')
plt.plot(data['timestamp'], data['tmpAmbient1'], label='tmpAmbient1', color='green')
plt.plot(data['timestamp'], data['tmpAmbient2'], label='tmpAmbient2', color='red')
plt.plot(data['timestamp'], data['tmpRoom'], label='tmpRoom', color='orange')
plt.title('Temperaturen über der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Temperatur (°C)')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Wolkenbedeckung
plt.figure(figsize=(10,6))
plt.plot(data['timestamp'], data['clouds'], label='Clouds', color='gray')
plt.title('Wolkenbedeckung über der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Clouds')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Sonneneinstrahlung
plt.figure(figsize=(10,6))
plt.plot(data['timestamp'], data['SunPow'], label='SunPow', color='yellow')
plt.title('Sonneneinstrahlung über der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Sonneneinstrahlung')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 4: Sichtweite
plt.figure(figsize=(10,6))
plt.plot(data['timestamp'], data['visibility'], label='Visibility', color='purple')
plt.title('Sichtweite über der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Sichtweite (m)')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 5: xHeating
plt.figure(figsize=(10,6))
plt.plot(data['timestamp'], data['xHeating'], label='xHeating', color='red')
plt.title('Heizleistung (xHeating) über der Zeit')
plt.xlabel('Zeit')
plt.ylabel('Heizleistung')
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

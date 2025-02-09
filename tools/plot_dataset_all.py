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

# Erstelle ein großes Fenster für alle Plots
fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

# Plot 1: Temperaturen (tmpAmbient, tmpAmbient1, tmpAmbient2, tmpRoom)
axs[0].plot(data['timestamp'], data['tmpAmbient'], label='tmpAmbient', color='blue')
axs[0].plot(data['timestamp'], data['tmpAmbient1'], label='tmpAmbient1', color='green')
axs[0].plot(data['timestamp'], data['tmpAmbient2'], label='tmpAmbient2', color='red')
axs[0].plot(data['timestamp'], data['tmpRoom'], label='tmpRoom', color='orange')
axs[0].set_title('Temperaturen über der Zeit')
axs[0].set_ylabel('Temperatur (°C)')
axs[0].legend(loc='best')
axs[0].tick_params(axis='x', which='both', bottom=False, top=False)

# Plot 2: Wolkenbedeckung
axs[1].plot(data['timestamp'], data['clouds'], label='Clouds', color='gray')
axs[1].set_title('Wolkenbedeckung über der Zeit')
axs[1].set_ylabel('Clouds')
axs[1].legend(loc='best')
axs[1].tick_params(axis='x', which='both', bottom=False, top=False)

# Plot 3: Sonneneinstrahlung
axs[2].plot(data['timestamp'], data['SunPow'], label='SunPow', color='yellow')
axs[2].set_title('Sonneneinstrahlung über der Zeit')
axs[2].set_ylabel('Sonneneinstrahlung')
axs[2].legend(loc='best')
axs[2].tick_params(axis='x', which='both', bottom=False, top=False)

# Plot 4: Sichtweite
axs[3].plot(data['timestamp'], data['visibility'], label='Visibility', color='purple')
axs[3].set_title('Sichtweite über der Zeit')
axs[3].set_ylabel('Sichtweite (m)')
axs[3].legend(loc='best')
axs[3].tick_params(axis='x', which='both', bottom=False, top=False)

# Plot 5: xHeating
axs[4].plot(data['timestamp'], data['xHeating'], label='xHeating', color='red')
axs[4].set_title('Heizleistung (xHeating) über der Zeit')
axs[4].set_xlabel('Zeit')
axs[4].set_ylabel('Heizleistung')
axs[4].legend(loc='best')

# Alle Plots anpassen und anzeigen
plt.tight_layout()
plt.xticks(rotation=45)  # Drehe die x-Achsenbeschriftungen
plt.show()

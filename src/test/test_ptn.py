# es gibt nur einen sinnvollen parameter was man auch mit einer logistischen funktion berechnen kann!
# den zeitwert tau mit einer liste zu füttern liefert keine besonderen ergebnisse!

import numpy as np
import matplotlib.pyplot as plt

class PTNSystem:
    def __init__(self, dt):
        self.dt = dt
    
    def ptn(self, y, u, tau, n=1):
        if len(y) < n:
            y = [y[0]] * n  
        y_new = y.copy()
        for i in range(n):
            alpha = self.dt / (tau + self.dt)  
            y_new[i] = (1 - alpha) * y[i] + alpha * (u if i == 0 else y_new[i - 1])
        return y_new

# Simulationsparameter
dt = 0.1
t_max = 2000
t = np.arange(0, t_max, dt)
u = np.ones_like(t)  # Einheitssprung

tau1 = 100  # Trägheit am Anfang
tau2 = 100  # Trägheit am Ende
n = 20  # Ordnung des Systems

ptn_system = PTNSystem(dt)

# Erste Simulation (Trägheit vorn)
y = [0] * n
y_values1 = []
for u_t in u:
    y = ptn_system.ptn(y, u_t, tau1, n)
    y_values1.append(y[-1])  # Letzte Stufe als Ausgangswert

# Zweite Simulation (Trägheit hinten)
y = [0] * n
y_values2 = []
for u_t in u:
    y = ptn_system.ptn(y, u_t, tau2, n)
    y_values2.append(y[-1])

# Plot der Ergebnisse
plt.figure(figsize=(10, 5))
plt.plot(t, u, label="Eingangssignal (u)", linestyle="dashed", color="gray")
plt.plot(t, y_values1, label="Trägheit am Anfang (tau1)", linewidth=2)
plt.plot(t, y_values2, label="Trägheit am Ende (tau2)", linewidth=2, linestyle="dotted")
plt.xlabel("Zeit [s]")
plt.ylabel("Ausgang")
plt.title(f"Simulation einer PT{n}-Strecke mit variabler Trägheit")
plt.legend()
plt.grid()
plt.show()

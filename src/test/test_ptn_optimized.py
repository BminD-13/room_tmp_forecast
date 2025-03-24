import numpy as np
import matplotlib.pyplot as plt
import time


class ThermalObject:

    def __init__(self, tau:list, n:int , y0 = 0):
        self.set_param(n, tau)
        self.y = np.ones(self.n, dtype=np.float64) * y0
        self.dy = 0
    
    def set_param(self, n = None, tau=None):
        self.n   = int(n)
        self.tau = tau

    def set_tmp(self, y0):
        self.y = np.ones(self.n, dtype=np.float64) * y0
    
    def get_tmp(self):
        return self.y[-1]
    
    def ptn_interativ(self):
        self.y[0] += self.dy
        y_new = self.y.copy()
        for i in range(1, self.n):
            y_new[i] = (1 - self.tau) * self.y[i] + self.tau * y_new[i - 1]
        self.y = y_new
        self.dy = 0
        return self.y[-1]

    def ptn_vectorwise(self):
        y_shift = np.roll(self.y, 1)  # Vorne u, hinten abgeschnitten
        y_shift[0] = self.y[0] + self.dy
        self.y = (1 - self.tau) * self.y + self.tau * y_shift
        self.dy = 0
        return self.y[-1]
    
    def transfer_warming(self, u, rho=0.1):
        if rho != 0:
            self.dy += rho * (u - self.y[0])
    
    def sun_warming(self, dy, rho=0.2):
        if rho != 0:
            self.y[0] += rho * dy


# Simulationsparameter
n = 10000
ptn = 100
t = np.arange(0, n)
u = np.ones(n) # Einheitssprung
y1 = np.zeros(n) # Einheitssprung
y2 = np.zeros(n) # Einheitssprung

tau = 0.07

# Prozess 1
import time

for m in range(10,500,10):
    print(m)
    
    start_1 = time.time()

    walle1 = ThermalObject(tau, m)

    for i in range(n-10):
        walle1.transfer_warming(u[i], 0.1)
        y1[i] = walle1.ptn_interativ()

    end_1 = time.time()
    dauer_1 = end_1 - start_1
    print(f"Prozess 1 Laufzeit: {dauer_1:.6f} Sekunden")


    # Prozess 2
    start_2 = time.time()

    walle2 = ThermalObject(tau, m)

    for i in range(n-10):
        walle2.transfer_warming(u[i], 0.1)
        y2[i] = walle2.ptn_vectorwise()

    end_2 = time.time()
    dauer_2 = end_2 - start_2
    print(f"Prozess 2 Laufzeit: {dauer_2:.6f} Sekunden")


# Plot der Ergebnisse
plt.figure(figsize=(10, 5))
plt.plot(t, u, label="Eingangssignal (u)", linestyle="dashed", color="gray")
plt.plot(t, y1, label="erstes PT2-System", linewidth=2)
plt.plot(t, y2, label="zweites PT2-System", linewidth=2)

plt.xlabel("Zeit [s]")
plt.ylabel("Ausgang")
plt.title("Numerische Integration eines PT2-Systems (Euler-Verfahren)")
plt.legend()
plt.grid()
plt.show()

# Vektorielle Addition lohnt sich erst ab n = 30
# da die n bisher klein waren, müsste der iterative ansatz mit schleife genügen
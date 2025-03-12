import numpy as np
import matplotlib.pyplot as plt

# # Simulationsparameter
# n = 2000
# t = np.arange(0, n + 20)
# u = np.ones(n+20) # Einheitssprung
# y = np.zeros(n+20)
# y2 = np.zeros(n+20)
# tau = [0.002, 0.5, 0.2]

# def derivative(y):
#     n = len(y)
#     x = np.arange(1, n + 1) # 1, 2, ..., n
#     y = np.array(y)

#     # Design-Matrix für quadratisches Modell (x^2, x, konstante 1)
#     A = np.vstack([x**2, x, np.ones_like(x)]).T

#     # Least-Squares-Lösung (a, b, c)
#     theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
#     dy = 2 * theta[0] * n + theta[1]
#     ddy = theta[1]
#     return  dy , ddy
 
# def ptn_step(y, u, tau):
#     """ Numerische Simulation eines PT2-Systems """
#     dy , ddy = derivative(y)
#     y_neu = y[-1] + tau[0] * (u - y[-1]) * dy # + tau[2] * ddy
#     return y_neu

# # Beispiel 1: Langsame PT2-Strecke
# for i in range(int(n)):
#     y[i+20] = ptn_step(y[i+10:i+20], u[i+20], tau)


# # Plot der Ergebnisse
# plt.figure(figsize=(10, 5))
# plt.plot(t, u, label="Eingangssignal (u)", linestyle="dashed", color="gray")
# plt.plot(t, y, label="Langsames PT2-System", linewidth=2)
# plt.plot(t, y2, label="Schnelles PT2-System", linewidth=2, linestyle="dotted")

# plt.xlabel("Zeit [s]")
# plt.ylabel("Ausgang")
# plt.title("Numerische Integration eines PT2-Systems (Euler-Verfahren)")
# plt.legend()
# plt.grid()
# plt.show()

class thConductor():

    def __init__(self, tau, n:int = 3, y0 = 0):
        self.n      = n
        self.tau    = tau
        self.y      = np.zeros(n)

    def set_param(self, tau):
        self.tau = tau

    def ptn_iterativ(self, u):
        self.y[0] = self.y[0] + self.tau[0] * (u - self.y[0])
        for i in range(1, self.n-1):
            self.y[i] = self.y[i] + self.tau[1] * (self.y[i-1] - self.y[i])
        self.y[-1] = self.y[-1] + self.tau[-1] * (self.y[-2] - self.y[-1])
        return self.y[-1]

    def ptn_vector(self, u):
        tau = np.ones(self.n) * self.tau[1]
        tau[0] = self.tau[0]
        tau[-1] = self.tau[-1]
        y_shift = np.roll(self.y, 1)  # Vorne u, hinten abgeschnitten
        y_shift[0] = u
        self.y = self.y + tau * (y_shift - self.y)
        return self.y[-1]

# Simulationsparameter
n = 2000
ptn = 100
t = np.arange(0, n)
u = np.ones(n) # Einheitssprung
y = np.zeros(n) # Einheitssprung
tau = [0.01, 0.3, 0.01]

walle = thConductor(tau, ptn)

for i in range(n-10):
    y[i] = walle.ptn_iterativ(u[i])

# Plot der Ergebnisse
plt.figure(figsize=(10, 5))
plt.plot(t, u, label="Eingangssignal (u)", linestyle="dashed", color="gray")
plt.plot(t, y, label="Langsames PT2-System", linewidth=2)

plt.xlabel("Zeit [s]")
plt.ylabel("Ausgang")
plt.title("Numerische Integration eines PT2-Systems (Euler-Verfahren)")
plt.legend()
plt.grid()
plt.show()

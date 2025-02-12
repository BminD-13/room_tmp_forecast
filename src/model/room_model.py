# model
import numpy as np
import csv

class RaumModell:
    def __init__(self, dt, param_file=None, **kwargs):
        self.default_params = {
            "tau_wand": 1, "tau_speicher": 1, "tau_raum": 1,
            "n_wand": 1, "n_speicher": 3, "n_raum_storage": 3, "n_raum_wand": 3,
            "fensterfaktor": 1
        }

        self.dt = dt
        
        if param_file:
            self.load_parameters(param_file)
        
        for key, value in kwargs.items():
            if key in self.default_params:
                self.default_params[key] = value
        
        for key, value in self.default_params.items():
            setattr(self, key, value)
    
    def save_parameters(self, param_file):
        with open(param_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for key, value in self.default_params.items():
                writer.writerow([key, value])
    
    def load_parameters(self, param_file):
        with open(param_file, 'r') as file:
            reader = csv.reader(file)
            params = {rows[0]: float(rows[1]) for rows in reader}
        for key, value in params.items():
            if key in self.default_params:
                setattr(self, key, value)
    
    def ptn(self, y, u, tau, n=1):
        """
        Simuliert eine PTn-Strecke (n-stufiges Verzögerungsglied).
        
        Parameter:
        y:   Liste mit vorherigen Zuständen der PTn-Strecke.
        u:   Eingangsgröße (z. B. Temperatur, Steuergröße).
        dt:  Zeitschritt der Simulation.
        tau: Zeitkonstante des Systems (je größer tau, desto träger die Reaktion).
        n:   Ordnung der PTn-Strecke (also wie viele hintereinandergeschaltete PT1-Glieder).
        """
        if len(y) < n:
            y = [y[0]] * n  
        alpha = self.dt / (tau + self.dt)  
        y_new = y.copy()
        for i in range(n):
            y_new[i] = (1 - alpha) * y[i] + alpha * (u if i == 0 else y_new[i - 1])
        return y_new

    def orthogonalität(self, azimuth, elevation, surface_azimuth, surface_tilt):
        """
        Berechnet die projizierte Fläche einer Wand für eine Liste von Sonnenpositionen.

        Parameter:
        azimuth: Array der Sonnenazimutwinkel (0° = Norden, 90° = Osten, ...)
        elevation: Array der Sonnenhöhenwinkel über dem Horizont (0° = Horizont, 90° = Zenit)
        surface_azimuth: Ausrichtung der Oberfläche (0° = Norden, 90° = Osten, ...)
        surface_tilt: Neigungswinkel der Oberfläche (0° = vertikal, 90° = horizontal)

        Rückgabe:
        Array mit der Projektion der Sonnenstrahlung auf die Fläche (Werte zwischen 0 und 1)
        """

        # Umwandlung in Radiant für alle Werte
        azimuth, elevation = np.radians(azimuth), np.radians(elevation)
        surface_azimuth, surface_tilt = np.radians(surface_azimuth), np.radians(surface_tilt)

        # Sonnenvektoren (vektorisiert)
        sun_x = np.cos(elevation) * np.cos(azimuth)
        sun_y = np.cos(elevation) * np.sin(azimuth)
        sun_z = np.sin(elevation)

        # Flächennormalen (skalar, da konstant)
        normal_x = np.cos(surface_tilt) * np.cos(surface_azimuth)
        normal_y = np.cos(surface_tilt) * np.sin(surface_azimuth)
        normal_z = np.sin(surface_tilt)

        # Skalarprodukt (vektorisiert)
        ortho = sun_x * normal_x + sun_y * normal_y + sun_z * normal_z

        return np.maximum(0, ortho)  # Keine negativen Werte (wenn die Fläche im Schatten liegt)


    
    def sonnenstrahlung(self, tmp, sonnenleistung, orthogonalität):
        return tmp + sonnenleistung * orthogonalität  * self.fensterfaktor
    
    def raumtemperatur_model(self, tmp_0, tmp_aussen, sonnenleistung, orthogonalität):
        tmp_wall =  	[tmp_aussen[0]] * self.n_wand
        tmp_storage =  [tmp_0] * self.n_speicher
        tmp_room =      [tmp_0] * self.n_raum_storage
        
        ergebnisse = []
        for i in range(len(tmp_aussen)):

            tmp_wall = self.ptn(tmp_wall, tmp_aussen[i], self.tau_wand, self.n_wand)
            tmp_wall = self.sonnenstrahlung(tmp_wall, sonnenleistung[i], orthogonalität[i])
            
            tmp_storage = self.ptn(tmp_storage, tmp_room[-1], self.tau_speicher, self.n_speicher)
            tmp_storage = self.sonnenstrahlung(tmp_storage, sonnenleistung[i], orthogonalität[i])
            
            tmp_room = self.ptn(tmp_room, tmp_storage[-1], self.tau_raum, self.n_raum_storage)
            tmp_room = self.ptn(tmp_room, tmp_wall[-1], self.tau_raum, self.n_raum_wand)
            tmp_room = self.sonnenstrahlung(tmp_room, sonnenleistung[i], orthogonalität[i])
            
            ergebnisse.append(tmp_room[-1])
        
        return ergebnisse
    
    def run_model(self, dataset):

        sunOrtho = self.orthogonalität(dataset["azimuth"], dataset["elevation"], 180, 0)

        return  self.raumtemperatur_model(
                    tmp_0          = 21.5357487923,
                    tmp_aussen     = dataset["tmpAmbient"],
                    sonnenleistung = dataset["sunPower"],
                    orthogonalität = sunOrtho
                )

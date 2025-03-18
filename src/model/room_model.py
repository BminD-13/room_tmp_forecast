# model
import numpy as np
import csv

class RaumModell:

    def __init__(self, dt, params=None):

        self.ThermalObjects = []

        #                            R  W  S  F  s  a  h  g
        self.weights = np.matrix([  [0, 1, 1, 1, 1, 0, 0, 0], # Room
                                    [1, 0, 0, 0, 1, 1, 0, 0], # Wall
                                    [1, 0, 0, 0, 1, 0, 0, 0], # Storage
                                    [1, 0, 0, 0, 0, 0, 1, 1]]) # Floor

        #                    room   wall  storage floor
        self.objekt_param = [[1,1], [1,1], [1,1], [1,1]]

        self.dt = dt

        if isinstance(params, str):  
            self.load_parameters_from_file(params)  

        elif isinstance(params, np.ndarray):  
            self.load_parameters_from_matrix(params)

        self.initialize_thermal_objects()

    def save_parameters(self, param_file):
        with open(param_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Kombinieren von weights und objekt_param zu einer Matrix
            combined = np.hstack((self.weights, self.objekt_param))
            # Speichern in CSV-Datei
            writer.writerows(combined)

    def load_parameters_from_matrix(self, param_file):
        """Lädt die gespeicherte Matrix und speichert sie in den Objektvariablen."""
        self.weights = param_file[:, :-2]
        self.objekt_param = param_file[:, -2:]

    def load_parameters_from_file(self, param_file):
        with open(param_file, 'r') as file:
            reader = csv.reader(file)
            data = np.array([list(map(int, row)) for row in reader])
        
        # Trennen von weights und objekt_param
        self.weights = data[:, :-2]  # Alle Spalten außer den letzten beiden
        self.objekt_param = data[:, -2:]  # Die letzten zwei Spalten

    class ThermalObject:

        def __init__(self, tau:list, n:int , y0 = 0):
            self.set_param(n, tau)
            self.y = np.ones(self.n, dtype=np.float64) * y0
        
        def set_param(self, n = None, tau=None):
            self.n   = int(n)
            self.tau = tau

        def set_tmp(self, y0):
            self.y = np.ones(self.n, dtype=np.float64) * y0
        
        def get_tmp(self):
            return self.y[-1]
        
        def calc_ptn(self):
            y_new = self.y.copy()
            for i in range(1, self.n):
                y_new[i] = (1 - self.tau) * self.y[i] + self.tau * y_new[i - 1]
            self.y = y_new
            return self.y[-1]
        
        def transfer_warming(self, u, rho=0.1):
            if rho != 0:
                self.y[0] = self.y[0] + rho * (u - self.y[0])
        
        def sun_warming(self, dy, rho=0.1):
            if rho != 0:
                self.y[0] += rho * dy

    def initialize_thermal_objects(self, y0=[20.5, 20.5, 20.5, 20.5]):
        for i in range(len(self.objekt_param)):
            self.ThermalObjects.append(self.ThermalObject(
                tau = self.objekt_param[i, 0],
                n   = self.objekt_param[i, 1],
                y0  = y0[i]
                )
            )

    def raumtemperatur_model(self, tmp_aussen, sonnenleistung, orthogonalität, heating):
        K = len(tmp_aussen)
        M = len(self.weights)
        tmp_pred = np.zeros((M,K))

        for k in range(len(tmp_aussen)):
            for m in range(M):
                self.ThermalObjects[m].transfer_warming(self.ThermalObjects[0].get_tmp(), self.weights[m,0])
                self.ThermalObjects[m].transfer_warming(self.ThermalObjects[1].get_tmp(), self.weights[m,1])
                self.ThermalObjects[m].transfer_warming(self.ThermalObjects[2].get_tmp(), self.weights[m,2])
                self.ThermalObjects[m].transfer_warming(self.ThermalObjects[3].get_tmp(), self.weights[m,3])
                self.ThermalObjects[m].sun_warming(sonnenleistung[k] * orthogonalität[k], self.weights[m,5])
                self.ThermalObjects[m].transfer_warming(tmp_aussen[k], self.weights[m,5])
                self.ThermalObjects[m].transfer_warming(heating[k] * 30, self.weights[m,6])
                self.ThermalObjects[m].transfer_warming(1-heating[k] * 7, self.weights[m,7])
                self.ThermalObjects[m].calc_ptn()
                tmp_pred[m, k] = self.ThermalObjects[m].get_tmp()        
        return tmp_pred

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

    def sonnenstrahlung(self, tmp, sonnenleistung, orthogonalität, param):
        return tmp + sonnenleistung * orthogonalität  * param
    
    def run_model(self, dataset):

        sunOrtho = self.orthogonalität(dataset["azimuth"], dataset["elevation"], 180, 90)

        return  self.raumtemperatur_model(
                    tmp_aussen     = dataset["tmpAmbient"],
                    sonnenleistung = dataset["sunPower"],
                    heating        = dataset["xHeating"],
                    orthogonalität = sunOrtho
                )

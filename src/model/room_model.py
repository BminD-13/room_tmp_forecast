# model
import csv

class RaumModell:
    def __init__(self, dt, param_file=None, **kwargs):
        self.default_params = {
            "tau_wand": 1000, "tau_speicher": 2000, "tau_raum": 500,
            "n_wand": 2, "n_speicher": 1, "n_raum": 3,
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

    def orthogonalität(self, azimuth, elevation, normale):
        """
        Berechnet die pojizierte Fläche einer Wand,
        welche von der Sonne bestrahlt wird

        Parameter:
        azimuth:   Himmelsrichtung der Sonne
        elevation: Winkel der Sonne vom Boden gemessen
        normale:   Flächennormale der betrachteten Oberfläche
        """
        return azimuth, elevation
    
    def sonnenstrahlung(self, tmp, sonnenleistung, orthogonalität):
        return tmp + sonnenleistung * orthogonalität # * self.fensterfaktor
    
    def raumtemperatur_model(self, tmp_0, tmp_aussen, sonnenleistung, orthogonalität):
        tmp_wand =  	[tmp_aussen[0]] * self.n_wand
        tmp_speicher =  [tmp_aussen[0]] * self.n_speicher
        tmp_raum =      [tmp_0] * self.n_raum
        
        ergebnisse = []
        for i in range(len(tmp_aussen)):

            tmp_wand = self.ptn(tmp_wand, tmp_aussen[i], self.tau_wand, self.n_wand)
            tmp_wand = self.sonnenstrahlung(tmp_wand, sonnenleistung[i], orthogonalität[i])
            
            tmp_speicher = self.ptn(tmp_speicher, tmp_raum[-1], self.tau_speicher, self.n_speicher)
            tmp_speicher = self.sonnenstrahlung(tmp_speicher, sonnenleistung[i], orthogonalität[i])
            
            tmp_raum = self.ptn(tmp_raum, tmp_speicher[-1], self.tau_raum, self.n_raum)
            tmp_raum = self.ptn(tmp_raum, tmp_wand[-1], self.tau_raum, self.n_raum)
            tmp_raum = self.sonnenstrahlung(tmp_raum, sonnenleistung[i], orthogonalität[i])
            
            ergebnisse.append(tmp_raum[-1])
        
        return ergebnisse

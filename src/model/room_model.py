# model
import numpy as np
import csv

class RaumModell:
    def __init__(self, param_file=None, **kwargs):
        self.default_params = {
            "tau_wand": 1000, "tau_speicher": 2000, "tau_raum": 500,
            "n_wand": 2, "n_speicher": 1, "n_raum": 3,
            "fensterflaeche": 2,
            "absorption_wand": 0.5, "absorption_speicher": 0.6, "absorption_raum": 0.4
        }
        
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
    
    def ptn(self, y_prev, u, dt, tau, n=1):
        if len(y_prev) < n:
            y_prev = [y_prev[0]] * n  
        alpha = dt / (tau + dt)  
        y_new = y_prev.copy()
        for i in range(n):
            y_new[i] = (1 - alpha) * y_prev[i] + alpha * (u if i == 0 else y_new[i - 1])
        return y_new
    
    def sonnenstrahlung(self, sonnenleistung, einfallswinkel):
        return sonnenleistung * einfallswinkel * self.fensterflaeche
    
    def raumtemperatur_model(self, t_aussen, sonnenleistung, einfallswinkel, dt=60):
        t_wand = [t_aussen[0]] * self.n_wand
        t_speicher = [t_aussen[0]] * self.n_speicher
        t_raum = [t_aussen[0]] * self.n_raum
        
        ergebnisse = []
        for i in range(len(t_aussen)):
            p_sonne_wand = self.sonnenstrahlung(sonnenleistung[i], einfallswinkel[i]) * self.absorption_wand
            p_sonne_speicher = self.sonnenstrahlung(sonnenleistung[i], einfallswinkel[i]) * self.absorption_speicher
            p_sonne_raum = self.sonnenstrahlung(sonnenleistung[i], einfallswinkel[i]) * self.absorption_raum
            
            t_wand = self.ptn(t_wand, t_aussen[i], dt, self.tau_wand, self.n_wand)
            t_wand = self.ptn(t_wand, p_sonne_wand, dt, self.tau_wand, self.n_wand)
            
            t_speicher = self.ptn(t_speicher, t_raum[-1], dt, self.tau_speicher, self.n_speicher)
            t_speicher = self.ptn(t_speicher, p_sonne_speicher, dt, self.tau_speicher, self.n_speicher)
            
            t_raum = self.ptn(t_raum, t_speicher[-1], dt, self.tau_raum, self.n_raum)
            t_raum = self.ptn(t_raum, t_wand[-1], dt, self.tau_raum, self.n_raum)
            t_raum = self.ptn(t_raum, p_sonne_raum, dt, self.tau_raum, 1)
            
            ergebnisse.append(t_raum[-1])
        
        return ergebnisse

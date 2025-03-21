import os
import sys
import numpy as np


sys.path.append(os.path.abspath('./src'))  
from model.room_model import RaumModell

# ==============================
# Fitness-Funktion
# ==============================
def fitness(params, dataset, real_temp):
    """Berechnet die Fitness eines Parametersatzes durch Simulation."""
    model = RaumModell(dt=1, params=params)
    tmp_pred = model.run_model(dataset)
    room_tmp_pred = tmp_pred[0]
    wall_tmp_pred_max = np.max(tmp_pred[1])
    real_temp = np.array(real_temp)

    min_length = min(len(room_tmp_pred), len(real_temp))
    room_tmp_pred = room_tmp_pred[:min_length]
    real_temp = real_temp[:min_length]

    # Fehlerberechnung
    quadr_error = np.sum((room_tmp_pred - real_temp) ** 2)

    # Gradientendifferenz
    pred_grad = np.diff(room_tmp_pred)
    real_grad = np.diff(real_temp)
    gradient_error = np.sum((pred_grad - real_grad) ** 2)

    # Normalisierung
    sigma_real = np.std(real_temp)
    total_error = (quadr_error + gradient_error) / (sigma_real + 1e-6)

    # Unrealistische Wandtemperatur 
    if wall_tmp_pred_max > 30:
        total_error *= wall_tmp_pred_max / 30

    return -total_error  # Negativer Fehler für Maximierung

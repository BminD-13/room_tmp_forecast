import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell

# ==============================
# Vergleichsplot: Vorhersage vs. Tatsächliche Temperatur
# ==============================
def plot_full_data(best_params, dataset, log_dir):
    """Plottet die Modellvorhersage gegen die tatsächlichen Werte mit mehreren Subplots."""
    model = RaumModell(dt=1, params=best_params)
    tmp_pred = model.run_model(dataset)

    fig = plt.figure(figsize=(10, 15))  # Gesamthöhe der Figur    fig.suptitle("Modellvorhersage vs. Tatsächliche Temperatur", fontsize=14)
    gs = gridspec.GridSpec(3, 1, height_ratios=[12, 5, 1])  # Höhenverhältnisse für die 5 Subplots
    axes = []
    for i in range(3):
        axes.append(fig.add_subplot(gs[i]))

    # 1. Tatsächliche Temperatur
    axes[0].plot(dataset["tmpAmbient"], label="Außentemperatur", color="cyan", linestyle="-")
    axes[0].plot(dataset["tmpRoom"], label="Tatsächliche Temperatur", color="blue", linestyle="-")
    axes[0].plot(tmp_pred[0], label="Raum", color="red", linestyle="dashed")
    axes[0].plot(tmp_pred[1], label="Wand", color="yellow", linestyle="dashed")
    axes[0].plot(tmp_pred[2], label="Inventar", color="green", linestyle="dashed")
    axes[0].plot(tmp_pred[3], label="Boden", color="orange", linestyle="dashed")
    axes[0].set_ylabel("Temperatur [°C]")
    axes[0].legend()
    axes[0].grid(True)

    # 2. SunPower
    axes[1].plot(dataset["sunPower"], label="sunPower", color="red", linestyle="-")
    axes[1].plot(dataset["elevation"], label="elevation", color="orange", linestyle="-")
    axes[1].set_ylabel("val [0-1023]")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Heizleistung
    axes[2].plot(dataset["xHeating"], label="Heizung", color="red", linestyle="-")
    axes[2].set_ylabel("Leistung")
    axes[2].legend()
    axes[2].grid(True)

    # Layout anpassen und speichern
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(log_dir, "prediction_vs_actual.png"))
    plt.close()

    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")

def plot_prediction_vs_actual(best_params, dataset, log_dir):
    """Plottet die Modellvorhersage gegen die tatsächlichen Werte."""
    model = RaumModell(dt=1, params=best_params)
    tmp_pred = model.run_model(dataset)

    plt.figure(figsize=(10, 5))
    plt.plot(dataset["tmpRoom"], label="Tatsächliche Temperatur", color="blue", linestyle="-")
    plt.plot(tmp_pred[0], label="Raum", color="red", linestyle="dashed")
    plt.plot(tmp_pred[1], label="Wand", color="yellow", linestyle="dashed")
    plt.plot(tmp_pred[2], label="Inventar", color="green", linestyle="dashed")
    plt.plot(tmp_pred[3], label="Boden", color="orange", linestyle="dashed")
    plt.xlabel("Zeit")
    plt.ylabel("Temperatur [°C]")
    plt.title("Modellvorhersage vs. Tatsächliche Temperatur")
    plt.legend()
    plt.grid(True)

    # Speichern des Plots
    plt.savefig(os.path.join(log_dir, "prediction_vs_actual.png"))
    plt.close()

    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")


# ==============================
# Fitness-Plot-Funktion
# ==============================
def plot_fitness(fitness_history, log_dir):
    """Erstellt und speichert den Fitness-Plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fitness_history) + 1), np.abs(fitness_history), marker="o", linestyle="-", color="b")
    plt.yscale("log")    
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness-Entwicklung über Generationen")
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "fitness_plot.png"))
    plt.close()
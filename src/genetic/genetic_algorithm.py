import numpy as np
import sys
import os
import json
import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell

# ==============================
# Parameter für den GA
# ==============================
POPULATION_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.3

# ==============================
# Logging-Verzeichnis anlegen
# ==============================
def create_log_directory():
    """Erstellt ein neues Log-Verzeichnis mit Zeitstempel."""
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    #log_dir = os.path.join("./data/logged/genetic", timestamp)
    log_dir = os.path.join("./data/logged/test", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# ==============================
# Zufälliges Individuum generieren
# ==============================
def random_individual():
    """Generiert ein zufälliges Individuum."""
    individual = {
        "tau_raum_wand":    np.random.uniform(0, 1),  #
        "tau_raum_speicher":np.random.uniform(0, 20),    
        "tau_storage_room": np.random.uniform(0, 10),  #
        "tau_wall_ambient": np.random.uniform(0, 10),
        "sun_wall":         np.random.uniform(0, 1),  #
        "sun_room":         np.random.uniform(0, 5),  #
        "sun_storage":      np.random.uniform(0, 5),  # 
    }
    return individual


# ==============================
# Initialisierung der Population
# ==============================
def initialize_population():
    """Erstellt die initiale Population mit zufälligen Parametern."""
    population = []
    for _ in range(POPULATION_SIZE):
        individual = random_individual()
        population.append(individual)
    return population


# ==============================
# Fitness-Funktion
# ==============================
def fitness_function(params, dataset, real_temp):
    """Berechnet die Fitness eines Parametersatzes durch Simulation."""
    model = RaumModell(dt=1, **params)
    predicted_temp = model.run_model(dataset)
    predicted_temp = np.array(predicted_temp)
    real_temp = np.array(real_temp)

    min_length = min(len(predicted_temp), len(real_temp))
    predicted_temp = predicted_temp[:min_length]
    real_temp = real_temp[:min_length]

    # Fehlerberechnung mit Zeitgewichtung
    time_weights = np.linspace(1, 2, min_length)
    weighted_error = np.sum((predicted_temp - real_temp) ** 2 * time_weights)

    # Gradientendifferenz
    pred_grad = np.diff(predicted_temp)
    real_grad = np.diff(real_temp)
    gradient_error = np.sum((pred_grad - real_grad) ** 2)

    # Normalisierung
    sigma_real = np.std(real_temp)
    total_error = (weighted_error + gradient_error) / (sigma_real + 1e-6)

    return -total_error  # Negativer Fehler für Maximierung

# ==============================
# Selektion, Crossover, Mutation
# ==============================
def selection(population, scores):
    """Roulette-Wheel-Selektion basierend auf exponentieller Fitness."""
    probabilities = np.exp(scores - np.max(scores))  
    probabilities /= np.sum(probabilities)
    return population[np.random.choice(len(population), p=probabilities)]

def crossover(parent1, parent2):
    """Mittelwert-Crossover: Mischt zwei Eltern zu einem Kind."""
    return {key: (parent1[key] + parent2[key]) / 2 for key in parent1}

def mutate(individual):
    """Mutiert ein Individuum durch zufällige Änderungen."""
    mutated = individual.copy()
    for key in mutated:
        if np.random.rand() < MUTATION_RATE:
            if "n_" in key:
                mutated[key] = np.clip(mutated[key] + np.random.randint(-1, 2), 1, 5)
            else:
                mutated[key] *= np.random.uniform(0.9, 1.1)  
    return mutated

# ==============================
# Vergleichsplot: Vorhersage vs. Tatsächliche Temperatur
# ==============================
def plot_prediction_vs_actual(best_params, dataset, real_temp, log_dir):
    """Plottet die Modellvorhersage gegen die tatsächlichen Werte."""
    model = RaumModell(dt=1, **best_params)
    predicted_temp = model.run_model(dataset)

    plt.figure(figsize=(10, 5))
    plt.plot(real_temp, label="Tatsächliche Temperatur", color="blue", linestyle="-")
    plt.plot(predicted_temp, label="Vorhersage", color="red", linestyle="dashed")
    plt.xlabel("Zeit")
    plt.ylabel("Temperatur [°C]")
    plt.title("Modellvorhersage vs. Tatsächliche Temperatur")
    plt.legend()
    plt.grid(True)

    # Speichern des Plots
    plt.savefig(os.path.join(log_dir, "prediction_vs_actual.png"))
    plt.close()

    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")

import random

# ==============================
# Genetischer Algorithmus mit frischem Blut und Behalten des besten Individuums
# ==============================
def genetic_algorithm():
    """Führt den genetischen Algorithmus aus und speichert Ergebnisse."""
    
    # Daten laden
    DataModule = DataModuleStatic()
    DataModule.load_csv(r"data\training\240331_Dataset_01.csv")
    start, end = DataModule.get_time_range()
    endtime = DataModule.df["timestamp"]
    dataset = DataModule.get_timespan(start, endtime[3000])
    real_temp = dataset["tmpRoom"]

    # Log-Ordner erstellen
    log_dir = create_log_directory()

    # Population initialisieren
    population = initialize_population()

    best_params_per_epoch = []
    fitness_history = []

    for generation in range(GENERATIONS):
        scores = [fitness_function(ind, dataset, real_temp) for ind in population]

        # Beste Fitness & Parameter speichern
        best_fitness = max(scores)
        best_individual = population[np.argmax(scores)]
        best_params_per_epoch.append({"generation": generation + 1, "fitness": best_fitness, "params": best_individual})
        fitness_history.append(best_fitness)

        print(f"Generation {generation+1}: Beste Fitness = {best_fitness:.4f}")

        # Auswahl der besten 8 Individuen (mit hoher Fitness)
        selected_parents = selection_for_best(population, scores, num_parents=8)

        # Erstellen von Nachkommen durch Crossover und Mutation (6 Individuen)
        new_children = []
        while len(new_children) < 6:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_children.extend([child1, child2])

        # Erstellen von zufälligen Individuen (6 Individuen)
        random_individuals = [random_individual() for _ in range(6)]

        # Neue Population generieren (beste 8 Individuen + 6 durch Crossover + 6 zufällige)
        population = selected_parents + new_children + random_individuals

        # Sicherstellen, dass das beste Individuum immer erhalten bleibt
        if best_individual not in population:
            population[random.randint(0, len(population) - 1)] = best_individual

    # Beste Parameter in JSON speichern
    with open(os.path.join(log_dir, "best_params_per_epoch.json"), "w") as f:
        json.dump(best_params_per_epoch, f, indent=4)

    # Fitness-Plot speichern
    plot_fitness(fitness_history, log_dir)

    # Vergleichsplot: Vorhersage vs. Tatsächliche Temperatur
    plot_prediction_vs_actual(best_individual, dataset, real_temp, log_dir)

    print("\nOptimierung abgeschlossen!")
    print(f"Beste Parameter gespeichert unter: {log_dir}/best_params_per_epoch.json")
    print(f"Fitness-Plot gespeichert unter: {log_dir}/fitness_plot.png")
    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")

    return best_individual

# ==============================
# Auswahl der besten Individuen
# ==============================
def selection_for_best(population, scores, num_parents):
    """Wählt die besten Individuen basierend auf ihrer Fitness aus."""
    # Paare (Individuum, Fitness) und sortieren nach Fitness
    indexed_population = list(zip(population, scores))
    sorted_population = sorted(indexed_population, key=lambda x: x[1], reverse=True)
    
    # Wählen der besten `num_parents` Individuen
    selected_parents = [indiv for indiv, _ in sorted_population[:num_parents]]
    
    return selected_parents

# ==============================
# Fitness-Plot-Funktion
# ==============================
def plot_fitness(fitness_history, log_dir):
    """Erstellt und speichert den Fitness-Plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker="o", linestyle="-", color="b")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness-Entwicklung über Generationen")
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "fitness_plot.png"))
    plt.close()

# ==============================
# Starten des Algorithmus
# ==============================
if __name__ == "__main__":
    best_params = genetic_algorithm()
    print("\nOptimierte Modellparameter:")
    print(best_params)

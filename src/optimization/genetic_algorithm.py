import numpy as np
import sys
import os
import json
import datetime
import evaluate

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell
from data_visual.plots import *

# ==============================
# Parameter für den GA
# ==============================
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.25
TIME_SPAN = 15000

sub_dir = "genetic_03"

# ==============================
# Logging-Verzeichnis anlegen
# ==============================
def create_log_directory(additional = ""):
    """Erstellt ein neues Log-Verzeichnis mit Zeitstempel."""
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    #log_dir = os.path.join("./data/logged/genetic", timestamp)
    log_dir = os.path.join("./data/logged", sub_dir, timestamp + additional)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# ==============================
# Exponentiell Individuum generieren
# ==============================
#                             R    W    S    F    s    a    h    g    t   n
default_matrix = np.matrix([[0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1 ,1],  # Room
                            [0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1 ,1],  # Wall
                            [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1 ,1],  # Storage
                            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1 ,1]]) # Floor

def random_exp_individual(preset = default_matrix):
    individual = np.zeros_like(preset, float)
    for i in range(preset.shape[0]):
        for j in range(preset.shape[1]):
            if preset[i, j] != 0:
                if j == 9:
                    individual[i, j] = np.random.poisson(lam=preset[i, j]) + 1
                else:
                    individual[i, j] = np.random.exponential(preset[i, j])
    return individual

# ==============================
# Initialisierung der Population
# ==============================
def initialize_population(population_size = POPULATION_SIZE):
    return [random_exp_individual() for _ in range(population_size)]

def load_param_from_json(root_dir, fitness_filter=-30000):
    matrices = []

    # Alle Dateien in Unterordnern rekursiv durchsuchen
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".json"):  # Nur JSON-Dateien verarbeiten
                file_path = os.path.join(dirpath, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)  # JSON laden
                    
                    # `params`-Matrizen aus jedem Eintrag extrahieren
                    for entry in data:
                        if "params" in entry:
                            if entry["fitness"] > fitness_filter:
                                matrices.append(np.array(entry["params"]))  # In NumPy-Array umwandeln

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Fehler beim Einlesen von {file_path}: {e}")

    return matrices

# ==============================
# Selektion, Crossover, Mutation
# ==============================
def selection(population, scores):
    probabilities = np.exp(scores - np.max(scores))  
    probabilities /= np.sum(probabilities)
    return population[np.random.choice(len(population), p=probabilities)]

def crossover(parent1, parent2):
    return np.where(np.random.rand(*parent1.shape) > 0.5, parent1, parent2)

def mutate(individual, mutation_rate=0.1):
    mutation_mask = np.random.rand(*individual.shape) < mutation_rate
    
    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            if mutation_mask[i, j] and default_matrix[i, j] != 0:
                if j == 9:
                    individual[i, j] = np.random.poisson(lam=5) + 1
                else:
                    individual[i, j] *= np.random.exponential(0.05)
    return individual

import random

# ==============================
# Genetischer Algorithmus mit frischem Blut und Behalten des besten Individuums
# ==============================
def genetic_algorithm():
    """Führt den genetischen Algorithmus aus und speichert Ergebnisse."""
    
    # Daten laden
    DataModule = DataModuleStatic()
    DataModule.load_csv(r"data\training\240331_Dataset_01.csv")
    timestamps = DataModule.df["timestamp"]
    length = DataModule.len()

    dataset = DataModule.get_timespan(timestamps[8000], timestamps[20000])
    dataset.reset_index(drop=True, inplace=True)
    real_temp = dataset["tmpRoom"]

    # Log-Ordner erstellen
    log_dir = create_log_directory()

    # Population initialisieren
    population = initialize_population()

    best_params_per_epoch = []
    fitness_history = []

    for generation in range(GENERATIONS):

        # random_start = np.random.randint(0, length - (TIME_SPAN + 1))
        # dataset = DataModule.get_timespan(timestamps[random_start], timestamps[random_start + TIME_SPAN])
        # dataset.reset_index(drop=True, inplace=True)
        # real_temp = dataset["tmpRoom"]

        scores = [evaluate.fitness(ind, dataset, real_temp) for ind in population]

        # Beste Fitness & Parameter speichern
        if np.all(np.isnan(scores)):  
            print("Alle Werte sind NaN! Kein valider Maximalwert.")
        else:
            best_fitness = np.nanmax(scores)
            best_individual = population[np.nanargmax(scores)]
        best_params_per_epoch.append({"generation": generation + 1, "fitness": best_fitness, "params": best_individual})
        fitness_history.append(best_fitness)

        print(f"Generation {generation + 1}: Beste Fitness = {best_fitness:.4f}")

        # Auswahl der besten Individuen (mit hoher Fitness)
        selected_parents = selection_for_best(population, scores, num_parents=int(POPULATION_SIZE * 0.2))

        # Erstellen von Nachkommen durch Crossover und Mutation
        new_children = []
        while len(new_children) < POPULATION_SIZE * 0.4:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_children.extend([child1, child2])
        
        # besten aus allen generationen erhalten
        fitness_scores = np.array([entry["fitness"] for entry in best_params_per_epoch])
        best_index = np.nanargmax(fitness_scores)
        all_time_best = best_params_per_epoch[best_index]["params"]        # Erstellen von zufälligen Individuen

        n_randoms = int(POPULATION_SIZE * 0.4 -1)
        random_individuals = [random_exp_individual(population[i]) for i in range(n_randoms)]

        # Neue Population generieren (beste Individuen + Crossover + zufällige)
        population = [all_time_best] + selected_parents + new_children + random_individuals

    # Fitness-Plot speichern
    plot_fitness(fitness_history, log_dir)

    # Vergleichsplot: Vorhersage vs. Tatsächliche Temperatur
    plot_full_data(best_individual, dataset, log_dir)

    # NumPy-Arrays in Listen konvertieren
    for entry in best_params_per_epoch:
        entry["params"] = entry["params"].tolist()  # Wandelt NumPy-Array in eine Liste um

    # In eine JSON-Datei speichern
    save_dir = os.path.join(log_dir, "best_params_per_epoch.json")
    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(best_params_per_epoch, f, indent=4, ensure_ascii=False) 

    print("\nOptimierung abgeschlossen!")
    print(f"Beste Parameter gespeichert unter: {log_dir}/best_params_per_epoch.json")
    print(f"Fitness-Plot gespeichert unter: {log_dir}/fitness_plot.png")
    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")

    return best_params_per_epoch[-1]

# ==============================
# Genetischer Algorithmus zur lokalen optimierung mit bestehenden individuen
# ==============================
def genetic_local_algo(folder_with_json_files):
    """Führt den genetischen Algorithmus aus und speichert Ergebnisse."""
    
    # Daten laden
    DataModule = DataModuleStatic()
    DataModule.load_csv(r"data\training\240331_Dataset_01.csv")
    timestamps = DataModule.df["timestamp"]
    length = DataModule.len()

    dataset = DataModule.get_timespan(timestamps[8000], timestamps[20000])
    dataset.reset_index(drop=True, inplace=True)
    real_temp = dataset["tmpRoom"]

    # Log-Ordner erstellen
    log_dir = create_log_directory()

    # Population initialisieren
    population = load_param_from_json(root_dir=folder_with_json_files)
    print(f"Anzahl Individuen: {len(population)}")

    best_params_per_epoch = []
    fitness_history = []

    for generation in range(GENERATIONS):

        scores = [evaluate.fitness(ind, dataset, real_temp) for ind in population]

        # Beste Fitness & Parameter speichern
        if np.all(np.isnan(scores)):  
            print("Alle Werte sind NaN! Kein valider Maximalwert.")
        else:
            best_fitness = np.nanmax(scores)
            best_individual = population[np.nanargmax(scores)]
        best_params_per_epoch.append({"generation": generation + 1, "fitness": best_fitness, "params": best_individual})
        fitness_history.append(best_fitness)

        print(f"Generation {generation + 1}: Beste Fitness = {best_fitness:.4f}")

        # Auswahl der besten Individuen (mit hoher Fitness)
        selected_parents = selection_for_best(population, scores, num_parents=int(POPULATION_SIZE * 0.4))

        # Erstellen von Nachkommen durch Crossover und Mutation
        new_children = []
        while len(new_children) < POPULATION_SIZE * 0.6 - 1:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_children.extend([child1, child2])
        
        # besten aus allen generationen erhalten
        fitness_scores = np.array([entry["fitness"] for entry in best_params_per_epoch])
        best_index = np.nanargmax(fitness_scores)
        all_time_best = best_params_per_epoch[best_index]["params"]        # Erstellen von zufälligen Individuen

        # Neue Population generieren (beste Individuen + Crossover + zufällige)
        population = [all_time_best] + selected_parents + new_children

    # Fitness-Plot speichern
    plot_fitness(fitness_history, log_dir)

    # Vergleichsplot: Vorhersage vs. Tatsächliche Temperatur
    plot_full_data(best_individual, dataset, log_dir)

    # NumPy-Arrays in Listen konvertieren
    for entry in best_params_per_epoch:
        entry["params"] = entry["params"].tolist()  # Wandelt NumPy-Array in eine Liste um

    # In eine JSON-Datei speichern
    save_dir = os.path.join(log_dir, "best_params_per_epoch.json")
    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(best_params_per_epoch, f, indent=4, ensure_ascii=False) 

    print("\nOptimierung abgeschlossen!")
    print(f"Beste Parameter gespeichert unter: {log_dir}/best_params_per_epoch.json")
    print(f"Fitness-Plot gespeichert unter: {log_dir}/fitness_plot.png")
    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")

    return best_params_per_epoch[-1]

# ==============================
# Genetischer Algorithmus zur lokalen optimierung mit bestehenden individuen
# ==============================
def filter_results_by_fitness(folder_with_json_files, n=200):
    """Führt den genetischen Algorithmus aus und speichert Ergebnisse."""
    
    # Daten laden
    DataModule = DataModuleStatic()
    DataModule.load_csv(r"data\training\240331_Dataset_01.csv")
    timestamps = DataModule.df["timestamp"]
    length = DataModule.len()

    # Log-Ordner erstellen
    log_dir = create_log_directory("filtered_by_fitness")

    # Population initialisieren
    population = load_param_from_json(root_dir=folder_with_json_files)
    print(f"Anzahl Individuen: {len(population)}")

    best_params_per_epoch = []
    fitness_history = []


    dataset = DataModule.get_df()
    real_temp = dataset["tmpRoom"]

    scores = [evaluate.fitness(ind, dataset, real_temp) for ind in population]
    
    if np.all(np.isnan(scores)):  
        print("Alle Werte sind NaN! Kein valider Maximalwert.")
    else:
        # Sortiere die Indizes nach Score (höchste zuerst)
        sorted_indices = np.argsort(scores)[::-1]  # Absteigend sortieren
        sorted_indices = [idx for idx in sorted_indices if not np.isnan(scores[idx])]  # NaN entfernen
        
        # N beste Individuen auswählen
        best_individuals = [population[idx] for idx in sorted_indices[:n]]
        best_fitness_values = [scores[idx] for idx in sorted_indices[:n]]

        # Speichern der besten Individuen
        for j in range(len(best_individuals)):
            best_params_per_epoch.append({
                "generation": i + 1,
                "fitness": best_fitness_values[j],
                "params": best_individuals[j]
            })
        
        # Das beste Fitness-Ergebnis speichern
        fitness_history.append(best_fitness_values[0])

    # NumPy-Arrays in Listen konvertieren
    for entry in fitness_history:
        entry["params"] = entry["params"].tolist()  # Wandelt NumPy-Array in eine Liste um

    # In eine JSON-Datei speichern
    save_dir = os.path.join(log_dir, "best_params_per_epoch.json")
    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(best_params_per_epoch, f, indent=4, ensure_ascii=False) 

    print("\nOptimierung abgeschlossen!")
    print(f"Beste Parameter gespeichert unter: {log_dir}/best_params_per_epoch.json")
    print(f"Fitness-Plot gespeichert unter: {log_dir}/fitness_plot.png")
    print(f"Vergleichsplot gespeichert unter: {log_dir}/prediction_vs_actual.png")

    return best_params_per_epoch[-1]

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
# Starten des Algorithmus
# ==============================
if __name__ == "__main__":

    best_params_per_training = []

    for i in range(2):
        print(i)
        best_params = genetic_algorithm()
        #best_params = genetic_local_algo("data\logged\genetic")
        best_params_per_training.append(best_params)

    # In eine JSON-Datei speichern
    log_dir = create_log_directory("_sum")
    save_dir = os.path.join(log_dir, "best_params_per_epoch.json")
    #with open(save_dir, "w", encoding="utf-8") as f:
        #json.dump(best_params_per_training, f, indent=4, ensure_ascii=False)
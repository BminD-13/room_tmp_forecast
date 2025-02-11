import numpy as np
import sys
import os

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell

# ==============================
# Parameter für den GA
# ==============================
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1

# ==============================
# Initialisierung
# ==============================

def initialize_population():
    """Erstellt die initiale Population mit zufälligen Parametern."""
    population = []
    for _ in range(POPULATION_SIZE):
        individual = {
            "tau_wand": np.random.uniform(0, 1),        # 0.173 fitness kleiner 3Mrd
            "tau_speicher": np.random.uniform(0, 1),    # 0.903
            "tau_raum": np.random.uniform(0, 1),        # 0.023
        }
        population.append(individual)
    return population

# ==============================
# Fitness-Funktion
# ==============================

def fitness_function(params, dataset, real_temp):
    """
    Berechnet die Fitness eines Parametersatzes durch Simulation des Modells.
    """
    model = RaumModell(dt=1, **params)
    predicted_temp = model.run_model(dataset)
    predicted_temp = np.array(predicted_temp)
    real_temp = np.array(real_temp)

    min_length = min(len(predicted_temp), len(real_temp))
    predicted_temp = predicted_temp[:min_length]
    real_temp = real_temp[:min_length]

    # Zeitgewichtung
    time_weights = np.linspace(1, 2, min_length)
    weighted_error = np.sum((predicted_temp - real_temp) ** 2 * time_weights)

    # Gradientenvergleich
    pred_grad = np.diff(predicted_temp)
    real_grad = np.diff(real_temp)
    gradient_error = np.sum((pred_grad - real_grad) ** 2)

    # Normalisierung
    sigma_real = np.std(real_temp)
    total_error = (weighted_error + gradient_error) / (sigma_real + 1e-6)

    return -total_error  # Negativer Fehler für Maximierung

# ==============================
# Selektion
# ==============================

def selection(population, scores):
    """Wählt ein Individuum proportional zur Fitness aus (Roulette-Wheel Selektion)."""
    probabilities = np.exp(scores - np.max(scores))  # Exponentielle Normalisierung
    probabilities /= np.sum(probabilities)
    return population[np.random.choice(len(population), p=probabilities)]

# ==============================
# Crossover
# ==============================

def crossover(parent1, parent2):
    """Mittelwert-Crossover: Mischt zwei Eltern zu einem Kind."""
    child = {key: (parent1[key] + parent2[key]) / 2 for key in parent1}
    return child

# ==============================
# Mutation
# ==============================

def mutate(individual):
    """Mutiert ein Individuum durch zufällige Änderung einzelner Parameter."""
    mutated = individual.copy()
    for key in mutated:
        if np.random.rand() < MUTATION_RATE:
            if "n_" in key:
                mutated[key] = np.clip(mutated[key] + np.random.randint(-1, 2), 1, 5)
            else:
                mutated[key] *= np.random.uniform(0.9, 1.1)  # ±10% Mutation
    return mutated

# ==============================
# Genetischer Algorithmus
# ==============================

def genetic_algorithm():
    """Führt den genetischen Algorithmus aus und optimiert das Modell."""
    
    # Daten laden
    DataModule = DataModuleStatic()
    DataModule.load_csv(r"data\training\240331_Dataset_01.csv")
    start, end = DataModule.get_time_range()
    DataModule.get_timespan(start, end)
    dataset = DataModule.df
    real_temp = dataset["tmpRoom"]

    # Population initialisieren
    population = initialize_population()

    for generation in range(GENERATIONS):
        scores = [fitness_function(ind, dataset, real_temp) for ind in population]

        # Neue Population generieren
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = selection(population, scores)
            parent2 = selection(population, scores)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])

        population = new_population

        # Ausgabe des besten Individuums pro Generation
        best_fitness = max(scores)
        best_individual = population[np.argmax(scores)]
        print(f"Generation {generation+1}: Beste Fitness = {best_fitness:.4f}")
        print(f"Bester Parametersatz: {best_individual}")

    return best_individual

# ==============================
# Starten des Algorithmus
# ==============================

if __name__ == "__main__":
    best_params = genetic_algorithm()
    print("\nOptimierte Modellparameter:")
    print(best_params)

import numpy as np
import random

# ==============================
# Konfigurationsparameter
# ==============================
POPULATION_SIZE = 50   # Anzahl der Individuen in der Population
GENOME_LENGTH = 10     # Anzahl der Parameter pro Individuum
MUTATION_RATE = 0.1    # Wahrscheinlichkeit für eine Mutation
CROSSOVER_RATE = 0.7   # Wahrscheinlichkeit für Crossover
GENERATIONS = 100      # Anzahl der Generationen

PARAMETER_BEREICHE = {
    "tau_wand": (500, 2000),
    "tau_speicher": (1000, 4000),
    "tau_raum": (250, 1000),
    "n_wand": (1, 5),
    "n_speicher": (1, 3),
    "n_raum": (1, 5),
}

# ==============================
# Initialisierung der Population
# ==============================
def random_individual():
    """Erzeugt eine zufällige Parameterkombination als Individuum"""
    return {key: np.random.uniform(low, high) if "tau" in key else random.randint(low, high)
            for key, (low, high) in PARAMETER_BEREICHE.items()}

def initialize_population():
    """Erzeugt eine Population mit zufälligen Individuen"""
    return [random_individual() for _ in range(POPULATION_SIZE)]

# ==============================
# Fitness-Funktion
# ==============================
def fitness_function(model, dataset, real_temp):
    """
    Berechnet die Fitness eines Modells basierend auf der Vorhersagegenauigkeit.

    :param model: Instanz von RaumModell
    :param dataset: Eingangsgrößen für die Simulation
    :param real_temp: Gemessene Raumtemperatur (Vergleichswerte)
    :return: Fitness-Wert (negativer Fehler)
    """
    # Modell laufen lassen
    predicted_temp = model.run_model(dataset)
    predicted_temp = np.array(predicted_temp)
    real_temp = np.array(real_temp)

    # Falls unterschiedliche Längen, kürzen
    min_length = min(len(predicted_temp), len(real_temp))
    predicted_temp = predicted_temp[:min_length]
    real_temp = real_temp[:min_length]
    
    # Zeitgewichtung
    time_weights = np.linspace(1, 2, min_length)  # Linear steigend von 1 bis 2
    weighted_error = np.sum((predicted_temp - real_temp)**2 * time_weights)

    # Gradientenvergleich
    pred_grad = np.diff(predicted_temp)
    real_grad = np.diff(real_temp)
    gradient_error = np.sum((pred_grad - real_grad)**2)

    # Normalisierung
    sigma_real = np.std(real_temp)  # Standardabweichung der echten Daten
    total_error = (weighted_error + gradient_error) / (sigma_real + 1e-6)  # +1e-6 um durch 0 zu vermeiden

    return -total_error  # Negativer Fehler für Maximierung im GA

# ==============================
# Selektion (Roulette Wheel)
# ==============================
def selection(population, scores):
    """Wählt ein Individuum basierend auf Fitness-Werten (Roulette-Wheel-Selection)"""
    total_fitness = sum(scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, score in enumerate(scores):
        current += score
        if current > pick:
            return population[i]
    return population[-1]

# ==============================
# Crossover (Uniform Crossover)
# ==============================
def crossover(parent1, parent2):
    """Erzeugt ein Kind aus zwei Elternteilen (Uniform Crossover)"""
    if random.random() < CROSSOVER_RATE:
        child = np.array([random.choice([g1, g2]) for g1, g2 in zip(parent1, parent2)])
    else:
        child = parent1.copy()
    return child

# ==============================
# Mutation
# ==============================
def mutate(individual):
    """Mutiert ein Individuum mit einer gewissen Wahrscheinlichkeit"""
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] += np.random.uniform(-0.5, 0.5)  # Kleine zufällige Änderung
    return individual

# ==============================
# Genetischer Algorithmus
# ==============================
def genetic_algorithm():
    """Führt den genetischen Algorithmus aus"""
    population = initialize_population()
    
    for generation in range(GENERATIONS):
        scores = [fitness_function(ind) for ind in population]
        
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

    # Bestes Individuum zurückgeben
    return best_individual

# ==============================
# Algorithmus ausführen
# ==============================
best_solution = genetic_algorithm()
print("Beste gefundene Lösung:", best_solution)

import matplotlib.pyplot as plt
from mealpy.swarm_based import FA
from mealpy.utils.space import FloatVar
from benchmark_functions import Ackley, MartinGaddy

def run_experiment(params, problem, label):
    model = FA.OriginalFA(**params)
    best_position = model.solve(problem=problem)
    return model.history.list_current_best_fit, best_position

ackley = Ackley(n_dimensions=2)
martin_gaddy = MartinGaddy()

lower_bound, upper_bound = ackley.suggested_bounds()
dimension = 2

bounds = [FloatVar(lower_bound[0], upper_bound[0])] * dimension

problem_ackley = {
    "bounds": bounds,
    "obj_func": ackley,
    "minmax": "min",
    "name": "Ackley Function Optimization",
    "verbose": True
}

problem_mg = {
    "bounds": bounds,
    "obj_func": martin_gaddy,
    "minmax": "min",
    "name": "Martin and Gaddy Function Optimization",
    "verbose": True
}

param_sets = [
    {
        "epoch": 50,
        "pop_size": 50,
        "max_sparks": 100,
        "p_a": 0.04,
        "p_b": 0.8,
        "max_ea": 40,
        "m_sparks": 100
    },
    {
        "epoch": 50,
        "pop_size": 50,
        "max_sparks": 200,
        "p_a": 0.02,
        "p_b": 0.9,
        "max_ea": 50,
        "m_sparks": 200
    },
    {
        "epoch": 50,
        "pop_size": 100,
        "max_sparks": 150,
        "p_a": 0.03,
        "p_b": 0.85,
        "max_ea": 60,
        "m_sparks": 150
    }
]

labels = [
    "Config 1: max_sparks=100, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=100",
    "Config 2: max_sparks=200, p_a=0.02, p_b=0.9, max_ea=50, m_sparks=200",
    "Config 3: max_sparks=150, p_a=0.03, p_b=0.85, max_ea=60, m_sparks=150"
]

# Generowanie wykresów dla różnych konfiguracji
plt.figure(figsize=(14, 12))

# Wykres dla funkcji Ackley
plt.subplot(2, 1, 1)
for params, label in zip(param_sets, labels):
    best_fitness, best_position = run_experiment(params, problem_ackley, label)
    epochs = range(1, params["epoch"] + 1)
    plt.plot(epochs, best_fitness, label=label)
plt.xlabel("Epoch")
plt.ylabel("Best Fitness Value")
plt.title("Best Fitness Value per Epoch (Ackley Function)")
plt.legend()

# Wykres dla funkcji Martin and Gaddy
plt.subplot(2, 1, 2)
for params, label in zip(param_sets, labels):
    best_fitness, best_position = run_experiment(params, problem_mg, label)
    epochs = range(1, params["epoch"] + 1)
    plt.plot(epochs, best_fitness, label=label)
plt.xlabel("Epoch")
plt.ylabel("Best Fitness Value")
plt.title("Best Fitness Value per Epoch (Martin and Gaddy Function)")
plt.legend()

plt.tight_layout()
plt.show()

# Wyniki dla najlepszych konfiguracji
for params, label in zip(param_sets, labels):
    best_fitness_ackley, best_position_ackley = run_experiment(params, problem_ackley, label)
    best_fitness_mg, best_position_mg = run_experiment(params, problem_mg, label)
    print(f"Konfiguracja: {label}")
    print(f"Najlepsza pozycja (Ackley): {best_position_ackley.solution}")
    print(f"Najlepsza wartość funkcji (Ackley): {best_position_ackley.target}")
    print(f"Najlepsza pozycja (Martin and Gaddy): {best_position_mg.solution}")
    print(f"Najlepsza wartość funkcji (Martin and Gaddy): {best_position_mg.target}\n")

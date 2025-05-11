import numpy as np
import json
from collections import Counter

# --- Parameters ---
n_patients = 1_000_000

# Base prevalence (adjustable)
p_hyp = 0.30
p_dia = 0.15
p_car = 0.10

# Joint probabilities (adjustable)
p_hyp_dia = 0.12
p_hyp_car = 0.08
p_dia_car = 0.05
p_all_three = 0.03

# Comorbidity costs
costs = {
    "hypertension": 2000,
    "diabetes": 12000,
    "cardio": 30000
}

# Function to assign comorbidities using joint probabilities
def simulate_patient():
    r = np.random.rand()
    if r < p_all_three:
        return "hypertension_diabetes_cardio"
    elif r < p_all_three + p_hyp_dia:
        return "hypertension_diabetes"
    elif r < p_all_three + p_hyp_dia + p_hyp_car:
        return "hypertension_cardio"
    elif r < p_all_three + p_hyp_dia + p_hyp_car + p_dia_car:
        return "diabetes_cardio"
    else:
        # Assign singles
        flags = []
        if np.random.rand() < p_hyp: flags.append("hypertension")
        if np.random.rand() < p_dia: flags.append("diabetes")
        if np.random.rand() < p_car: flags.append("cardio")
        if not flags:
            return "none"
        return "_".join(sorted(flags))

# --- Simulation ---
clusters = [simulate_patient() for _ in range(n_patients)]
counts = Counter(clusters)

# --- Cost Analysis ---
def estimate_cost(cluster):
    if cluster == "none":
        return 0
    keys = cluster.split("_")
    return sum(costs.get(k, 0) for k in keys)

cost_by_cluster = {k: estimate_cost(k) for k in counts}
average_costs = {k: estimate_cost(k) for k in counts}
prevalence = {k: v / n_patients for k, v in counts.items()}
total_cost = sum(counts[k] * cost_by_cluster[k] for k in counts)

# --- Output ---
output = {
    "population": n_patients,
    "joint_prevalence": prevalence,
    "average_costs": average_costs,
    "total_cost": total_cost
}

# Save to JSON
with open("monte_carlo_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Simulation complete. Total economic burden: ${total_cost:,.0f}")
print("Output saved to monte_carlo_results.json")


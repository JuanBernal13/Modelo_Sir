# visualization/plot_matplotlib.py

import matplotlib.pyplot as plt
import numpy as np
from models.states import EpidemicState

def plot_epidemic(t_vals, state_counts_replicates, replicate_count):
    plt.figure(figsize=(16, 10))

    # Subgráfico 1: Estados Epidemiológicos
    plt.subplot(3, 1, 1)
    for state in [EpidemicState.S, EpidemicState.V, EpidemicState.E,
                  EpidemicState.I, EpidemicState.U, EpidemicState.R, EpidemicState.D]:
        all_counts = np.array(state_counts_replicates[state])  # (num_replicates, max_steps)
        mean = all_counts.mean(axis=0)
        lower = np.percentile(all_counts, 2.5, axis=0)
        upper = np.percentile(all_counts, 97.5, axis=0)
        plt.plot(t_vals, mean, label=f"Mean {state.name}")
        plt.fill_between(t_vals, lower, upper, alpha=0.2, label=f"95% CI {state.name}")

    plt.title("Simulación SEIURD para Klebsiella pneumoniae (Agent-Based)")
    plt.xlabel("Tiempo (pasos)")
    plt.ylabel("Número de agentes")
    plt.legend(loc='upper right')
    plt.grid(True)

def plot_icu(t_vals, icu_occupied_replicates, icu_capacity_total_replicates, replicate_count):
    plt.subplot(3, 1, 2)
    occ_array = np.array(icu_occupied_replicates)
    cap_array = np.array(icu_capacity_total_replicates)

    mean_occ = occ_array.mean(axis=0)
    lower_occ = np.percentile(occ_array, 2.5, axis=0)
    upper_occ = np.percentile(occ_array, 97.5, axis=0)

    mean_cap = cap_array.mean(axis=0)
    lower_cap = cap_array.min(axis=0)
    upper_cap = cap_array.max(axis=0)

    plt.plot(t_vals, mean_occ, label="Mean Camas Ocupadas", color='red')
    plt.fill_between(t_vals, lower_occ, upper_occ, color='red', alpha=0.2)
    plt.plot(t_vals, mean_cap, label="Capacidad Total (Mean)", color='gray', linestyle='--')
    plt.fill_between(t_vals, lower_cap, upper_cap, color='gray', alpha=0.2)

    plt.xlabel("Tiempo (pasos)")
    plt.ylabel("Ocupación / Capacidad UCI")
    plt.legend(loc='upper right')
    plt.grid(True)

def plot_vaccination_mask(t_vals, vaccination_rates_replicates, mask_usage_rates_replicates, replicate_count):
    plt.subplot(3, 1, 3)
    vacc_array = np.array(vaccination_rates_replicates)
    mean_vacc = vacc_array.mean(axis=0)
    lower_vacc = np.percentile(vacc_array, 2.5, axis=0)
    upper_vacc = np.percentile(vacc_array, 97.5, axis=0)

    mask_array = np.array(mask_usage_rates_replicates)
    mean_mask = mask_array.mean(axis=0)
    lower_mask = np.percentile(mask_array, 2.5, axis=0)
    upper_mask = np.percentile(mask_array, 97.5, axis=0)

    plt.plot(t_vals, mean_vacc*100, label="Vacunación (%)", color='green')
    plt.fill_between(t_vals, lower_vacc*100, upper_vacc*100, color='green', alpha=0.2)

    plt.plot(t_vals, mean_mask*100, label="Uso Mascarilla (%)", color='blue')
    plt.fill_between(t_vals, lower_mask*100, upper_mask*100, color='blue', alpha=0.2)

    plt.xlabel("Tiempo (pasos)")
    plt.ylabel("Porcentaje (%)")
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("simulacion_seiurd_klebsiella.png")

# visualization/plot_matplotlib.py

import matplotlib.pyplot as plt
import numpy as np
from models.states import EpidemicState

def plot_epidemic(t_vals, state_counts_replicates, replicate_count):
    plt.figure(figsize=(16, 10))

    # Subgráfica 1: Estados Epidemiológicos
    plt.subplot(3, 1, 1)
    for state in [EpidemicState.S, EpidemicState.V, EpidemicState.E,
                 EpidemicState.I, EpidemicState.U, EpidemicState.R, EpidemicState.D]:
        # Acceder directamente a las listas de conteos por réplica para el estado actual
        all_counts = np.array(state_counts_replicates[state])  # Cada elemento es una lista de conteos por paso para una réplica
        mean = all_counts.mean(axis=0)
        lower = np.percentile(all_counts, 2.5, axis=0)
        upper = np.percentile(all_counts, 97.5, axis=0)
        plt.plot(t_vals, mean, label=f"Mean {state.name}")
        plt.fill_between(t_vals, lower, upper, alpha=0.2, label=f"95% CI {state.name}")
    
    plt.title("Simulación SEIURD para \textit{Klebsiella pneumoniae} con Triage en UCI y Vacunación")
    plt.xlabel("Tiempo (pasos)")
    plt.ylabel("Número de agentes")
    plt.legend(loc='upper right')
    plt.grid(True)

def plot_icu(t_vals, icu_occupied_replicates, icu_capacity_total_replicates, replicate_count):
    plt.subplot(3, 1, 2)
    all_occupied = np.array(icu_occupied_replicates)  # Shape: (num_replicates, max_steps)
    mean_occupied = all_occupied.mean(axis=0)
    lower_occupied = np.percentile(all_occupied, 2.5, axis=0)
    upper_occupied = np.percentile(all_occupied, 97.5, axis=0)

    all_capacity = np.array(icu_capacity_total_replicates)  # Shape: (num_replicates, max_steps)
    mean_capacity = all_capacity.mean(axis=0)
    lower_capacity = all_capacity.min(axis=0)
    upper_capacity = all_capacity.max(axis=0)

    plt.plot(t_vals, mean_occupied, label="Mean Camas Ocupadas", color='red')
    plt.fill_between(t_vals, lower_occupied, upper_occupied,
                     color='red', alpha=0.2, label='95% CI Camas Ocupadas')
    plt.plot(t_vals, mean_capacity, label="Mean Capacidad Total UCI", color='gray', linestyle='--')
    plt.fill_between(t_vals, lower_capacity, upper_capacity,
                     color='gray', alpha=0.2, label='Capacidad UCI Rango')

    plt.xlabel("Tiempo (pasos)")
    plt.ylabel("Camas UCI")
    plt.legend(loc='upper right')
    plt.grid(True)

def plot_vaccination_mask(t_vals, vaccination_rates_replicates, mask_usage_rates_replicates, replicate_count):
    plt.subplot(3, 1, 3)
    # Vacunación
    all_vaccinations = np.array(vaccination_rates_replicates)  # Shape: (num_replicates, max_steps)
    mean_vaccination = all_vaccinations.mean(axis=0)
    lower_vaccination = np.percentile(all_vaccinations, 2.5, axis=0)
    upper_vaccination = np.percentile(all_vaccinations, 97.5, axis=0)
    plt.plot(t_vals, mean_vaccination * 100, label="Mean Tasa Vacunación (%)", color='green')
    plt.fill_between(t_vals, lower_vaccination * 100, upper_vaccination * 100,
                     color='green', alpha=0.2, label='95% CI Vacunación')

    # Uso de mascarillas
    all_masks = np.array(mask_usage_rates_replicates)  # Shape: (num_replicates, max_steps)
    mean_masks = all_masks.mean(axis=0)
    lower_masks = np.percentile(all_masks, 2.5, axis=0)
    upper_masks = np.percentile(all_masks, 97.5, axis=0)
    plt.plot(t_vals, mean_masks * 100, label="Mean Uso Mascarillas (%)", color='blue')
    plt.fill_between(t_vals, lower_masks * 100, upper_masks * 100,
                     color='blue', alpha=0.2, label='95% CI Mascarillas')

    plt.xlabel("Tiempo (pasos)")
    plt.ylabel("Porcentaje (%)")
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("simulacion_seiurd_klebsiella.png")

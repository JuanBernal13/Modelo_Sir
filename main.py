# main.py

import random
import uuid
from collections import defaultdict
from services.simulation_service import SimulationService
from models.agent import HealthcareWorker, Patient
from models.states import EpidemicState
from visualization.plot_matplotlib import plot_epidemic, plot_icu, plot_vaccination_mask
from visualization.plot_plotly import plot_spatial, plot_interactive
import numpy as np
from scipy.stats import beta, gamma as gamma_dist  # <-- Importación corregida

def sample_parameters():
    """
    Muestra de parámetros basados en distribuciones de probabilidad.
    Estos parámetros están adaptados a las características de K. pneumoniae.
    """
    # Usando distribuciones Beta para tasas de infección y mortalidad
    beta_h = beta.rvs(a=2, b=5)  # Parámetro de trabajadores de salud
    beta_g = beta.rvs(a=2, b=5)  # Parámetro de pacientes
    sigma = gamma_dist.rvs(a=2, scale=1/0.2)  # Tasa de exposición a infectado (media=5)
    alpha = beta.rvs(a=2, b=5)  # Probabilidad de ingresar a UCI
    gamma = gamma_dist.rvs(a=2, scale=1/0.142)  # Tasa de recuperación (media≈7)
    gamma_u = gamma_dist.rvs(a=2, scale=1/0.071)  # Tasa de recuperación en UCI (media≈14)
    mu_i = beta.rvs(a=2, b=98)  # Tasa de mortalidad en estado I (media≈0.02)
    mu_u = beta.rvs(a=2, b=32)  # Tasa de mortalidad en UCI (media≈0.06)
    mu_nat = 1/(80*365)  # Tasa de mortalidad natural (≈3.42e-05)
    
    return beta_h, beta_g, sigma, alpha, gamma, gamma_u, mu_i, mu_u, mu_nat

def run_simulation(replicate_id, max_steps, initial_patients, initial_workers):
    """
    Ejecuta una única réplica de la simulación.
    """
    # Parámetros de la simulación
    width, height = 20, 20  # Tamaño de la grilla
    n_icu_rows = 3          # Filas designadas como UCI
    icu_capacity = 5        # Capacidad de camas por celda UCI

    service = SimulationService(width=width, height=height, n_icu_rows=n_icu_rows, icu_capacity=icu_capacity)

    # Muestreo de parámetros para esta réplica
    beta_h, beta_g, sigma, alpha, gamma, gamma_u, mu_i, mu_u, mu_nat = sample_parameters()

    # Crear trabajadores de la salud con posibilidad de estar vacunados, usar mascarillas y resistencia
    for i in range(initial_workers):
        cell = service.get_random_cell()
        unique_id = f"HW-{uuid.uuid4()}"
        # Decidir si el trabajador está vacunado
        vaccinated = random.random() < 0.8  # 80% vacunados
        # Decidir si usa mascarilla
        wears_mask = random.random() < 0.9  # 90% usan mascarilla
        # Decidir si es resistente a antibióticos
        resistant = random.random() < 0.2  # 20% resistentes
        hw = HealthcareWorker(
            unique_id, cell,
            beta_h, sigma, alpha, gamma, gamma_u,
            mu_i, mu_u, mu_nat,
            age=random.randint(25, 65),
            use_probabilistic=True,
            vaccinated=vaccinated,
            wears_mask=wears_mask,
            resistant=resistant
        )
        service.add_worker(hw)

    # Crear pacientes iniciales
    for i in range(initial_patients):
        cell = service.get_random_cell()
        pid = str(uuid.uuid4())
        severity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        vaccinated = random.random() < 0.3  # 30% vacunados
        wears_mask = random.random() < 0.7  # 70% usan mascarilla
        resistant = random.random() < 0.3  # 30% resistentes
        p = Patient(
            pid, cell,
            beta_g, sigma, alpha, gamma, gamma_u,
            mu_i, mu_u, mu_nat,
            severity=severity,
            age=random.randint(30, 90),
            use_probabilistic=True,
            vaccinated=vaccinated,
            wears_mask=wears_mask,
            resistant=resistant
        )
        service.add_patient(p)

    # Forzar que algunos arranquen en estado I
    initial_infectious = min(5, len(service.patients))
    for i in range(initial_infectious):
        service.patients[i].state = EpidemicState.I

    # Listas para estadísticos
    t_vals = []
    state_counts = defaultdict(lambda: defaultdict(list))  # {state: {replica: counts}}
    icu_occupied = []
    icu_capacity_total = []
    vaccination_rates = []
    mask_usage_rates = []

    # Preparar datos espaciales para visualización
    spatial_data = defaultdict(lambda: defaultdict(list))  # {step: {state: [(x, y), ...]}}

    for step in range(1, max_steps+1):
        service.step(
            current_step=step,
            beta_h=beta_h, beta_g=beta_g, sigma=sigma, alpha=alpha, gamma=gamma, gamma_u=gamma_u,
            mu_i=mu_i, mu_u=mu_u, mu_nat=mu_nat
        )

        # Guardar conteos
        counts = service.get_state_counts()
        for state in EpidemicState:
            state_counts[state][replicate_id].append(counts[state])
        t_vals.append(step)

        # Guardar ocupación UCI
        occupied, capacity_total = service.get_icu_occupancy()
        icu_occupied.append(occupied)
        icu_capacity_total.append(capacity_total)

        # Guardar tasa de vacunación y uso de mascarillas
        vaccination_rates.append(service.get_vaccination_rate())
        total_agents = len(service.workers) + len(service.patients)
        total_mask = sum(1 for w in service.workers if w.wears_mask) + \
                     sum(1 for p in service.patients if p.wears_mask)
        mask_usage_rates.append(total_mask / total_agents if total_agents > 0 else 0)

        # Guardar datos espaciales
        for agent in service.workers + service.patients:
            spatial_data[step][agent.state].append((agent.current_cell.x, agent.current_cell.y))

        if step % 50 == 0 or step == max_steps:
            service.logger.info(f"Replica {replicate_id} - Step={step} => Infectious={service.get_total_infectious()}, "
                                f"Deaths={service.deaths_count}, Recoveries={service.recoveries_count}, "
                                f"TotalP={len(service.patients)}, UCI Occupied={occupied}/{capacity_total}, "
                                f"Vaccination Rate={service.get_vaccination_rate():.2%}, "
                                f"Mask Usage={mask_usage_rates[-1]:.2%}")

    # Retornar los resultados de esta réplica
    return {
        't_vals': t_vals,
        'state_counts': state_counts,
        'icu_occupied': icu_occupied,
        'icu_capacity_total': icu_capacity_total,
        'vaccination_rates': vaccination_rates,
        'mask_usage_rates': mask_usage_rates,
        'spatial_data': spatial_data
    }

def main():
    # Parámetros de la simulación
    width, height = 20, 20  # Tamaño de la grilla
    n_icu_rows = 3          # Filas designadas como UCI
    icu_capacity = 5        # Capacidad de camas por celda UCI

    # Configuración de la simulación
    num_replicates = 5  # Número de réplicas
    max_steps = 50
    initial_patients = 50
    initial_workers = 10

    # Listas para almacenar resultados de todas las réplicas
    all_state_counts = defaultdict(list)  # {state: [list of counts per replica]}
    all_icu_occupied = []
    all_icu_capacity_total = []
    all_vaccination_rates = []
    all_mask_usage_rates = []
    all_t_vals = []
    all_spatial_data = []

    for replicate in range(1, num_replicates + 1):
        replicate_id = f"Replicate-{replicate}"
        print(f"Iniciando {replicate_id}...")
        results = run_simulation(replicate_id, max_steps, initial_patients, initial_workers)

        # Guardar resultados
        all_t_vals = results['t_vals']  # Asumiendo que todas las réplicas tienen el mismo número de pasos
        for state in EpidemicState:
            all_state_counts[state].append(results['state_counts'][state][replicate_id])
        all_icu_occupied.append(results['icu_occupied'])
        all_icu_capacity_total.append(results['icu_capacity_total'])
        all_vaccination_rates.append(results['vaccination_rates'])
        all_mask_usage_rates.append(results['mask_usage_rates'])
        all_spatial_data.append(results['spatial_data'])

    # Cálculo de estadísticas (media y 95% intervalos de confianza)
    state_stats = {}
    for state in EpidemicState:
        state_array = np.array(all_state_counts[state])  # Shape: (num_replicates, max_steps)
        mean = state_array.mean(axis=0)
        lower = np.percentile(state_array, 2.5, axis=0)
        upper = np.percentile(state_array, 97.5, axis=0)
        state_stats[state] = {'mean': mean, 'lower': lower, 'upper': upper}

    icu_occupied_array = np.array(all_icu_occupied)  # Shape: (num_replicates, max_steps)
    icu_capacity_total_array = np.array(all_icu_capacity_total)  # Shape: (num_replicates, max_steps)
    mean_icu_occupied = icu_occupied_array.mean(axis=0)
    lower_icu_occupied = np.percentile(icu_occupied_array, 2.5, axis=0)
    upper_icu_occupied = np.percentile(icu_occupied_array, 97.5, axis=0)

    mean_icu_capacity = icu_capacity_total_array.mean(axis=0)
    lower_icu_capacity = icu_capacity_total_array.min(axis=0)
    upper_icu_capacity = icu_capacity_total_array.max(axis=0)

    vaccination_rates_array = np.array(all_vaccination_rates)  # Shape: (num_replicates, max_steps)
    mean_vaccination = vaccination_rates_array.mean(axis=0)
    lower_vaccination = np.percentile(vaccination_rates_array, 2.5, axis=0)
    upper_vaccination = np.percentile(vaccination_rates_array, 97.5, axis=0)

    mask_usage_rates_array = np.array(all_mask_usage_rates)  # Shape: (num_replicates, max_steps)
    mean_masks = mask_usage_rates_array.mean(axis=0)
    lower_masks = np.percentile(mask_usage_rates_array, 2.5, axis=0)
    upper_masks = np.percentile(mask_usage_rates_array, 97.5, axis=0)

    # Preparar datos para visualización
    # Convertir state_stats a un formato adecuado para plot_epidemic
    state_counts_replicates = {}
    for state in EpidemicState:
        state_counts_replicates[state] = all_state_counts[state]  # Cada estado mapea a una lista de conteos por réplica

    # Visualización con Matplotlib
    plot_epidemic(all_t_vals, state_counts_replicates, num_replicates)
    plot_icu(all_t_vals, all_icu_occupied, all_icu_capacity_total, num_replicates)
    plot_vaccination_mask(all_t_vals, all_vaccination_rates, all_mask_usage_rates, num_replicates)

    # Visualización Espacial con Plotly
    # Seleccionar una réplica para visualización espacial
    replicate_to_visualize = 0  # Puedes cambiar esto para visualizar diferentes réplicas
    spatial_data = all_spatial_data[replicate_to_visualize]
    steps_to_visualize = [0, max_steps//4, max_steps//2, 3*max_steps//4, max_steps]
    plot_spatial(spatial_data, steps_to_visualize, width, height)

    # Visualización Interactiva con Plotly para una réplica específica
    plot_interactive(spatial_data, max_steps, width, height)

    # Guardar Resultados en CSV (Opcional)
    import csv

    with open("resultados_simulacion_klebsiella.csv", "w", newline='') as csvfile:
        fieldnames = ['Step', 'Susceptible_Mean', 'Susceptible_Lower', 'Susceptible_Upper',
                      'Vacunado_Mean', 'Vacunado_Lower', 'Vacunado_Upper',
                      'Expuesto_Mean', 'Expuesto_Lower', 'Expuesto_Upper',
                      'Infectado_Mean', 'Infectado_Lower', 'Infectado_Upper',
                      'En_UCI_Mean', 'En_UCI_Lower', 'En_UCI_Upper',
                      'Recuperado_Mean', 'Recuperado_Lower', 'Recuperado_Upper',
                      'Fallecido_Mean', 'Fallecido_Lower', 'Fallecido_Upper',
                      'Camas_Ocupadas_Mean', 'Camas_Ocupadas_Lower', 'Camas_Ocupadas_Upper',
                      'Capacidad_UCI_Mean', 'Capacidad_UCI_Lower', 'Capacidad_UCI_Upper',
                      'Tasa_Vacunacion_Mean', 'Tasa_Vacunacion_Lower', 'Tasa_Vacunacion_Upper',
                      'Uso_Mascarillas_Mean', 'Uso_Mascarillas_Lower', 'Uso_Mascarillas_Upper']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_steps):
            writer.writerow({
                'Step': all_t_vals[i],
                'Susceptible_Mean': state_stats[EpidemicState.S]['mean'][i],
                'Susceptible_Lower': state_stats[EpidemicState.S]['lower'][i],
                'Susceptible_Upper': state_stats[EpidemicState.S]['upper'][i],
                'Vacunado_Mean': state_stats[EpidemicState.V]['mean'][i],
                'Vacunado_Lower': state_stats[EpidemicState.V]['lower'][i],
                'Vacunado_Upper': state_stats[EpidemicState.V]['upper'][i],
                'Expuesto_Mean': state_stats[EpidemicState.E]['mean'][i],
                'Expuesto_Lower': state_stats[EpidemicState.E]['lower'][i],
                'Expuesto_Upper': state_stats[EpidemicState.E]['upper'][i],
                'Infectado_Mean': state_stats[EpidemicState.I]['mean'][i],
                'Infectado_Lower': state_stats[EpidemicState.I]['lower'][i],
                'Infectado_Upper': state_stats[EpidemicState.I]['upper'][i],
                'En_UCI_Mean': state_stats[EpidemicState.U]['mean'][i],
                'En_UCI_Lower': state_stats[EpidemicState.U]['lower'][i],
                'En_UCI_Upper': state_stats[EpidemicState.U]['upper'][i],
                'Recuperado_Mean': state_stats[EpidemicState.R]['mean'][i],
                'Recuperado_Lower': state_stats[EpidemicState.R]['lower'][i],
                'Recuperado_Upper': state_stats[EpidemicState.R]['upper'][i],
                'Fallecido_Mean': state_stats[EpidemicState.D]['mean'][i],
                'Fallecido_Lower': state_stats[EpidemicState.D]['lower'][i],
                'Fallecido_Upper': state_stats[EpidemicState.D]['upper'][i],
                'Camas_Ocupadas_Mean': mean_icu_occupied[i],
                'Camas_Ocupadas_Lower': lower_icu_occupied[i],
                'Camas_Ocupadas_Upper': upper_icu_occupied[i],
                'Capacidad_UCI_Mean': mean_icu_capacity[i],
                'Capacidad_UCI_Lower': lower_icu_capacity[i],
                'Capacidad_UCI_Upper': upper_icu_capacity[i],
                'Tasa_Vacunacion_Mean': mean_vaccination[i] * 100,
                'Tasa_Vacunacion_Lower': lower_vaccination[i] * 100,
                'Tasa_Vacunacion_Upper': upper_vaccination[i] * 100,
                'Uso_Mascarillas_Mean': mean_masks[i] * 100,
                'Uso_Mascarillas_Lower': lower_masks[i] * 100,
                'Uso_Mascarillas_Upper': upper_masks[i] * 100
            })

    service.logger.info("Resultados guardados en 'resultados_simulacion_klebsiella.csv'.")

if __name__ == "__main__":
    main()

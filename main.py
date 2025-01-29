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
from scipy.stats import beta, gamma as gamma_dist  

def sample_parameters():
    """
   Parametros ajustados a distribuciones de probabilidad.
    """
    # Tasas distintas para trabajadores de la salud y pacientes
    beta_h = beta.rvs(a=2, b=5)                # Tasa de transmisión base para trabajadores
    beta_g = beta.rvs(a=2, b=5)                # Tasa de transmisión base para pacientes
    sigma = gamma_dist.rvs(a=2, scale=1/0.2)   # Tasa de progresión E->I (media ~ 5 días)
    alpha = beta.rvs(a=2, b=5)                 # Tasa de ingreso a UCI
    gamma_ = gamma_dist.rvs(a=2, scale=1/0.142) # Tasa de recuperación I->R (media ~ 7 días)
    gamma_u = gamma_dist.rvs(a=2, scale=1/0.071)# Tasa de recuperación en UCI (media ~ 14 días)
    mu_i = beta.rvs(a=2, b=98)                 # Mortalidad en I (media ~ 0.02)
    mu_u = beta.rvs(a=2, b=32)                 # Mortalidad en UCI (media ~ 0.06)
    mu_nat = 1/(80*365)                        # Mortalidad natural (~3.42e-05)

    # ### ADAPTACIÓN SEIURD: Tasa de re-susceptibilización (R->S)
    p_resus = beta.rvs(a=2, b=200) 

    return beta_h, beta_g, sigma, alpha, gamma_, gamma_u, mu_i, mu_u, mu_nat, p_resus

def run_simulation(replicate_id, max_steps, initial_patients, initial_workers):
    """
    Ejecuta una única réplica de la simulación.
    """
    # Parámetros de la grilla
    width, height = 20, 20
    n_icu_rows = 3
    icu_capacity = 5

    service = SimulationService(width=width, height=height,
                                n_icu_rows=n_icu_rows,
                                icu_capacity=icu_capacity)

    # Muestreo de parámetros para esta réplica
    (beta_h, beta_g, sigma, alpha, gamma_, gamma_u,
     mu_i, mu_u, mu_nat, p_resus) = sample_parameters()

    # Crear trabajadores de la salud
    for i in range(initial_workers):
        cell = service.get_random_cell()
        unique_id = f"HW-{uuid.uuid4()}"
        vaccinated = random.random() < 0.8  # 80% vacunados
        wears_mask = random.random() < 0.9  # 90% usan mascarilla
        resistant = random.random() < 0.2   # 20% resistentes
        hw = HealthcareWorker(
            unique_id, cell,
            beta_h, sigma, alpha, gamma_, gamma_u,
            mu_i, mu_u, mu_nat,
            p_resus=p_resus,             # ### ADAPTACIÓN SEIURD
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
        resistant = random.random() < 0.3   # 30% resistentes
        p = Patient(
            pid, cell,
            beta_g, sigma, alpha, gamma_, gamma_u,
            mu_i, mu_u, mu_nat,
            p_resus=p_resus,             
            severity=severity,
            age=random.randint(30, 90),
            use_probabilistic=True,
            vaccinated=vaccinated,
            wears_mask=wears_mask,
            resistant=resistant
        )
        service.add_patient(p)

    # Inicio estado I
    initial_infectious = min(5, len(service.patients))
    for i in range(initial_infectious):
        service.patients[i].state = EpidemicState.I

    # Listas para estadísticos
    t_vals = []
    state_counts = defaultdict(lambda: defaultdict(list))
    icu_occupied = []
    icu_capacity_total = []
    vaccination_rates = []
    mask_usage_rates = []
    spatial_data = defaultdict(lambda: defaultdict(list))

    for step in range(1, max_steps+1):
        service.step(
            current_step=step,
            beta_h=beta_h, beta_g=beta_g, sigma=sigma, alpha=alpha,
            gamma=gamma_, gamma_u=gamma_u,
            mu_i=mu_i, mu_u=mu_u, mu_nat=mu_nat
        )

        # Guardar conteos
        counts = service.get_state_counts()
        for st in EpidemicState:
            state_counts[st][replicate_id].append(counts[st])
        t_vals.append(step)

        # Guardar ocupación UCI
        occ, cap_total = service.get_icu_occupancy()
        icu_occupied.append(occ)
        icu_capacity_total.append(cap_total)

        # Guardar tasa de vacunación y uso de tapabocas
        vaccination_rates.append(service.get_vaccination_rate())
        total_agents = len(service.workers) + len(service.patients)
        total_masked = sum(1 for w in service.workers if w.wears_mask) \
                     + sum(1 for p in service.patients if p.wears_mask)
        mask_usage_rates.append(total_masked / total_agents if total_agents > 0 else 0)

        # Guardar datos espaciales
        for ag in service.workers + service.patients:
            spatial_data[step][ag.state].append((ag.current_cell.x, ag.current_cell.y))

        if step % 50 == 0 or step == max_steps:
            service.logger.info(
                f"Replica {replicate_id} - Step={step} => "
                f"Infectious={service.get_total_infectious()}, "
                f"Deaths={service.deaths_count}, Recoveries={service.recoveries_count}, "
                f"TotalP={len(service.patients)}, UCI Occupied={occ}/{cap_total}, "
                f"Vaccination Rate={service.get_vaccination_rate():.2%}, "
                f"Mask Usage={mask_usage_rates[-1]:.2%}"
            )

    # Retornar resultados para posterior agregación
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
    num_replicates = 5
    max_steps = 50
    initial_patients = 50
    initial_workers = 10

    all_state_counts = defaultdict(list)
    all_icu_occupied = []
    all_icu_capacity_total = []
    all_vaccination_rates = []
    all_mask_usage_rates = []
    all_t_vals = []
    all_spatial_data = []

    for replicate in range(1, num_replicates+1):
        replicate_id = f"Replicate-{replicate}"
        print(f"Iniciando {replicate_id}...")
        results = run_simulation(replicate_id, max_steps, initial_patients, initial_workers)

        # Guardar resultados
        all_t_vals = results['t_vals']
        for st in EpidemicState:
            all_state_counts[st].append(results['state_counts'][st][replicate_id])
        all_icu_occupied.append(results['icu_occupied'])
        all_icu_capacity_total.append(results['icu_capacity_total'])
        all_vaccination_rates.append(results['vaccination_rates'])
        all_mask_usage_rates.append(results['mask_usage_rates'])
        all_spatial_data.append(results['spatial_data'])

    # Cálculo de estadísticas
    state_stats = {}
    for st in EpidemicState:
        arr = np.array(all_state_counts[st])  # (num_replicates, max_steps)
        mean = arr.mean(axis=0)
        lower = np.percentile(arr, 2.5, axis=0)
        upper = np.percentile(arr, 97.5, axis=0)
        state_stats[st] = {'mean': mean, 'lower': lower, 'upper': upper}

    icu_occ_arr = np.array(all_icu_occupied)
    mean_icu_occ = icu_occ_arr.mean(axis=0)
    lower_icu_occ = np.percentile(icu_occ_arr, 2.5, axis=0)
    upper_icu_occ = np.percentile(icu_occ_arr, 97.5, axis=0)

    icu_cap_arr = np.array(all_icu_capacity_total)
    mean_icu_cap = icu_cap_arr.mean(axis=0)
    lower_icu_cap = icu_cap_arr.min(axis=0)
    upper_icu_cap = icu_cap_arr.max(axis=0)

    vacc_arr = np.array(all_vaccination_rates)
    mean_vacc = vacc_arr.mean(axis=0)
    lower_vacc = np.percentile(vacc_arr, 2.5, axis=0)
    upper_vacc = np.percentile(vacc_arr, 97.5, axis=0)

    mask_arr = np.array(all_mask_usage_rates)
    mean_mask = mask_arr.mean(axis=0)
    lower_mask = np.percentile(mask_arr, 2.5, axis=0)
    upper_mask = np.percentile(mask_arr, 97.5, axis=0)

    # Preparar datos para visualización
    state_counts_replicates = {}
    for st in EpidemicState:
        state_counts_replicates[st] = all_state_counts[st]

    # Visualizaciones Matplotlib
    plot_epidemic(all_t_vals, state_counts_replicates, num_replicates)
    plot_icu(all_t_vals, all_icu_occupied, all_icu_capacity_total, num_replicates)
    plot_vaccination_mask(all_t_vals, all_vaccination_rates, all_mask_usage_rates, num_replicates)

    replicate_to_visualize = 0
    spatial_data = all_spatial_data[replicate_to_visualize]
    steps_to_visualize = [0, max_steps//4, max_steps//2, 3*max_steps//4, max_steps]
    plot_spatial(spatial_data, steps_to_visualize, 20, 20)

    plot_interactive(spatial_data, max_steps, 20, 20)

    import csv
    with open("resultados_simulacion_klebsiella.csv", "w", newline='') as csvfile:
        fieldnames = [
            'Step',
            'Susceptible_Mean', 'Susceptible_Lower', 'Susceptible_Upper',
            'Vacunado_Mean', 'Vacunado_Lower', 'Vacunado_Upper',
            'Expuesto_Mean', 'Expuesto_Lower', 'Expuesto_Upper',
            'Infectado_Mean', 'Infectado_Lower', 'Infectado_Upper',
            'En_UCI_Mean', 'En_UCI_Lower', 'En_UCI_Upper',
            'Recuperado_Mean', 'Recuperado_Lower', 'Recuperado_Upper',
            'Fallecido_Mean', 'Fallecido_Lower', 'Fallecido_Upper',
            'Camas_Ocupadas_Mean', 'Camas_Ocupadas_Lower', 'Camas_Ocupadas_Upper',
            'Capacidad_UCI_Mean', 'Capacidad_UCI_Lower', 'Capacidad_UCI_Upper',
            'Tasa_Vacunacion_Mean', 'Tasa_Vacunacion_Lower', 'Tasa_Vacunacion_Upper',
            'Uso_Tapabocas_Mean', 'Uso_Tapabocas_Lower', 'Uso_Tapabocas_Upper'
        ]
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
                'Camas_Ocupadas_Mean': mean_icu_occ[i],
                'Camas_Ocupadas_Lower': lower_icu_occ[i],
                'Camas_Ocupadas_Upper': upper_icu_occ[i],
                'Capacidad_UCI_Mean': mean_icu_cap[i],
                'Capacidad_UCI_Lower': lower_icu_cap[i],
                'Capacidad_UCI_Upper': upper_icu_cap[i],
                'Tasa_Vacunacion_Mean': mean_vacc[i] * 100,
                'Tasa_Vacunacion_Lower': lower_vacc[i] * 100,
                'Tasa_Vacunacion_Upper': upper_vacc[i] * 100,
                'Uso_Tapabocas_Mean': mean_mask[i] * 100,
                'Uso_Tapabocas_Lower': lower_mask[i] * 100,
                'Uso_Tapabocas_Upper': upper_mask[i] * 100
            })

if __name__ == "__main__":
    main()

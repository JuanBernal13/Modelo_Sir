# services/simulation_service.py

import random
import uuid
from collections import defaultdict
from models.cell import Cell
from models.agent import Agent, HealthcareWorker, Patient
from models.states import EpidemicState
from utils.logger import get_logger
import numpy as np
from scipy.stats import beta, gamma as gamma_dist

class SimulationService:
    def __init__(self, width, height, n_icu_rows=2, icu_capacity=3):
        """
        Crea una grilla de tamaño width x height.
        Las primeras 'n_icu_rows' filas se marcan como UCI.
        """
        self.width = width
        self.height = height
        self.grid = []
        for x in range(width):
            col = []
            for y in range(height):
                is_icu = (y < n_icu_rows)
                c = Cell(x, y, is_icu=is_icu, icu_capacity=icu_capacity)
                col.append(c)
            self.grid.append(col)

        self.workers = []
        self.patients = []
        self.deaths_count = 0
        self.recoveries_count = 0

        # Parámetros de llegada/entrada de nuevos pacientes
        self.arrival_rate = 0.6  

        self.icu_waiting_queue = []

        self.vaccinated_count = 0
        self.mask_wearers = 0

        self.logger = get_logger()

    def add_worker(self, worker):
        self.workers.append(worker)
        if worker.vaccinated:
            self.vaccinated_count += 1
        if worker.wears_mask:
            self.mask_wearers += 1

    def add_patient(self, patient):
        self.patients.append(patient)
        if patient.vaccinated:
            self.vaccinated_count += 1
        if patient.wears_mask:
            self.mask_wearers += 1

    def remove_patient(self, patient):
        if patient in self.patients:
            self.patients.remove(patient)
        cell = patient.current_cell
        if cell:
            cell.remove_agent(patient)

    def move_agent(self, agent):
        if not agent.current_cell:
            return
        cx = agent.current_cell.x
        cy = agent.current_cell.y
        nx = cx + random.randint(-1, 1)
        ny = cy + random.randint(-1, 1)
        nx = max(0, min(nx, self.width - 1))
        ny = max(0, min(ny, self.height - 1))
        new_cell = self.grid[nx][ny]
        agent.set_current_cell(new_cell)
        self.logger.debug(f"{agent.unique_id} se movió a la celda ({nx}, {ny}).")

    def get_random_cell(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        return self.grid[x][y]

    def get_total_infectious(self):
        """
        Infectious = I + U
        """
        inf_w = sum(1 for w in self.workers if w.state in [EpidemicState.I, EpidemicState.U])
        inf_p = sum(1 for p in self.patients if p.state in [EpidemicState.I, EpidemicState.U])
        return inf_w + inf_p

    def get_total_population(self):
        """
        Todos los agentes vivos (estado != D).
        """
        living_w = sum(1 for w in self.workers if w.state != EpidemicState.D)
        living_p = sum(1 for p in self.patients if p.state != EpidemicState.D)
        return living_w + living_p

    def register_death(self, agent: Agent):
        self.deaths_count += 1

    def register_recovery_from_icu(self, agent: Agent):
        self.recoveries_count += 1

    def get_state_counts(self):
        """
        Retorna cuántos agentes (workers + patients) hay en cada estado.
        """
        states = {st: 0 for st in EpidemicState}
        for w in self.workers:
            states[w.state] += 1
        for p in self.patients:
            states[p.state] += 1
        return states

    def spawn_patient_if_needed(self, beta_g, sigma, alpha, gamma_, gamma_u, mu_i, mu_u, mu_nat):
        """
        Con prob arrival_rate, entra un nuevo paciente S.
        """
        if random.random() < self.arrival_rate:
            cell = self.get_random_cell()
            pid = str(uuid.uuid4())
            severity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            vaccinated = random.random() < 0.3
            wears_mask = random.random() < 0.7
            resistant = random.random() < 0.3
            # NOTA: p_resus podría ser fijo o también sampleado aquí, si así se desea
            p = Patient(pid, cell,
                        beta_g, sigma, alpha, gamma_, gamma_u,
                        mu_i, mu_u, mu_nat,
                        p_resus=0.001,  # Se podría parametrizar
                        severity=severity,
                        age=None,
                        use_probabilistic=True,
                        vaccinated=vaccinated,
                        wears_mask=wears_mask,
                        resistant=resistant)
            self.add_patient(p)
            self.logger.debug(f"Nuevo paciente {pid} ingresó en la celda ({cell.x}, {cell.y}).")

    def request_icu(self, agent: Agent):
        """
        Añade al agente a la cola de UCI según su severidad.
        """
        priority = -getattr(agent, 'severity', 1)
        self.icu_waiting_queue.append((priority, agent))
        self.icu_waiting_queue.sort()
        self.logger.debug(f"{agent.unique_id} solicitado UCI con prioridad {priority}.")
        self.assign_icu_beds()

    def assign_icu_beds(self):
        """
        Asigna camas UCI a los agentes en cola según prioridad.
        """
        for priority, agent in list(self.icu_waiting_queue):
            if agent.state != EpidemicState.I:
                self.icu_waiting_queue.remove((priority, agent))
                continue
            bed_found = False
            for col in self.grid:
                for cell in col:
                    if cell.is_icu and cell.has_free_bed():
                        cell.occupy_bed()
                        agent.set_current_cell(cell)
                        agent.state = EpidemicState.U
                        self.icu_waiting_queue.remove((priority, agent))
                        bed_found = True
                        self.logger.debug(f"{agent.unique_id} asignado a UCI en ({cell.x}, {cell.y}).")
                        break
                if bed_found:
                    break
            if not bed_found:
                break

    def get_icu_occupancy(self):
        occupied = sum(cell.occupied_beds for col in self.grid for cell in col if cell.is_icu)
        capacity = sum(cell.icu_capacity for col in self.grid for cell in col if cell.is_icu)
        return occupied, capacity

    def get_vaccination_rate(self):
        total_population = self.get_total_population()
        if total_population == 0:
            return 0
        total_vaccinated = sum(1 for w in self.workers if w.vaccinated and w.state != EpidemicState.D) \
                         + sum(1 for p in self.patients if p.vaccinated and p.state != EpidemicState.D)
        return total_vaccinated / total_population

    def step(self, current_step, beta_h, beta_g, sigma, alpha, gamma, gamma_u, mu_i, mu_u, mu_nat):
        """
        Un paso de simulación:
          1. Posible arribo de un nuevo paciente
          2. step() de cada agente
          3. Asignar camas UCI
        """
        # 1) Llega un paciente con prob arrival_rate
        self.spawn_patient_if_needed(beta_g, sigma, alpha, gamma, gamma_u, mu_i, mu_u, mu_nat)

        # 2) Step de cada agente
        ws = list(self.workers)
        ps = list(self.patients)

        for w in ws:
            w.step(current_step, self)
        for p in ps:
            p.step(current_step, self)

        # 3) Asignar camas UCI si posible
        self.assign_icu_beds()

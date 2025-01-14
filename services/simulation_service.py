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

        # Parámetros de llegada/entrada
        self.arrival_rate = 0.6  # ~ 60% de prob. de que llegue un paciente por step

        # Cola de espera para UCI (prioridad alta para severidad alta)
        self.icu_waiting_queue = []

        # Estadísticas adicionales
        self.vaccinated_count = 0
        self.mask_wearers = 0

        # Logger
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
        """
        Remueve el paciente de la lista y de la celda donde esté.
        Simulando un alta voluntaria o un traspaso a otra área.
        """
        if patient in self.patients:
            self.patients.remove(patient)
        cell = patient.current_cell
        if cell:
            cell.remove_agent(patient)

    def move_agent(self, agent):
        """
        Movimiento aleatorio en la vecindad (Moore).
        """
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
        self.logger.debug(f"{agent.unique_id} se ha movido a la celda ({nx}, {ny}).")

    def get_random_cell(self):
        x = random.randint(0, self.width-1)
        y = random.randint(0, self.height-1)
        return self.grid[x][y]

    def get_total_infectious(self):
        """
        Cuenta cuántos agentes están en estado I o U.
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

    def spawn_patient_if_needed(self, 
                                beta_g, sigma, alpha, gamma, gamma_u,
                                mu_i, mu_u, mu_nat):
        """
        Con prob arrival_rate, ingresa un nuevo paciente
        en una celda aleatoria. Se asume que su estado es S inicialmente.
        """
        if random.random() < self.arrival_rate:
            cell = self.get_random_cell()
            pid = str(uuid.uuid4())  # Uso de UUID para IDs únicos
            # Asignar severidad aleatoria
            severity = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            # Decidir si el paciente está vacunado
            vaccinated = random.random() < 0.3  # 30% vacunados
            # Decidir si usa mascarilla
            wears_mask = random.random() < 0.7  # 70% usan mascarilla
            # Decidir si es resistente a antibióticos
            resistant = random.random() < 0.3  # 30% resistentes
            p = Patient(pid, cell,
                        beta_g, sigma, alpha, gamma, gamma_u,
                        mu_i, mu_u, mu_nat,
                        severity=severity,
                        age=None,  # aleatoria en el constructor
                        use_probabilistic=True,
                        vaccinated=vaccinated,
                        wears_mask=wears_mask,
                        resistant=resistant)
            self.add_patient(p)
            self.logger.debug(f"Nuevo paciente {pid} ingresado en la celda ({cell.x}, {cell.y}).")

    def request_icu(self, agent: Agent):
        """
        Añade al agente a la cola de espera de UCI según su severidad.
        """
        priority = -agent.severity  # Negativo para priorizar mayor severidad
        self.icu_waiting_queue.append((priority, agent))
        # Ordenar la cola cada vez que se añade
        self.icu_waiting_queue.sort()
        self.logger.debug(f"{agent.unique_id} añadido a la cola de UCI con prioridad {priority}.")
        # Intentar asignar camas
        self.assign_icu_beds()

    def assign_icu_beds(self):
        """
        Asigna camas UCI a los agentes en la cola de espera según prioridad.
        """
        # Usar copia de la cola para iterar
        for priority, agent in list(self.icu_waiting_queue):
            # Solo asignar UCI a quienes estén infectados (I)
            if agent.state != EpidemicState.I:
                self.icu_waiting_queue.remove((priority, agent))
                continue

            # Buscar una cama disponible en cualquier UCI
            bed_found = False
            for col in self.grid:
                for cell in col:
                    if cell.is_icu and cell.has_free_bed():
                        cell.occupy_bed()
                        agent.set_current_cell(cell)
                        agent.state = EpidemicState.U
                        self.icu_waiting_queue.remove((priority, agent))
                        bed_found = True
                        self.logger.debug(f"{agent.unique_id} asignado a la UCI en la celda ({cell.x}, {cell.y}).")
                        break
                if bed_found:
                    break

            if not bed_found:
                # No hay camas disponibles, se queda en la cola
                break

    def get_icu_occupancy(self):
        """
        Retorna el número total de camas ocupadas y la capacidad total de UCI.
        """
        occupied = sum(cell.occupied_beds for col in self.grid for cell in col if cell.is_icu)
        capacity = sum(cell.icu_capacity for col in self.grid for cell in col if cell.is_icu)
        return occupied, capacity

    def get_vaccination_rate(self):
        """
        Retorna el porcentaje de la población que está vacunada.
        """
        total_vaccinated = sum(1 for w in self.workers if w.vaccinated) + \
                           sum(1 for p in self.patients if p.vaccinated)
        total_population = self.get_total_population()
        if total_population == 0:
            return 0
        return total_vaccinated / total_population

    def step(self, current_step, 
             # Parámetros de los nuevos pacientes
             beta_h, beta_g, sigma, alpha, gamma, gamma_u,
             mu_i, mu_u, mu_nat):
        """
        Un tick de simulación:
          1. Llega un paciente con prob arrival_rate
          2. Llamamos step() de cada agente
        """
        # 1) Possible new arrival
        self.spawn_patient_if_needed(beta_g, sigma, alpha, gamma, gamma_u, mu_i, mu_u, mu_nat)

        # 2) Snapshot de agentes
        ws = list(self.workers)
        ps = list(self.patients)

        # 3) Step de cada uno
        for w in ws:
            w.step(current_step, self)
        for p in ps:
            p.step(current_step, self)

        # 4) Asignar camas UCI si es posible
        self.assign_icu_beds()

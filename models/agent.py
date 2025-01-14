# models/agent.py

import random
import math
import uuid
from models.cell import Cell
from models.states import EpidemicState
import numpy as np

class Agent:
    """
    Clase base para cualquier agente de la simulación.
    """
    def __init__(self, unique_id, cell: Cell):
        self.unique_id = unique_id
        self.current_cell = cell
        if cell is not None:
            cell.add_agent(self)

    def set_current_cell(self, new_cell: Cell):
        if self.current_cell:
            self.current_cell.remove_agent(self)
        self.current_cell = new_cell
        if new_cell:
            new_cell.add_agent(self)

    def step(self, current_step, simulation_service):
        """
        Se redefine en subclases (HealthcareWorker, Patient).
        """
        pass

    def _prob(self, rate):
        return 1 - math.exp(-rate)

class HealthcareWorker(Agent):
    """
    Trabajador de la salud con estados SEIURD.
    Adaptado para \textit{Klebsiella pneumoniae}.
    """
    def __init__(
        self, unique_id, cell,
        beta_h, sigma, alpha, gamma, gamma_u,
        mu_i, mu_u, mu_nat,
        age=None,
        use_probabilistic=True,
        vaccinated=False,
        wears_mask=True,
        resistant=False  
    ):
        super().__init__(unique_id, cell)
        self.state = EpidemicState.S
        self.vaccinated = vaccinated
        self.wears_mask = wears_mask
        self.resistant = resistant  # Indica si el agente es resistente a antibióticos

        self.severity = 1  # 

        # Tasas epidemiológicas
        self.beta_h = beta_h
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_u = gamma_u
        self.mu_i = mu_i
        self.mu_u = mu_u

        if age is None:
            age = random.gauss(40, 10)  # Promedio 40 años, std 10
        self.age = age

        # Escalamos la mortalidad natural con base en la edad
        age_factor = 1.0 + max(0, (self.age - 50)/50)*1.5
        self.mu_nat = mu_nat * age_factor

        self.use_probabilistic = use_probabilistic

    def step(self, current_step, simulation_service):
        if self.state == EpidemicState.D:
            return  # Ya fallecido, no hace nada

        # 1) Muerte natural
        p_death_nat = self._prob(self.mu_nat)
        if random.random() < p_death_nat:
            self.state = EpidemicState.D
            simulation_service.register_death(self)
            simulation_service.logger.debug(f"{self.unique_id} falleció por causas naturales.")
            return

        # 2) Transiciones según estado
        if self.state in [EpidemicState.S, EpidemicState.V]:
            total_i = simulation_service.get_total_infectious()
            total_pop = simulation_service.get_total_population()
            if total_pop > 0:
                # Si vacunado, reducir la tasa de infección
                beta = self.beta_h * (0.5 if self.vaccinated else 1.0)
                # Si usa tapabocas, reducir la tasa de infección
                beta *= 0.7 if self.wears_mask else 1.0
                lam = beta * (total_i / total_pop)
                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} ha sido expuesto (E).")

        elif self.state == EpidemicState.E:
            p_ei = self._prob(self.sigma)
            if random.random() < p_ei:
                self.state = EpidemicState.I
                simulation_service.logger.debug(f"{self.unique_id} se ha infectado (I).")

        elif self.state == EpidemicState.I:
            # Probabilidad de muerte en I
            p_id = self._prob(self.mu_i)
            if random.random() < p_id:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                simulation_service.logger.debug(f"{self.unique_id} falleció en estado I.")
            else:
                # Probabilidad de ingresar a UCI
                p_iu = self._prob(self.alpha)
                if random.random() < p_iu:
                    simulation_service.request_icu(self)
                else:
                    # Recuperación, afectada por resistencia a antibióticos
                    if self.resistant:
                        adjusted_gamma = self.gamma * 0.5  # Reducción de la tasa de recuperación
                    else:
                        adjusted_gamma = self.gamma
                    p_ir = self._prob(adjusted_gamma)
                    if random.random() < p_ir:
                        self.state = EpidemicState.R
                        simulation_service.logger.debug(f"{self.unique_id} se ha recuperado (R).")

        elif self.state == EpidemicState.U:
            # En UCI, probabilidades de muerte o recuperación
            p_ud = self._prob(self.mu_u)
            if random.random() < p_ud:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                self.current_cell.free_bed()
                simulation_service.logger.debug(f"{self.unique_id} falleció en UCI.")
            else:
                if self.resistant:
                    adjusted_gamma_u = self.gamma_u * 0.5  # Reducción de la tasa de recuperación en UCI
                else:
                    adjusted_gamma_u = self.gamma_u
                p_ur = self._prob(adjusted_gamma_u)
                if random.random() < p_ur:
                    self.state = EpidemicState.R
                    self.current_cell.free_bed()
                    simulation_service.logger.debug(f"{self.unique_id} se ha recuperado de UCI (R).")

        elif self.state == EpidemicState.R:
            # Re-susceptibilización con baja probabilidad
            p_resus = 0.001  # 0.1% por paso
            if random.random() < p_resus:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} se ha re-susceptibilizado (S).")

        # 3) Movilidad más personalizada
        mobility_factor = 0.1 + (50 - self.age) * 0.001  # Menor movilidad si mayor
        mobility_factor = min(max(mobility_factor, 0.05), 0.2)  # Limitar entre 0.05 y 0.2
        if random.random() < mobility_factor:
            simulation_service.move_agent(self)

class Patient(Agent):
    """
    Paciente con estados SEIURD.
    Adaptado para \textit{Klebsiella pneumoniae}.
    """
    def __init__(
        self, unique_id, cell,
        beta_g, sigma, alpha, gamma, gamma_u,
        mu_i, mu_u, mu_nat,
        severity=1,
        age=None,
        use_probabilistic=True,
        vaccinated=False,
        wears_mask=True,
        resistant=False  # Nuevo atributo para resistencia a antibióticos
    ):
        super().__init__(unique_id, cell)
        self.state = EpidemicState.S
        self.vaccinated = vaccinated
        self.wears_mask = wears_mask
        self.resistant = resistant  # Indica si el paciente es resistente a antibióticos

        self.beta_g = beta_g
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_u = gamma_u
        self.mu_i = mu_i
        self.mu_u = mu_u

        if age is None:
            age = random.gauss(65, 15)  # Pacientes mayores, p.ej. media=65, std=15
        self.age = age

        # Ajustamos la mortalidad natural según la edad
        age_factor = 1.0 + max(0, (self.age - 60)/40)*2.0
        self.mu_nat = mu_nat * age_factor

        self.use_probabilistic = use_probabilistic

        # Severidad para triaje (1: leve, 2: moderado, 3: grave)
        self.severity = severity  # Puede ser determinado al crear el paciente

        # Tiempo en UCI
        self.icu_time = 0
        self.max_icu_time = 14  # Días en UCI

    def step(self, current_step, simulation_service):
        if self.state == EpidemicState.D:
            return  # Fallecido => no hace nada

        # 1) Muerte natural
        p_death_nat = self._prob(self.mu_nat)
        if random.random() < p_death_nat:
            self.state = EpidemicState.D
            simulation_service.register_death(self)
            simulation_service.logger.debug(f"{self.unique_id} falleció por causas naturales.")
            return

        # 2) Transiciones
        if self.state in [EpidemicState.S, EpidemicState.V]:
            total_i = simulation_service.get_total_infectious()
            total_pop = simulation_service.get_total_population()
            if total_pop > 0:
                # Si vacunado, reducir la tasa de infección
                beta = self.beta_g * (0.5 if self.vaccinated else 1.0)
                # Si usa tapabocas, reducir la tasa de infección
                beta *= 0.7 if self.wears_mask else 1.0
                lam = beta * (total_i / total_pop)
                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} ha sido expuesto (E).")

        elif self.state == EpidemicState.E:
            p_ei = self._prob(self.sigma)
            if random.random() < p_ei:
                self.state = EpidemicState.I
                simulation_service.logger.debug(f"{self.unique_id} se ha infectado (I).")

        elif self.state == EpidemicState.I:
            # Probabilidad de muerte en I
            p_id = self._prob(self.mu_i)
            if random.random() < p_id:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                simulation_service.logger.debug(f"{self.unique_id} falleció en estado I.")
            else:
                # Probabilidad de ingresar a UCI basada en severidad y vacunación
                adjusted_alpha = self.alpha * self.severity * (0.8 if self.vaccinated else 1.0)
                p_iu = self._prob(adjusted_alpha)
                if random.random() < p_iu:
                    simulation_service.request_icu(self)
                else:
                    # Recuperación, afectada por resistencia a antibióticos
                    if self.resistant:
                        adjusted_gamma = self.gamma * 0.5  # Reducción de la tasa de recuperación
                    else:
                        adjusted_gamma = self.gamma
                    p_ir = self._prob(adjusted_gamma)
                    if random.random() < p_ir:
                        self.state = EpidemicState.R
                        simulation_service.logger.debug(f"{self.unique_id} se ha recuperado (R).")

        elif self.state == EpidemicState.U:
            self.icu_time += 1
            if self.icu_time >= self.max_icu_time:
                # Decisión de recuperación o fallecimiento
                p_ud = self._prob(self.mu_u)
                if random.random() < p_ud:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    simulation_service.logger.debug(f"{self.unique_id} falleció en UCI tras {self.icu_time} días.")
                else:
                    # Recuperación, afectada por resistencia a antibióticos
                    if self.resistant:
                        adjusted_gamma_u = self.gamma_u * 0.5  # Reducción de la tasa de recuperación en UCI
                    else:
                        adjusted_gamma_u = self.gamma_u
                    p_ur = self._prob(adjusted_gamma_u)
                    if random.random() < p_ur:
                        self.state = EpidemicState.R
                        simulation_service.register_recovery_from_icu(self)
                        simulation_service.logger.debug(f"{self.unique_id} se ha recuperado de UCI tras {self.icu_time} días.")
            else:
                # Probabilidades de muerte o recuperación por paso
                p_ud = self._prob(self.mu_u)
                if random.random() < p_ud:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    self.current_cell.free_bed()
                    simulation_service.logger.debug(f"{self.unique_id} falleció en UCI.")
                else:
                    # Recuperación, afectada por resistencia a antibióticos
                    if self.resistant:
                        adjusted_gamma_u = self.gamma_u * 0.5
                    else:
                        adjusted_gamma_u = self.gamma_u
                    p_ur = self._prob(adjusted_gamma_u)
                    if random.random() < p_ur:
                        self.state = EpidemicState.R
                        self.current_cell.free_bed()
                        simulation_service.logger.debug(f"{self.unique_id} se ha recuperado de UCI (R).")

        elif self.state == EpidemicState.R:
            # Alta del sistema con cierta probabilidad
            discharge_prob = 0.005  # 0.5% por paso => ~ diaria
            if random.random() < discharge_prob:
                simulation_service.remove_patient(self)
                simulation_service.logger.debug(f"{self.unique_id} ha sido dado de alta del sistema.")
                return

        # 3) Movilidad aleatoria más personalizada
        mobility_factor = 0.05 + (50 - self.age) * 0.0005  # Menor movilidad si mayor
        mobility_factor = min(max(mobility_factor, 0.02), 0.1)  
        if random.random() < mobility_factor:
            simulation_service.move_agent(self)

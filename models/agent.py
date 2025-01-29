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
        """
        Dada una tasa continua 'rate', se transforma a probabilidad de suceso
        en un intervalo de tiempo (asumido = 1 step).
        """
        return 1 - math.exp(-rate)

class HealthcareWorker(Agent):
    """
    Trabajador de la salud con estados SEIURD.
    Adaptado para Klebsiella pneumoniae.
    """
    def __init__(
        self, unique_id, cell,
        beta_h, sigma, alpha, gamma_, gamma_u,
        mu_i, mu_u, mu_nat,
        p_resus,               # ### ADAPTACIÓN SEIURD: Tasa de re-susceptibilización
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
        self.resistant = resistant  

        # Tasas epidemiológicas
        self.beta_h = beta_h
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma_
        self.gamma_u = gamma_u
        self.mu_i = mu_i
        self.mu_u = mu_u
        self.p_resus = p_resus  # ### ADAPTACIÓN SEIURD

        # Ajuste de edad
        if age is None:
            age = random.gauss(40, 10)  
        self.age = age

        # Escalamos la mortalidad natural con base en la edad
        age_factor = 1.0 + max(0, (self.age - 50)/50)*1.5
        self.mu_nat = mu_nat * age_factor

        self.use_probabilistic = use_probabilistic

    def step(self, current_step, simulation_service):
        if self.state == EpidemicState.D:
            return  # Agente fallecido => no hace nada

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
                # ### ADAPTACIÓN SEIURD: Reducir beta si vacunado y/o usa mascarilla
                beta_eff = self.beta_h
                if self.vaccinated:
                    beta_eff *= 0.5
                if self.wears_mask:
                    beta_eff *= 0.7

                lam = beta_eff * (total_i / total_pop)
                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} ha pasado a E (expuesto).")

        elif self.state == EpidemicState.E:
            # E->I a tasa sigma
            p_ei = self._prob(self.sigma)
            if random.random() < p_ei:
                self.state = EpidemicState.I
                simulation_service.logger.debug(f"{self.unique_id} ha pasado a I (infectado).")

        elif self.state == EpidemicState.I:
            # Probabilidad de muerte en I
            p_id = self._prob(self.mu_i)
            if random.random() < p_id:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                simulation_service.logger.debug(f"{self.unique_id} falleció en I.")
            else:
                # Probabilidad de ingresar a UCI
                p_iu = self._prob(self.alpha)
                if random.random() < p_iu:
                    simulation_service.request_icu(self)
                else:
                    # Recuperación afectada por resistencia a antibióticos
                    adjusted_gamma = self.gamma * (0.5 if self.resistant else 1.0)
                    p_ir = self._prob(adjusted_gamma)
                    if random.random() < p_ir:
                        self.state = EpidemicState.R
                        simulation_service.logger.debug(f"{self.unique_id} se recuperó (R).")

        elif self.state == EpidemicState.U:
            # En UCI: muerte o recuperación
            p_ud = self._prob(self.mu_u)
            if random.random() < p_ud:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                self.current_cell.free_bed()
                simulation_service.logger.debug(f"{self.unique_id} falleció en U.")
            else:
                adjusted_gamma_u = self.gamma_u * (0.5 if self.resistant else 1.0)
                p_ur = self._prob(adjusted_gamma_u)
                if random.random() < p_ur:
                    self.state = EpidemicState.R
                    self.current_cell.free_bed()
                    simulation_service.logger.debug(f"{self.unique_id} se recuperó desde UCI (R).")

        elif self.state == EpidemicState.R:
            # ### ADAPTACIÓN SEIURD: Re-susceptibilización
            p_rs = self._prob(self.p_resus)
            if random.random() < p_rs:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} R->S (re-susceptibilizado).")

        # 3) Movilidad
        mobility_factor = 0.1 + (50 - self.age)*0.001
        mobility_factor = min(max(mobility_factor, 0.05), 0.2)
        if random.random() < mobility_factor:
            simulation_service.move_agent(self)

class Patient(Agent):
    """
    Paciente con estados SEIURD.
    Adaptado para Klebsiella pneumoniae.
    """
    def __init__(
        self, unique_id, cell,
        beta_g, sigma, alpha, gamma_, gamma_u,
        mu_i, mu_u, mu_nat,
        p_resus,               # ### ADAPTACIÓN SEIURD
        severity=1,
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
        self.resistant = resistant

        self.beta_g = beta_g
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma_
        self.gamma_u = gamma_u
        self.mu_i = mu_i
        self.mu_u = mu_u
        self.p_resus = p_resus  # ### ADAPTACIÓN SEIURD

        if age is None:
            age = random.gauss(65, 15)  
        self.age = age

        # Ajustamos la mortalidad natural según la edad
        age_factor = 1.0 + max(0, (self.age - 60)/40)*2.0
        self.mu_nat = mu_nat * age_factor

        self.use_probabilistic = use_probabilistic
        self.severity = severity  # (1: leve, 2: moderado, 3: grave)
        self.icu_time = 0
        self.max_icu_time = 14

    def step(self, current_step, simulation_service):
        if self.state == EpidemicState.D:
            return

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
                beta_eff = self.beta_g
                if self.vaccinated:
                    beta_eff *= 0.5
                if self.wears_mask:
                    beta_eff *= 0.7

                lam = beta_eff * (total_i / total_pop)
                p_inf = self._prob(lam)
                if random.random() < p_inf:
                    self.state = EpidemicState.E
                    simulation_service.logger.debug(f"{self.unique_id} S->E (Expuesto).")

        elif self.state == EpidemicState.E:
            p_ei = self._prob(self.sigma)
            if random.random() < p_ei:
                self.state = EpidemicState.I
                simulation_service.logger.debug(f"{self.unique_id} E->I (Infectado).")

        elif self.state == EpidemicState.I:
            p_id = self._prob(self.mu_i)
            if random.random() < p_id:
                self.state = EpidemicState.D
                simulation_service.register_death(self)
                simulation_service.logger.debug(f"{self.unique_id} falleció en I.")
            else:
                # Probabilidad de ingresar a UCI con factor severidad + vacunación
                adjusted_alpha = self.alpha * self.severity * (0.8 if self.vaccinated else 1.0)
                p_iu = self._prob(adjusted_alpha)
                if random.random() < p_iu:
                    simulation_service.request_icu(self)
                else:
                    adjusted_gamma = self.gamma * (0.5 if self.resistant else 1.0)
                    p_ir = self._prob(adjusted_gamma)
                    if random.random() < p_ir:
                        self.state = EpidemicState.R
                        simulation_service.logger.debug(f"{self.unique_id} I->R (Recuperado).")

        elif self.state == EpidemicState.U:
            # Manejo de días en UCI
            self.icu_time += 1
            if self.icu_time >= self.max_icu_time:
                # Al final de max_icu_time, se decide muerte o recuperación
                p_ud = self._prob(self.mu_u)
                if random.random() < p_ud:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    simulation_service.logger.debug(f"{self.unique_id} falleció en U tras {self.icu_time} días.")
                else:
                    adjusted_gamma_u = self.gamma_u * (0.5 if self.resistant else 1.0)
                    p_ur = self._prob(adjusted_gamma_u)
                    if random.random() < p_ur:
                        self.state = EpidemicState.R
                        simulation_service.register_recovery_from_icu(self)
                        simulation_service.logger.debug(f"{self.unique_id} se recuperó de U tras {self.icu_time} días.")
            else:
                # En cada paso, también puede morir o recuperarse
                p_ud = self._prob(self.mu_u)
                if random.random() < p_ud:
                    self.state = EpidemicState.D
                    simulation_service.register_death(self)
                    self.current_cell.free_bed()
                    simulation_service.logger.debug(f"{self.unique_id} falleció en U.")
                else:
                    adjusted_gamma_u = self.gamma_u * (0.5 if self.resistant else 1.0)
                    p_ur = self._prob(adjusted_gamma_u)
                    if random.random() < p_ur:
                        self.state = EpidemicState.R
                        self.current_cell.free_bed()
                        simulation_service.logger.debug(f"{self.unique_id} se recuperó de U (R).")

        elif self.state == EpidemicState.R:
            # ### ADAPTACIÓN SEIURD: Re-susceptibilización = p_resus
            p_rs = self._prob(self.p_resus)
            if random.random() < p_rs:
                self.state = EpidemicState.S
                simulation_service.logger.debug(f"{self.unique_id} R->S (re-susceptibilizado).")
            else:
                # Dar de alta con cierta probabilidad
                discharge_prob = 0.005
                if random.random() < discharge_prob:
                    simulation_service.remove_patient(self)
                    simulation_service.logger.debug(f"{self.unique_id} ha sido dado de alta (R).")
                    return

        # 3) Movilidad
        mobility_factor = 0.05 + (50 - self.age) * 0.0005
        mobility_factor = min(max(mobility_factor, 0.02), 0.1)
        if random.random() < mobility_factor:
            simulation_service.move_agent(self)

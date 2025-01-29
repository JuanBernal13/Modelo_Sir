# models/states.py

from enum import Enum, auto

class EpidemicState(Enum):
    S = auto()  # Susceptible
    V = auto()  # Vacunado
    E = auto()  # Expuesto
    I = auto()  # Infectado
    U = auto()  # UCI
    R = auto()  # Recuperado
    D = auto()  # Fallecido

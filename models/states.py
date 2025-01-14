# models/states.py

from enum import Enum, auto

class EpidemicState(Enum):
    S = auto()  # Susceptible
    V = auto()  # Vacunado
    E = auto()  # Expuesto (período de latencia)
    I = auto()  # Infectado (contagioso)
    U = auto()  # En UCI
    R = auto()  # Recuperado
    D = auto()  # Fallecido

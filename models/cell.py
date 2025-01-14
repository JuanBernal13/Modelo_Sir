# models/cell.py

class Cell:
    """
    Representa una celda en la grilla (posiblemente UCI).
    """
    def __init__(self, x, y, is_icu=False, icu_capacity=2):
        self.x = x
        self.y = y
        self.agents = []
        self.is_icu = is_icu
        self.icu_capacity = icu_capacity if is_icu else 0
        self.occupied_beds = 0

    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)

    def has_free_bed(self):
        return self.occupied_beds < self.icu_capacity

    def occupy_bed(self):
        if self.has_free_bed():
            self.occupied_beds += 1

    def free_bed(self):
        self.occupied_beds = max(0, self.occupied_beds - 1)

from abc import ABC, abstractmethod

from ase import units


class Proposer(ABC):
    def __init__(self, temperature, *args, **kwargs) -> None:
        self.temperature = temperature
        return

    @abstractmethod
    def propose(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def calc_acceptability(self, *args, **kwargs) -> float:
        raise NotImplementedError

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature
        self._beta = 1 / units.kB / temperature

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self._beta = beta
        self._temperature = 1 / units.kB / beta

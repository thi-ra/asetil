from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from ase import Atoms, units

from asetil.monte_carlo.proposer import Proposer


class BaseMonteCarlo(ABC):
    def __init__(self, max_iter: int, temperature: float) -> None:
        self.max_iter = max_iter
        self.temperature = temperature
        pass

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

    def is_acceptable(self, acceptability):
        return np.random.rand() < acceptability

    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class MonteCarlo(BaseMonteCarlo):
    def __init__(self, max_iter, temperature, proposers: Iterable[Proposer]) -> None:
        super().__init__(max_iter, temperature=temperature)
        self.proposers = proposers
        return

    def step(self, system: Atoms):
        proposer = np.random.choice(self.proposers)
        tags = proposer.select_tags(system)
        candidate = proposer.propose(system, tags=tags)
        acceptability = proposer.calc_acceptability(system, candidate, beta=self.beta)
        if self.is_acceptable(acceptability):
            system = candidate
        return system

    def run(self, system: Atoms, current_iter=0):
        for _ in range(current_iter, self.max_iter):
            system = self.step(system)
        return system

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

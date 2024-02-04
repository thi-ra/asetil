from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from ase import Atoms, units

from asetil.monte_carlo.logger import Logger
from asetil.monte_carlo.proposer import Proposer
from asetil.monte_carlo.step_info import MCStepInfo


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
    def __init__(
        self,
        max_iter,
        temperature,
        proposers: Iterable[Proposer],
        loggers: Iterable[Logger] = None,
    ) -> None:
        super().__init__(max_iter, temperature=temperature)
        self.proposers = proposers
        self.loggers = loggers
        self.info = MCStepInfo(
            iteration=None,
            temperature=self.temperature,
            beta=self.beta,
            proposer=None,
            is_accepted=False,
            acceptability=0,
            system=None,
            candidate=None,
            latest_accepted_energy=np.nan,
        )

    def step(self, system: Atoms, iteration=None):
        proposer = np.random.choice(self.proposers)
        tags = proposer.select_tags(system)
        candidate = proposer.propose(system, tags=tags)
        acceptability = proposer.calc_acceptability(system, candidate, beta=self.beta)
        is_accepted = self.is_acceptable(acceptability)
        if is_accepted:
            latest_accepted_energy = candidate.get_potential_energy()
        else:
            latest_accepted_energy = self.info.latest_accepted_energy

        self.info = MCStepInfo(
            iteration=iteration,
            temperature=self.temperature,
            beta=self.beta,
            proposer=proposer,
            is_accepted=is_accepted,
            acceptability=acceptability,
            system=system,
            candidate=candidate,
            latest_accepted_energy=latest_accepted_energy,
        )
        if self.loggers is not None:
            for logger in self.loggers:
                logger.log(self.info)
        return candidate if is_accepted else system

    def run(self, system: Atoms, current_iter=0):
        if self.loggers is not None and current_iter == 0:
            print("initializing loggers")
            for logger in self.loggers:
                logger.initialize()
        for i in range(current_iter, self.max_iter):
            system = self.step(system, iteration=i)
        return system

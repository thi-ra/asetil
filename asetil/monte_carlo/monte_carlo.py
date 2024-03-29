from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from ase import Atoms, units

from asetil.monte_carlo.logger import Logger
from asetil.monte_carlo.sampler_selector import SamplerSelector
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
        sampler_selector: SamplerSelector,
        loggers: Iterable[Logger] = None,
    ) -> None:
        super().__init__(max_iter, temperature=temperature)
        self.sampler_selector = sampler_selector
        self.loggers = loggers
        self.info = MCStepInfo(
            iteration=None,
            temperature=self.temperature,
            beta=self.beta,
            sampler=None,
            is_accepted=False,
            acceptability=0,
            system=None,
            candidate=None,
            latest_accepted_energy=np.nan,
            delta_energy=np.nan,
            tags=None,
        )

    def step(self, system: Atoms, iteration=None):
        sampler = self.sampler_selector.select()
        tags = sampler.select_tags(system)
        candidate = sampler.sample(system, tags=tags)
        delta_energy = sampler.calc_delta_energy(system, candidate)
        acceptability = sampler.calc_acceptability(
            system, candidate, beta=self.beta, delta_energy=delta_energy
        )
        is_accepted = self.is_acceptable(acceptability)
        if is_accepted:
            latest_accepted_energy = sampler.calc_after_energy(candidate)
        else:
            latest_accepted_energy = self.info.latest_accepted_energy

        self.info = MCStepInfo(
            iteration=iteration,
            temperature=self.temperature,
            beta=self.beta,
            sampler=sampler,
            is_accepted=is_accepted,
            acceptability=acceptability,
            system=system,
            candidate=candidate,
            latest_accepted_energy=latest_accepted_energy,
            delta_energy=delta_energy,
            tags=tags,
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

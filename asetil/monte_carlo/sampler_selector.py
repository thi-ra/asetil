from abc import ABC, abstractclassmethod

import numpy as np

from asetil.monte_carlo.sampler import Sampler


class SamplerSelector(ABC):
    @abstractclassmethod
    def select(self, *args, **kwargs):
        raise NotImplementedError


class RandomSamplerSelector(SamplerSelector):
    def __init__(self, samplers: list[Sampler] = None, weights: list[float] = None):
        samplers = samplers or []
        weights = weights or []
        if len(samplers) != len(weights):
            raise RuntimeError("The length of sampler and weight should be the same")

        self._samplers = samplers
        self._weights = weights
        self._p = np.array(weights) / sum(weights)

    def add_sampler(self, sampler, weight):
        self._samplers.append(sampler)
        self._weights.append(weight)
        self._p = np.array(self._weights) / sum(self._weights)

    def select(self, *args, **kwargs):
        return np.random.choice(self._samplers, p=self._p)

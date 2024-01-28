from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from ase import units
from ase.atoms import Atoms


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


class TranslateProposer(Proposer):
    def __init__(
        self,
        temperature: float,
        x_range: Iterable[float, float] = (-0.15, 0.15),
        y_range: Iterable[float, float] = (-0.15, 0.15),
        z_range: Iterable[float, float] = (-0.15, 0.15),
    ) -> None:
        """
        temperature: float
            Temperature of the system in Kelvin
        x_range: Iterable[float]
            x-axis range of movement
        y_range: Iterable[float]
            y-axis range of movement
        z_range: Iterable[float]
            z-axis range of movement
        """
        super().__init__(temperature)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        return

    @property
    def x_range(self):
        return self._x_range

    @x_range.setter
    def x_range(self, x_range):
        if len(x_range) != 2:
            raise ValueError("x_range must be an array of length 2")
        self._x_range = x_range

    @property
    def y_range(self):
        return self._y_range

    @y_range.setter
    def y_range(self, y_range):
        if len(y_range) != 2:
            raise ValueError("y_range must be an array of length 2")
        self._y_range = y_range

    @property
    def z_range(self):
        return self._z_range

    @z_range.setter
    def z_range(self, z_range):
        if len(z_range) != 2:
            raise ValueError("z_range must be an array of length 2")
        self._z_range = z_range

    def propose(self, system: Atoms, tags: Iterable[int]) -> None:
        # split system into main and sub systems
        tags = set(tags)
        mask = np.array([i in tags for i in system.get_tags()])
        main_system = system[~mask]
        sub_system = system[mask]

        # create a new sub system with translated positions
        positions = sub_system.get_positions()
        translate_vector = np.array(
            [
                np.random.uniform(*self.x_range),
                np.random.uniform(*self.y_range),
                np.random.uniform(*self.z_range),
            ]
        )
        new_positions = positions + translate_vector
        sub_system.set_positions(new_positions)

        # combine main and sub systems
        candidate = main_system + sub_system
        return candidate

    def calc_acceptability(self, before: Atoms, after: Atoms) -> float:
        e_before = before.get_potential_energy()
        e_after = after.get_potential_energy()
        return min(1, np.exp(-self.beta * (e_after - e_before)))


class RotateProposer(Proposer):
    def __init__(
        self,
        temperature: float,
        phi_range: Iterable[float, float] = (-0.15, 0.15),
        theta_range: Iterable[float, float] = (-0.15, 0.15),
        psi_range: Iterable[float, float] = (-0.15, 0.15),
        center="COM",
    ) -> None:
        """
        temperature: float
            Temperature of the system in Kelvin
        phi_range: Iterable[float]
            Range of the 1st rotation angle around the z-axis. [degree]
        theta_range: Iterable[float]
            Range of rotation angle around the x-axis. [degree]
        psi_range: Iterable[float]
            Range of rotation around the z-axis. [degree]
        center: str or np.ndarray
            Center of rotation. If str, it must be one of "COM" or "COP" or "COU".
            If np.ndarray, it must be a 3D vector.
            See https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.euler_rotate
        """
        super().__init__(temperature)
        self.phi_range = phi_range
        self.theta_range = theta_range
        self.psi_range = psi_range
        self.center = center
        return

    @property
    def phi_range(self):
        return self._phi_range

    @phi_range.setter
    def phi_range(self, phi_range):
        if len(phi_range) != 2:
            raise ValueError("phi_range must be an array of length 2")
        self._phi_range = phi_range

    @property
    def theta_range(self):
        return self._theta_range

    @theta_range.setter
    def theta_range(self, theta_range):
        if len(theta_range) != 2:
            raise ValueError("theta_range must be an array of length 2")
        self._theta_range = theta_range

    @property
    def psi_range(self):
        return self._psi_range

    @psi_range.setter
    def psi_range(self, psi_range):
        if len(psi_range) != 2:
            raise ValueError("psi_range must be an array of length 2")
        self._psi_range = psi_range

    def propose(self, system: Atoms, tags: Iterable[int]) -> None:
        candidate = system.copy()
        for tag in tags:
            # split system into main and sub systems
            mask = candidate.get_tags() == tag
            main_system = candidate[~mask]
            sub_system = candidate[mask]

            # create a new sub system with euler rotation
            rotate_angle = [
                np.random.uniform(*self.phi_range),
                np.random.uniform(*self.theta_range),
                np.random.uniform(*self.psi_range),
            ]
            sub_system.euler_rotate(*rotate_angle, center=self.center)

            # combine main and sub systems
            candidate = main_system + sub_system
        return candidate

    def calc_acceptability(self, before: Atoms, after: Atoms) -> float:
        e_before = before.get_potential_energy()
        e_after = after.get_potential_energy()
        return min(1, np.exp(-self.beta * (e_after - e_before)))

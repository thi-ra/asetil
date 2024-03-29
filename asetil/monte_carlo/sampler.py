from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from ase.atoms import Atoms

from asetil.monte_carlo.tag_selector import TagSelector
from asetil.util import calc_thermal_de_broglie


class Sampler(ABC):
    def __init__(self, tag_selector: TagSelector, *args, **kwargs) -> None:
        self.tag_selector = tag_selector
        return

    @abstractmethod
    def sample(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def calc_delta_energy(self, before, after, *args, **kwargs) -> float:
        e_before = self.calc_before_energy(before)
        e_after = self.calc_after_energy(after)
        return e_after - e_before

    def calc_before_energy(self, atoms: Atoms, *args, **kwargs) -> float:
        return atoms.get_potential_energy()

    def calc_after_energy(self, atoms: Atoms, *args, **kwargs) -> float:
        return atoms.get_potential_energy()

    @abstractmethod
    def calc_acceptability(
        self,
        before: Atoms,
        after: Atoms,
        beta: float,
        delta_energy: float,
        *args,
        **kwargs,
    ) -> float:
        raise NotImplementedError

    def select_tags(self, system: Atoms):
        return self.tag_selector.select(system)


class TranslateSampler(Sampler):
    name = "Translate"

    def __init__(
        self,
        tag_selector: TagSelector,
        x_range: Iterable[float] = (-0.15, 0.15),
        y_range: Iterable[float] = (-0.15, 0.15),
        z_range: Iterable[float] = (-0.15, 0.15),
    ) -> None:
        """
        tag_selector: TagSelector
            TagSelector object
        x_range: Iterable[float]
            x-axis range of movement
        y_range: Iterable[float]
            y-axis range of movement
        z_range: Iterable[float]
            z-axis range of movement
        """
        super().__init__(tag_selector)
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

    def sample(self, system: Atoms, tags: Iterable[int]) -> None:
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
        candidate.calc = system.calc
        return candidate

    def calc_acceptability(
        self,
        before: Atoms,
        after: Atoms,
        beta: float,
        delta_energy: float,
        *args,
        **kwargs,
    ) -> float:
        return min(1, np.exp(-beta * delta_energy))


class EulerRotateSampler(Sampler):
    name = "EulerRotate"

    def __init__(
        self,
        tag_selector: TagSelector,
        phi_range: Iterable[float] = (-60, 60),
        theta_range: Iterable[float] = (-60, 60),
        psi_range: Iterable[float] = (-60, 60),
        center="COM",
    ) -> None:
        """
        tag_selector: TagSelector
            TagSelector object
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
        super().__init__(tag_selector)
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

    def sample(self, system: Atoms, tags: Iterable[int]) -> None:
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
        candidate.calc = system.calc
        return candidate

    def calc_acceptability(
        self,
        before: Atoms,
        after: Atoms,
        beta: float,
        delta_energy: float,
        *args,
        **kwargs,
    ) -> float:
        return min(1, np.exp(-beta * delta_energy))


class XYZAxesRotateSampler(Sampler):
    name = "XYZRotate"

    def __init__(
        self,
        tag_selector: TagSelector,
        x_axis_range: Iterable[float] = (-60, 60),
        y_axis_range: Iterable[float] = (-60, 60),
        z_axis_range: Iterable[float] = (-60, 60),
        center="COM",
    ) -> None:
        """
        tag_selector: TagSelector
            TagSelector object
        x_axis_range: Iterable[float]
            Range of rotation angle around the x-axis. [degree]
        y_axis_range: Iterable[float]
            Range of rotation angle around the y-axis. [degree]
        z_axis_range: Iterable[float]
            Range of rotation angle around the z-axis. [degree]
        center: str or np.ndarray
            Center of rotation. If str, it must be one of "COM" or "COP" or "COU".
            If np.ndarray, it must be a 3D vector.
            See https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.rotate
        """
        super().__init__(tag_selector)
        self.x_axis_range = x_axis_range
        self.y_axis_range = y_axis_range
        self.z_axis_range = z_axis_range
        self.center = center

    @property
    def x_axis_range(self):
        return self._x_axis_range

    @x_axis_range.setter
    def x_axis_range(self, x_axis_range):
        if len(x_axis_range) != 2:
            raise ValueError("x_axis_range must be an array of length 2")
        self._x_axis_range = x_axis_range

    @property
    def y_axis_range(self):
        return self._y_axis_range

    @y_axis_range.setter
    def y_axis_range(self, y_axis_range):
        if len(y_axis_range) != 2:
            raise ValueError("y_axis_range must be an array of length 2")
        self._y_axis_range = y_axis_range

    @property
    def z_axis_range(self):
        return self._z_axis_range

    @z_axis_range.setter
    def z_axis_range(self, z_axis_range):
        if len(z_axis_range) != 2:
            raise ValueError("z_axis_range must be an array of length 2")
        self._z_axis_range = z_axis_range

    def sample(self, system: Atoms, tags: Iterable[int]) -> None:
        candidate = system.copy()
        for tag in tags:
            # split system into main and sub systems
            mask = candidate.get_tags() == tag
            main_system = candidate[~mask]
            sub_system = candidate[mask]

            # create a new sub system with rotation
            sub_system.rotate(
                np.random.uniform(*self.x_axis_range), "x", center=self.center
            )
            sub_system.rotate(
                np.random.uniform(*self.y_axis_range), "y", center=self.center
            )
            sub_system.rotate(
                np.random.uniform(*self.z_axis_range), "z", center=self.center
            )

            # combine main and sub systems
            candidate = main_system + sub_system
        candidate.calc = system.calc
        return candidate

    def calc_acceptability(
        self,
        before: Atoms,
        after: Atoms,
        beta: float,
        delta_energy: float,
        *args,
        **kwargs,
    ) -> float:
        return min(1, np.exp(-beta * delta_energy))


class AddSampler(Sampler):
    name = "Add"

    def __init__(self, tag_selector: TagSelector, additive: Atoms) -> None:
        self.tag_selector = tag_selector
        self.additive = additive
        return

    def sample(self, system: Atoms, tags: Iterable[int]) -> Atoms:
        candidate = system.copy()
        for tag in tags:
            # set tags to additive
            additive = self.additive.copy()
            additive.set_tags([tag] * len(additive))

            # move additive to random position
            cell = system.get_cell()
            random_positions = np.sum([cell[i] * np.random.rand() for i in range(3)])
            translate_vector = random_positions - additive.get_center_of_mass()
            additive.translate(translate_vector)

            # rotate additive to random orientation
            phi, theta, psi = np.random.rand(3) * 360
            additive.euler_rotate(phi, theta, psi, center="COM")
            candidate += additive
        candidate.calc = system.calc
        return system

    def calc_acceptability(
        self,
        before: Atoms,
        after: Atoms,
        beta: float,
        delta_energy: float,
        *args,
        **kwargs,
    ) -> float:
        # thermal de Broglie wavelength
        mass = sum(self.additive.get_masses())
        dbl = calc_thermal_de_broglie(molecular_weight=mass, beta=beta)

        # number of melecules in system
        num_molecule = len(set([i for i in after.get_tags() if i != 0]))
        # volume of system
        vol = after.get_volume()

        # acceptability
        acceptability = np.exp(beta * delta_energy) * vol / (num_molecule * dbl**3)
        return min(1, acceptability)


class RemoveSampler(Sampler):
    name = "Remove"

    def __init__(self, tag_selector: TagSelector, additive: Atoms) -> None:
        self.tag_selector = tag_selector
        self.additive = additive
        return

    def sample(self, system: Atoms, tags: Iterable[int]) -> Atoms:
        mask = [i not in tags for i in system.get_tags()]
        candidate = system[mask]
        candidate.calc = system.calc
        return system

    def calc_acceptability(
        self,
        before: Atoms,
        after: Atoms,
        beta: float,
        delta_energy: float,
        *args,
        **kwargs,
    ) -> float:
        # thermal de Broglie wavelength
        mass = sum(self.additive.get_masses())
        dbl = calc_thermal_de_broglie(molecular_weight=mass, beta=beta)

        # number of melecules in system
        num_molecule = len(set([i for i in after.get_tags() if i != 0]))
        # volume of system
        vol = after.get_volume()

        # acceptability
        acceptability = (
            np.exp(beta * delta_energy) / vol * ((num_molecule + 1) * dbl**3)
        )
        return min(1, acceptability)

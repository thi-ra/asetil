from typing import Iterable

import numpy as np
from ase import Atoms

from asetil.monte_carlo.logger import Logger
from asetil.monte_carlo.sampler import Sampler
from asetil.monte_carlo.step_info import MCStepInfo
from asetil.monte_carlo.tag_selector import TagSelector


class SiteExchangeSampler(Sampler):
    name = "SiteExchange"

    def __init__(
        self,
        tag1_selector: TagSelector,
        tag2_selector: TagSelector,
    ) -> None:
        """
        tag1_selector: TagSelector
            Selector for tag1
        tag2_selector: TagSelector
            Selector for tag2
        """
        self.tag1_range = tag1_selector
        self.tag2_range = tag2_selector
        return

    def select_tags(self, system: Atoms):
        tag1 = self.tag1_selector.select(system)
        tag2 = self.tag2_selector.select(system)
        tags = [(t1, t2) for t1, t2 in zip(tag1, tag2)]
        return tags

    def sample(self, system: Atoms, tags: Iterable[tuple[int]]) -> None:
        candidate = system.copy()
        for tag in tags:
            mask1 = np.array(candidate.get_tags()) == tag[0]
            mask2 = np.array(candidate.get_tags()) == tag[1]
            atoms1 = candidate[mask1]
            atoms2 = candidate[mask2]

            center1 = atoms1.get_center_of_mass()
            center2 = atoms2.get_center_of_mass()

            atoms1.translate(center2 - center1)
            atoms2.translate(center1 - center2)

            new_pos = candidate.get_positions()
            new_pos[mask1] = atoms1.get_positions()
            new_pos[mask2] = atoms2.get_positions()
            candidate.set_positions(new_pos)
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


class ChemicalSymbolExchangeSampler(Sampler):
    name = "SymbolExchange"

    def __init__(
        self,
        tag1_selector: TagSelector,
        tag2_selector: TagSelector,
    ) -> None:
        """
        tag1_selector: TagSelector
            Selector for tag1
        tag2_selector: TagSelector
            Selector for tag2
        """
        self.tag1_range = tag1_selector
        self.tag2_range = tag2_selector
        return

    def select_tags(self, system: Atoms):
        tag1 = self.tag1_selector.select(system)
        tag2 = self.tag2_selector.select(system)
        tags = [(t1, t2) for t1, t2 in zip(tag1, tag2)]
        return tags

    def sample(self, system: Atoms, tags: Iterable[tuple[int]]) -> Atoms:
        candidate = system.copy()
        symbols = candidate.get_chemical_symbols()
        all_tags = candidate.get_tags()
        for t1, t2 in tags:
            index1 = [i for i, t in enumerate(candidate.get_tags()) if t == t1]
            index2 = [i for i, t in enumerate(candidate.get_tags()) if t == t2]
            for i1, i2 in zip(index1, index2):
                symbols[i1], symbols[i2] = symbols[i2], symbols[i1]
                all_tags[i1], all_tags[i2] = all_tags[i2], all_tags[i1]
        candidate.set_chemical_symbols(symbols)
        candidate.set_tags(all_tags)
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


class SiteExchangeClusterGenerationSampler(ChemicalSymbolExchangeSampler):
    def __init__(self, tag1_selector: TagSelector, tag2_selector: TagSelector) -> None:
        """
        Please use the symbol X to represent an empty site.
        Set tag_selector1 as a TagSelector that returns tags representing X.
        Set tag_selector2 as a TagSelector that returns tags representing actual elements.
        For the calculation of delta_energy, compute the energy difference in the structure with X removed.

        tag1_selector: TagSelector
            Selector for tag1, representing X
        tag2_selector: TagSelector
            Selector for tag2, representing actual elements
        """
        self.tag1_selector = tag1_selector
        self.tag2_selector = tag2_selector
        return

    def calc_before_energy(self, atoms: Atoms, *args, **kwargs) -> float:
        wo_vac = atoms[np.array(atoms.get_chemical_symbols()) != "X"]
        wo_vac.calc = atoms.calc
        return super().calc_before_energy(wo_vac, *args, **kwargs)

    def calc_after_energy(self, atoms: Atoms, *args, **kwargs) -> float:
        wo_vac = atoms[np.array(atoms.get_chemical_symbols()) != "X"]
        wo_vac.calc = atoms.calc
        return super().calc_before_energy(wo_vac, *args, **kwargs)


class SEMCPrintLogger(Logger):
    def __init__(self, log_interval) -> None:
        super().__init__(log_interval)

    def initialize(self):
        text = (
            f"{"iteration":>10}, {"sampler.name":>15}, {"latest_accepted_energy":>22}, {"delta_e":>10}, "
            f"{"acceptability":>15}, {"is_accepted":>12}, {"tags":>20}\n"
        )
        print(text, end="")

    def log(self, info: MCStepInfo):
        if info.iteration % self.log_interval != 0:
            return

        text = (
            f"{info.iteration:>10}, {info.sampler.name:>15}, {info.latest_accepted_energy:22.6f}, {info.delta_energy:10.6f}, "
            f"{info.acceptability:15.6f}, {str(info.is_accepted):>12}, {str(info.tags):>20}\n"
        )
        print(text, end="")

from dataclasses import dataclass

from ase.atoms import Atoms


@dataclass
class MCStepInfo:
    iteration: int
    temperature: float
    beta: float
    sampler: object
    is_accepted: bool
    acceptability: float
    system: Atoms
    candidate: Atoms
    latest_accepted_energy: float
    delta_energy: float

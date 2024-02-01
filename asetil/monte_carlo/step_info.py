from dataclasses import dataclass

from ase.atoms import Atoms


@dataclass
class MCStepInfo:
    iteration: int
    temperature: float
    beta: float
    proposer: object
    is_accepted: bool
    acceptability: float
    system: Atoms
    candidate: Atoms

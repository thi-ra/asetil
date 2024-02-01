from abc import ABC, abstractclassmethod

from asetil.monte_carlo.step_info import MCStepInfo


class Logger(ABC):
    def __init__(self, log_interval) -> None:
        self.log_interval = log_interval

    @abstractclassmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError


class MCPrintLogger(Logger):
    def __init__(self, log_interval) -> None:
        super().__init__(log_interval)

    def initialize(self):
        text = (
            f"{"iteration":>10}, {"proposer.name":>15}, {"delta_e":>10}, "
            f"{"acceptability":15}:, {"is_accepted":>12}\n",
        )
        print(text, end="")

    def log(self, info: MCStepInfo):
        if info.iteration % self.log_interval != 0:
            return

        delta_energy = (
            info.candidate.get_potential_energy() - info.system.get_potential_energy()
        )
        text = (
            f"{info.iteration:>10}, {info.proposer.name:>15}, {delta_energy:10.6f}, "
            f"{info.acceptability:15.6f}:, {info.is_accepted:>12}\n"
        )
        print(text, end="")


class FileLogger(Logger):
    def __init__(self, log_interval) -> None:
        super().__init__(log_interval)


class AtomsLogger(Logger):
    def __init__(self, log_interval) -> None:
        super().__init__(log_interval)

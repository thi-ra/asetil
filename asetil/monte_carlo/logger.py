from abc import ABC, abstractclassmethod
from pathlib import Path

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
            f"{"iteration":>10}, {"sampler.name":>15}, {"latest_accepted_energy":>22}, {"delta_e":>10}, "
            f"{"acceptability":>15}, {"is_accepted":>12}\n"
        )
        print(text, end="")

    def log(self, info: MCStepInfo):
        if info.iteration % self.log_interval != 0:
            return

        text = (
            f"{info.iteration:>10}, {info.sampler.name:>15}, {info.latest_accepted_energy:22.6f}, {info.delta_energy:10.6f}, "
            f"{info.acceptability:15.6f}, {str(info.is_accepted):>12}\n"
        )
        print(text, end="")


class MCPFileLogger(Logger):
    def __init__(self, log_interval, out_file, force_overwrite=False) -> None:
        super().__init__(log_interval)
        self.out_file = Path(out_file)
        self.force_overwrite = force_overwrite
        if not self.force_overwrite:
            raise RuntimeError(f"File {self.out_file} already exists.")

    def initialize(self):
        if self.out_file.exists() and not self.force_overwrite:
            raise RuntimeError(f"File {self.out_file} already exists.")

        text = (
            f"{"iteration":>10}, {"sampler.name":>15}, {"latest_accepted_energy":>22}, {"delta_e":>10}, "
            f"{"acceptability":>15}, {"is_accepted":>12}\n"
        )
        with open(self.out_file, "w") as f:
            f.write(text)

    def log(self, info: MCStepInfo):
        if info.iteration % self.log_interval != 0:
            return

        text = (
            f"{info.iteration:>10}, {info.sampler.name:>15}, {info.latest_accepted_energy:22.6f}, {info.delta_energy:10.6f}, "
            f"{info.acceptability:15.6f}, {str(info.is_accepted):>12}\n"
        )
        with open(self.out_file, "a") as f:
            f.write(text)


class MCInMemoryLogger(Logger):
    def __init__(self, log_interval) -> None:
        super().__init__(log_interval)
        self.log_data = []

    def initialize(self):
        self.log_data = []

    def log(self, info: MCStepInfo):
        if info.iteration % self.log_interval != 0:
            return
        self.log_data.append(info)

    def get_log(self, with_columns=True):
        log = [
            [
                i.iteration,
                i.sampler.name,
                i.latest_accepted_energy,
                i.delta_energy,
                i.acceptability,
                i.is_accepted,
            ]
            for i in self.log_data
        ]
        if with_columns:
            columns = (
                "iteration",
                "sampler.name",
                "latest_accepted_energy",
                "delta_e",
                "acceptability",
                "is_accepted",
            )
            return log, columns
        else:
            return log


class AtomsLogger(Logger):
    def __init__(self, log_interval) -> None:
        super().__init__(log_interval)

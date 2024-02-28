from abc import ABC, abstractmethod
from typing import List

import numpy as np
from ase import Atoms


class TagSelector(ABC):
    def __init__(self) -> None:
        return

    @abstractmethod
    def select(self, *args, **kwargs) -> None:
        raise NotImplementedError


class RandomTagSelector(TagSelector):
    """
    Select tags randomly from the system.
    """

    def __init__(self, target_tags=None, ignore_tags=None) -> None:
        self.target_tags = target_tags
        self.ignore_tags = ignore_tags if ignore_tags is not None else []

    def select(self, system: Atoms, num_tags: int = 1) -> List[int]:
        if self.target_tags is not None:
            tags = system.get_tags()
        else:
            tags = self.target_tags
        available_tags = [tag for tag in tags if tag not in self.ignore_tags]
        return np.random.choice(available_tags, size=num_tags, replace=False).tolist()


class NotExistTagSelector(TagSelector):
    """
    Select tags that do not exist in the system.
    """

    def __init__(self, ignore_tags=None) -> None:
        self.ignore_tags = ignore_tags if ignore_tags is not None else []

    def select(self, system: Atoms, num_tags: int = 1) -> List[int]:
        tags = set(system.get_tags()) | set(self.ignore_tags)
        selected_tags = []
        tag = 0
        while len(selected_tags) < num_tags:
            while tag in tags:
                tag += 1
            selected_tags.append(tag)
            tags.add(tag)
        return selected_tags

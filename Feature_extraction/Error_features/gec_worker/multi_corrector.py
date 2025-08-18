from typing import List
import logging

from .corrector import Corrector

logger = logging.getLogger(__name__)


class MultiCorrector(Corrector):
    def __init__(self) -> None:
        super().__init__()
        self.correctors = []

    def add_corrector(self, corrector):
        self.correctors.append(corrector)
        if corrector.max_positions is not None:
            if self.max_positions is None:
                self.max_positions = corrector.max_positions
            else:
                self.max_positions = min(self.max_positions, corrector.max_positions)

    def correct(self, sentences: List[str]) -> List[str]:
        temp = sentences
        for corrector in self.correctors:
            temp = corrector.correct(temp)
        return temp

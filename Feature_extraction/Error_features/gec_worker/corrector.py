from abc import ABC, abstractmethod

from typing import List
from .utils import sentence_tokenize, generate_spans
from .dataclasses import Response, Request
import logging
import warnings
import itertools

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )

class Corrector(ABC):
    max_positions = None
    @abstractmethod
    def correct(self, sentences: List[str]) -> List[str]:
        pass

    def process_request(self, request: Request) -> Response:
        sentences, delimiters = sentence_tokenize(
            request.text,
            self.max_positions
        )

        predictions = [correction.strip() if sentences[idx] != '' else '' for idx, correction in enumerate(
            self.correct(sentences))]

        corrected = ''.join(itertools.chain.from_iterable(zip(delimiters, predictions))) + delimiters[-1]
        logger.debug(corrected)

        corrections = generate_spans(sentences, predictions, delimiters)
        response = Response(corrections=corrections, original_text=request.text, corrected_text=corrected)

        logger.debug(response)

        return response

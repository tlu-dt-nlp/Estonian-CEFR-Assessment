import logging
import os
import jamspell
from typing import List

from .corrector import Corrector

logger = logging.getLogger(__name__)


class Speller(Corrector):
    def __init__(self, model_config):
        self.corrector = jamspell.TSpellCorrector()

        if not os.path.isfile(model_config.model_bin):
            logger.info(f"Speller model {model_config.model_bin} not found. Trying to download...")
            model_config.download()
        logger.info(f"Loading model {model_config.model_bin}, this may take several minutes...")
        self.corrector.LoadLangModel(model_config.model_bin)
        logger.info(f"Speller model {model_config.model_bin} loaded")

    def correct(self, sentences: List[str]) -> List[str]:
        return [self.corrector.FixFragment(sentence) for sentence in sentences]

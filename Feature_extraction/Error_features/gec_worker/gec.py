import logging
import os
from typing import List
import warnings

import torch

from .corrector import Corrector
from .modular_interface import ModularHubInterface
from .nelb_interface import NELBHubInterface

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class GEC(Corrector):
    model = None

    def __init__(self, model_config):
        self.model_config = model_config

        try:
            self._load_model()
        except OSError:
            logger.exception("Model loading failed. Trying to download model...")
            self.model_config.download()
            self._load_model()

        self.source_language = self.model_config.source_language
        self.target_language = self.model_config.target_language

        self.max_positions = self.model.max_positions[0] #if self.model_config.task == 'translation' else \
        #self.model.max_positions[f"{self.source_language}-{self.target_language}"][0]

        logger.info("GEC model loaded")

    def _load_model(self):
        sentencepiece_path = os.path.join(self.model_config.sentencepiece_dir, self.model_config.sentencepiece_prefix)
        if self.model_config.type == "nelb":
            self.model = NELBHubInterface.from_pretrained(
            model_path=self.model_config.checkpoint,
            sentencepiece_prefix=sentencepiece_path,
            dictionary_path=self.model_config.dict_dir,
            task=self.model_config.task,
            source_language=self.model_config.source_language,
            target_language=self.model_config.target_language)
        else:
            self.model = ModularHubInterface.from_pretrained(
                model_path=self.model_config.checkpoint,
                sentencepiece_prefix=sentencepiece_path,
                truecase_model=self.model_config.truecase_model,
                dictionary_path=self.model_config.dict_dir,
                task=self.model_config.task,
                source_language=self.model_config.source_language,
                target_language=self.model_config.target_language)

        if torch.cuda.is_available():
            self.model.cuda()
        

    def correct(self, sentences: List[str]) -> List[str]:
        return self.model.translate(sentences,
                                    src_language=self.model_config.source_language,
                                    tgt_language=self.model_config.target_language)


import logging
import os
from typing import List
import stanza

from .corrector import Corrector

logger = logging.getLogger(__name__)


class CorrectionList(Corrector):
    def __init__(self, model_config):
        self.tokenizer = stanza.Pipeline(lang='et', processors='tokenize', tokenize_no_ssplit=True)
        self.corrections={}

        if not os.path.isfile(model_config.model_bin):
            logger.info(f"Speller model {model_config.model_bin} not found. Trying to download...")
            model_config.download()
        logger.info(f"Loading model {model_config.model_bin}")
        f=open(model_config.model_bin, encoding="utf-8")
        f.readline()
        for r in f:
          m=r.split(";")
          self.corrections[m[0]]=m[1].strip()
        f.close()
        logger.info(f"Correction model {model_config.model_bin} loaded")

    def correct(self, sentences: List[str]) -> List[str]:
        result=[]
        for sentence in sentences:
           m=self.tokenizer(sentence)
           sep=[]
           for nr in range(len(m.sentences[0].words)-1): 
             sep.append(m.sentences[0].words[nr+1].start_char-m.sentences[0].words[nr].end_char)
           m2=[self.corrections[w.text] if w.text in self.corrections else w.text for w in m.sentences[0].words]
           s2=[]
           for nr in range(len(m.sentences[0].words)-1):
              s2.append(m2[nr]+(sep[nr]*" "))
           s2.append(m2[-1])
           result.append("".join(s2))
        return result

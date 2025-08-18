import logging
import copy
from typing import Dict, List, Iterator, Any, Optional

from fairseq.data import Dictionary
from fairseq import utils, hub_utils
from fairseq.models.transformer import TransformerModel
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from omegaconf import open_dict, DictConfig
from sentencepiece import SentencePieceProcessor

import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList, Module

logger = logging.getLogger(__name__)

class NELBHubInterface(Module):
    def __init__(
            self,
            models: List[TransformerModel],
            task: TranslationMultiSimpleEpochTask,
            cfg: DictConfig,
            sp_model: SentencePieceProcessor,
    ):
        super().__init__()
        self.sp_model = sp_model
        self.models = ModuleList(models)
        self.task = task
        self.cfg = cfg
        self.dicts: Dict[str, Dictionary] = task.dicts
        self.langs = task.langs

        for model in self.models:
            model.prepare_for_inference_(self.cfg)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @classmethod
    def from_pretrained(
            cls,
            model_path: str,
            sentencepiece_prefix: str,
            dictionary_path: str,
            task: str = "multilingual_translation",
            source_language: str = 'est_Latn',
            target_language: str = 'est_Latn',
    ):
        x = hub_utils.from_pretrained(
            "./",
            checkpoint_file=model_path,
            fixed_dictionaty=f'{dictionary_path}/dict.est_Latn.txt',
            archive_map={},
            lang_pairs=f"{source_language}-{target_language}",
            data_name_or_path=dictionary_path,
            task=task,
            fp16=True,
            source_lang=source_language,
            target_lang=target_language
        )
        
        sp_model = SentencePieceProcessor(model_file=f"{sentencepiece_prefix}.model")

        return cls(
            models=x["models"],
            task=x["task"],
            cfg=x["args"],
            sp_model=sp_model,
        )

    @property
    def device(self):
        return self._float_tensor.device

    def binarize(self, sentence: str, language: str) -> LongTensor:
        return self.dicts[language].encode_line(sentence, add_if_not_exist=False).long()

    def apply_bpe(self, sentence: str) -> str:
        return " ".join(self.sp_model.encode(sentence, out_type=str))


    def string(self, tokens: Tensor, language: str) -> str:
        return self.dicts[language].string(tokens)

    @staticmethod
    def remove_bpe(sentence: str) -> str:
        return sentence.replace(" ", "").replace("\u2581", " ").strip()



    def encode(self, sentence: str, language: str) -> LongTensor:
        bpe_token_sent = self.apply_bpe(sentence)
        logger.debug(f"Preprocessed: {sentence} into {bpe_token_sent}.")
        return self.binarize(bpe_token_sent, language)

    def decode(self, tokens: Tensor, language: str) -> str:
        bpe_token_sent = self.string(tokens, language)
        decoded_sent = self.remove_bpe(bpe_token_sent)
        logger.debug(f"Postprocessed: {bpe_token_sent} into {decoded_sent}.")
        return decoded_sent

    def translate(
            self,
            sentences: List[str],
            src_language: str,
            tgt_language: str,
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = 1000,
    ) -> List[str]:
        """
        :param sentences: list of sentences to be translated
        :param src_language: source language
        :param tgt_language: target language
        :param beam: beam size for the beam search algorithm (decoding)
        :param max_sentences: max number of sentences in each batch
        :param max_tokens: max number of tokens in each batch, all sentences must be shorter than max_tokens.
        :return: list of translations corresponding to the input sentences
        """
        logger.debug(f"Translating from {src_language} to {tgt_language}")
        tokenized_sentences = [self.encode(sentence, src_language) for sentence in sentences]

        batched_hypos = self._generate(
            tokenized_sentences,
            src_language,
            tgt_language,
            beam=beam,
            max_sentences=max_sentences,
            max_tokens=max_tokens
        )
        return [self.decode(hypos, tgt_language) for hypos in batched_hypos]

    def _generate(
            self,
            tokenized_sentences: List[LongTensor],
            src_lang: str,
            tgt_lang: str,
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None,
            skip_invalid_size_inputs=False,
    ) -> List[List[Dict[str, Tensor]]]:
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam

        generator = self.task.build_generator(self.models, gen_args)
        results = []
        for batch in self._build_batches(
                tokenized_sentences,
                skip_invalid_size_inputs=skip_invalid_size_inputs,
                max_sentences=max_sentences,
                max_tokens=max_tokens
        ):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)

            translations = self.task.inference_step(
                generator, self.models, batch
            )
            symbols_to_ignore = generator.symbols_to_strip_from_output
           
            for idx, hypos in zip(batch["id"].tolist(), translations):
                hypos = [i for i in hypos[0]["tokens"] if i.item() not in symbols_to_ignore]
                results.append((idx, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        return outputs

    def _build_batches(
            self,
            tokens: List[LongTensor],
            skip_invalid_size_inputs: bool,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        lengths = LongTensor([t.numel() for t in tokens])
        max_positions =  self.max_positions
        dataset = self.task.build_dataset_for_inference(tokens, lengths)
        batch_iterator = self.task.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

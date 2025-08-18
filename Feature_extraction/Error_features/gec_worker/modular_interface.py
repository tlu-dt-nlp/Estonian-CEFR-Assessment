import logging
import copy
from typing import Dict, List, Iterator, Any, Optional
from .utils import processLine, loadModel
from mosestokenizer import MosesDetokenizer
import stanza
from fairseq.data import Dictionary, LanguagePairDataset, FairseqDataset
from fairseq import utils, search, hub_utils
from fairseq.models.transformer import TransformerModel
from fairseq.tasks.translation import TranslationTask
from fairseq.sequence_generator import SequenceGenerator

from omegaconf import open_dict, DictConfig
from sentencepiece import SentencePieceProcessor

import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList, Module

logger = logging.getLogger(__name__)


class ModularHubInterface(Module):
    def __init__(
            self,
            models: List[TransformerModel],
            task: TranslationTask,
            cfg: DictConfig,
            sp_models: Dict[str, SentencePieceProcessor],
            tokenizer: stanza.Pipeline,
            detokenizer: MosesDetokenizer,
            truecaser: Dict
    ):
        super().__init__()
        self.sp_models = sp_models
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.truecaser = truecaser
        self.models = ModuleList(models)
        self.task = task
        self.cfg = cfg
        self.dicts: Dict[str, Dictionary] = {cfg.task.source_lang : task.src_dict, cfg.task.target_lang : task.tgt_dict} if type(self.task) == TranslationTask else task.dicts
        self.langs = [cfg.task.source_lang, cfg.task.target_lang] if type(self.task) == TranslationTask else task.langs

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
            truecase_model: str,
            dictionary_path: str,
            task: str = "translation",
            source_language: str = 'et0',
            target_language: str = 'et',
    ):
        x = hub_utils.from_pretrained(
            "./",
            checkpoint_file=model_path,
            archive_map={},
            data_name_or_path=dictionary_path,
            task=task,
            source_lang=source_language,
            target_lang=target_language
        )
        
        all_langs = [source_language, target_language] if task == "translation" else [lang for lang in x["task"].langs]
        sp_models = {
            lang: SentencePieceProcessor(
                model_file=f"{sentencepiece_prefix}.{lang}.model"
            ) for lang in all_langs
        }

        tokenizer = stanza.Pipeline(lang='et', processors='tokenize', tokenize_no_ssplit=True) if task == "translation" else None

        detokenizer = MosesDetokenizer('et') if task == "translation" else None
        truecaser = loadModel(truecase_model) if task == "translation" else None


        return cls(
            models=x["models"],
            task=x["task"],
            cfg=x["args"],
            sp_models=sp_models,
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            truecaser=truecaser
        )

    @property
    def device(self):
        return self._float_tensor.device

    def binarize(self, sentence: str, language: str) -> LongTensor:
        return self.dicts[language].encode_line(sentence, add_if_not_exist=False).long()

    def apply_bpe(self, sentence: str, language: str) -> str:
        return " ".join(self.sp_models[language].encode(sentence, out_type=str))

    def truecase(self, sentence: str) -> str:
        return processLine(self.truecaser, sentence)

    def tokenize(self, sentence: str) -> str:
        doc = self.tokenizer(sentence)
        tokens = []
        for sentence in doc.sentences:
            tokens += [ token.text for token in sentence.tokens ]
        return " ".join(tokens)

    def string(self, tokens: Tensor, language: str) -> str:
        return self.dicts[language].string(tokens)

    @staticmethod
    def remove_bpe(sentence: str) -> str:
        return sentence.replace(" ", "").replace("\u2581", " ").strip()

    @staticmethod
    def remove_truecase(sentence: str) -> str:
        return sentence[0].upper() + sentence[1:]

    def remove_tokenization(self, sentence: str) -> str:
        return self.detokenizer(sentence.split(" "))

    def encode(self, sentence: str, language: str) -> LongTensor:
        if self.tokenizer is not None:
            sentence = self.tokenize(sentence)
        if self.tokenizer is not None:
            sentence = self.truecase(sentence)
        bpe_token_sent = self.apply_bpe(sentence, language)
        logger.debug(f"Preprocessed: {sentence} into {bpe_token_sent}.")
        return self.binarize(bpe_token_sent, language)

    def decode(self, tokens: Tensor, language: str) -> str:
        bpe_token_sent = self.string(tokens, language)
        decoded_sent = self.remove_bpe(bpe_token_sent)
        decoded_sent = self.remove_truecase(decoded_sent)
        if self.detokenizer is not None:
            decoded_sent = self.remove_tokenization(decoded_sent)
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
        return [self.decode(hypos[0]["tokens"], tgt_language) for hypos in batched_hypos]

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
        generator = self._build_generator(src_lang, tgt_lang, gen_args)

        results = []
        for batch in self._build_batches(
                tokenized_sentences,
                src_lang,
                tgt_lang,
                skip_invalid_size_inputs=skip_invalid_size_inputs,
                max_sentences=max_sentences,
                max_tokens=max_tokens
        ):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch
            )
            for idx, hypos in zip(batch["id"].tolist(), translations):
                results.append((idx, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        return outputs

    def _build_dataset_for_inference(
            self, src_tokens: List[LongTensor],
            src_lengths: LongTensor,
            src_lang: str,
            tgt_lang: str,
    ) -> FairseqDataset:
        if type(self.task) == TranslationTask:
            return LanguagePairDataset(
                src_tokens, src_lengths, self.dicts[src_lang]
            )
        else:
            return self.task.alter_dataset_langtok(
                LanguagePairDataset(
                    src_tokens, src_lengths, self.dicts[src_lang]
                ),
                src_eos=self.dicts[src_lang].eos(),
                src_lang=src_lang,
                tgt_eos=self.dicts[tgt_lang].eos(),
                tgt_lang=tgt_lang,
            )

    def _build_batches(
            self,
            tokens: List[LongTensor],
            src_lang: str,
            tgt_lang: str,
            skip_invalid_size_inputs: bool,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        lengths = LongTensor([t.numel() for t in tokens])
        max_positions = self.max_positions if type(self.task) == TranslationTask else self.max_positions[f"{src_lang}-{tgt_lang}"]
        batch_iterator = self.task.get_batch_iterator(
            dataset=self._build_dataset_for_inference(tokens, lengths, src_lang, tgt_lang),
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def _build_generator(self, src_lang, tgt_lang, args):
        used_modules = self.models if type(self.task) == TranslationTask else ModuleList([model.models[f"{src_lang}-{tgt_lang}"] for model in self.models])
        return SequenceGenerator(
            used_modules,
            self.dicts[tgt_lang],
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(self.dicts[tgt_lang]),
        )

from pydantic_settings import BaseSettings


class WorkerConfig(BaseSettings):
    """
    Imports general worker configuration from environment variables
    """
    max_input_length: int = 10000

    class Config:
        env_file = '.env'
        env_prefix = 'worker_'



class GECConfig:
    checkpoint = "models/grammar/en-et-de-cs-nelb/checkpoint_1.3b_half.pt"
    dict_dir = "models/grammar/en-et-de-cs-nelb/dicts/"
    sentencepiece_dir = "models/grammar/en-et-de-cs-nelb/sentencepiece/"
    sentencepiece_prefix = "flores200_sacrebleu_tokenizer_spm"
    truecase_model = "no"
    source_language = "est_Latn"
    target_language = "est_Latn"
    task = "translation_multi_simple_epoch"
    type = "nelb"



class SpellConfig:
    model_bin = "models/spelling/etnc19_reference_corpus_model_6000000_lines.bin"



class CorrectionListConfig:
    model_bin = "models/spelling/correction_list.csv"


def get_gec_config():
    model_config = GECConfig()
    return model_config


def get_speller_config():
    model_config = SpellConfig()
    return model_config


def get_correction_list_config():
    model_config = CorrectionListConfig()
    return model_config



worker_config = WorkerConfig()

# Spelling and Grammatical Error Detection for CEFR-based Text Classification

This is a modified version of the [repository](https://github.com/TartuNLP/grammar-worker), which contains code for running Estonian spell-checking and grammatical error correction (GEC) models. This error-correction toolkit was developed in collaboration of the [NLP research group](https://tartunlp.ai/) at the University of Tartu and the [language technology research group](https://elle.tlu.ee/about/people) at the Tallinn University. The project (2021–2023) was funded by the 'Estonian Language Technology' programme.

The GEC implementation uses Transformer-based machine translation to normalize the input text. The model was trained using custom [modular NMT implementation of FairSeq](https://github.com/TartuNLP/fairseq). Statistical spelling correction relies on the Jamspell algorithm that analyzes word contexts based on a trigram language model.

## Usage

The models should be downloaded from Hugging Face and placed into respective subdirectories of the `models` directory:

*   The GEC model: https://huggingface.co/tartuNLP/en-et-de-cs-nelb
*   The spell-checking model: https://huggingface.co/Jaagup/etnc19_reference_corpus_model_6000000_lines

The script `error_finder.py` calculates the following features in given texts:

*   Ratio of spell-corrected words
*   Ratio of spell-corrected sentences
*   Avg. number of spelling corrections per sentence
*   Avg. ratio of spell-corrected words in a sentence

*   Ratio of words overlapping with a grammar correction\*
*   Ratio of sentences with grammar corrections
*   Number of grammar corrections per word
*   Number of grammar corrections per sentence
*   Avg. ratio of words in a sentence overlapping with a grammar correction

\* Corrections made by the grammar-checking model also include spelling corrections but error types are not distinguished.

The results are saved into the file `error_data.csv`, which can be merged with the datasets containing other linguistic features using the script `dataset_merger.py` (in the `Feature_extraction` directory).

For this experiment, the Linux operating system and Python 3.8.10 were applied.

## References

*   Allkivi-Metsoja, K., & Kippar, J. (2023). Spelling Correction for Estonian Learner Language. In Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa) (pp. 782−788). https://aclanthology.org/2023.nodalida-1.79 
*   Luhtaru, A., Korotkova, A., & Fishel, M. (2024). No Error Left Behind: Multilingual Grammatical Error Correction with Pre-trained Translation Models. Proceedings of EACL 2024 (pp. 1209–1222). https://aclanthology.org/2024.eacl-long.73

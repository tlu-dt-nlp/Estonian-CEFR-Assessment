import os
from statistics import mean
import stanza
nlp = stanza.Pipeline('et', processors='tokenize, pos')
from pprint import pprint
from dataclasses import asdict
from gec_worker import GEC, get_gec_config
from gec_worker import Speller, get_speller_config
from gec_worker import CorrectionList, get_correction_list_config
from gec_worker.dataclasses import Request
from gec_worker import MultiCorrector

#Load grammatical error correction model
gec = GEC(get_gec_config())

#Load spell-checking model
speller = Speller(get_speller_config())

#Load the corrections of typical L2 spelling errors
correction_list = CorrectionList(get_correction_list_config())

#Pipeline including the correction list and spell-checker
multi_corrector = MultiCorrector()
multi_corrector.add_corrector(correction_list)
multi_corrector.add_corrector(speller)

#Calculating error features and saving feature values
with open('Feature_extraction/Error_features/error_data.txt', 'w') as output:
  output.write('file_name,spell_word_ratio,spell_sent_ratio,spell_errors_per_sent,\
    avg_spell_word_ratio,error_word_ratio,error_sent_ratio,errors_per_word,errors_per_sent,\
    avg_error_word_ratio\n')

  directories = ['Texts/Train_test1/A2', 'Texts/Train_test1/B1', 
    'Texts/Train_test1/B2', 'Texts/Train_test1/C1']
  for directory in directories:
    for entry in os.scandir(directory):
      if entry.name.endswith('.txt'):
        with open(directory+'/'+ entry.name, 'r') as input:
          source_text = input.read().rstrip()
          doc = nlp(source_text)

          word_count = 0
          sent_count = 0
          sent_with_errors = 0
          sent_with_spell_errors = 0
          words_with_errors = 0
          words_with_spell_errors = 0
          error_ratio_list = []
          spell_ratio_list = []

          for sent in doc.sentences:
            error_words_in_sent = 0
            sent_count += 1
            sent_word_count = len([word for word in sent.words if word.upos != 'PUNCT'])
            word_count += sent_word_count
            sent_text = sent.text
            request = Request(text=sent_text, language='et')
            spell_response = multi_corrector.process_request(request)
            spell_response = asdict(spell_response)
            gec_response = gec.process_request(request)
            gec_response = asdict(gec_response)

            if spell_response['corrections']:
              spell_error_count = len(spell_response['corrections'])
              sent_with_spell_errors += 1
              words_with_spell_errors += spell_error_count
              spell_ratio_list.append(spell_error_count / sent_word_count)
            else:
              spell_ratio_list.append(0)

            if gec_response['corrections']:
              error_count = len(gec_response['corrections'])
              sent_with_errors += 1
              for error in gec_response['corrections']:
                error_text = error['span']['value']
                error_words_in_sent += len(error_text.split())
                words_with_errors += len(error_text.split())
              error_ratio_list.append(error_words_in_sent / sent_word_count)
            else:
              error_ratio_list.append(0)
          
          #Ratio of spell-corrected words
          spell_word_ratio = words_with_spell_errors / word_count
          #Ratio of spell-corrected sentences
          spell_sent_ratio = sent_with_spell_errors / sent_count
          #Avg. number of spelling corrections per sentence
          spell_errors_per_sent = words_with_spell_errors / sent_count
          #Avg. ratio of spell-corrected words in a sentence
          avg_spell_word_ratio = mean(spell_ratio_list)

          #Ratio of words overlapping with a grammar correction
          error_word_ratio = words_with_errors / word_count
          #Ratio of sentences with grammar corrections
          error_sent_ratio = sent_with_errors / sent_count
          #Number of grammar corrections per word
          errors_per_word = error_count / word_count
          #Number of grammar corrections per sentence
          errors_per_sent = error_count / sent_count
          #Avg. ratio of words in a sentence overlapping with a grammar correction
          avg_error_word_ratio = mean(error_ratio_list)

          output.write(entry.name + ',' + str(spell_word_ratio) + ',' +
            str(spell_sent_ratio) + ',' + str(spell_errors_per_sent) + ',' +
            str(avg_spell_word_ratio) + ',' + str(error_word_ratio) + ',' +
            str(error_sent_ratio) + ',' + str(errors_per_word) + ',' +
            str(errors_per_sent) + ',' + str(avg_error_word_ratio) + '\n')
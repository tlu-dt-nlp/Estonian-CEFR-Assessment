import math
import os
import pandas as pd
import stanza_tagger as st
import linguistic_functions as lf

#Reading the list of variable labels
with open('Feature extraction/variable_labels.txt', 'r') as var_list:
    var_labels = var_list.read().splitlines()

#Creating a list to store the data rows
dataset = []
dataset.append(','.join(var_labels)+'\n')

#Tagging the texts with Stanza and reading the CoNLL-U format output into DataFrame
directories = ['Texts/Train_test1/A2', 'Texts/Train_test1/B1', 
    'Texts/Train_test1/B2', 'Texts/Train_test1/C1'] #texts of the main dataset
#directories = ['Texts/Test2/A2', 'Texts/Test2/B1', 'Texts/Test2/B2_arg_writings',
    #'Texts/Test2/B2_semiformal_letters', 'Texts/Test2/C1'] #texts of the additional test set
for directory in directories:
	for entry in os.scandir(directory):
		if entry.name.endswith('.txt'):
			file_name = entry.name
			tagged_text = st.tag_text(directory, file_name)
			with open(directory+'/tagged/'+file_name, 'w') as output_f:
				output_f.write('Index\tWord\tLemma\tUpos\tXpos\tFeats\n')
				output_f.write(tagged_text)
			text = pd.read_csv(directory+'/tagged/'+file_name, sep = '\t', engine='python')

			#Creating a dictionary to store variable values
			var = {}
			for label in var_labels:
				var[label] = None

			#Defining metadata variables based on the file names (main dataset)
			var['file_name'] = file_name
			var['prof_level'] = file_name[:2]
			var['exam_time'] = file_name.split('_')[1]
			if var['prof_level'] == 'C1':
				var['text_type'] = 'argumentative writing'
			else:
				if var['prof_level'] == 'A2':
					genres_by_exam = {
						'personal letter': ['2018I', '2018II', '2018III', '2020II'],
						'description/narration': ['2018IV', '2020IV']
					}
				elif var['prof_level'] == 'B1':
					genres_by_exam = {
						'personal letter': ['2018II', '2020I', '2020II'],
						'description/narration': ['2018I', '2018III', '2018IV']
					}
				elif var['prof_level'] == 'B2':
					genres_by_exam = {
						'personal letter': ['2018I', '2018III'],
						'formal letter': ['2018II', '2018IV'],
						'argumentative writing': ['2020II', '2020IV']
					}
				for key, values in genres_by_exam.items():
					if var['exam_time'] in values:
						var['text_type'] = key
            
			#Defining metadata variables based on the directories (test set 2)
			#var['file_name'] = file_name
			#var['prof_level'] = directory.split('/')[-1][:2]
			#if var['prof_level'] == 'A2' or var['prof_level'] == 'B1':
				#var['text_type'] = 'description/narration'
			#elif (var['prof_level'] == 'B2' and directory.endswith('arg_writings'))\
				#or var['prof_level'] == 'C1':
					#var['text_type'] = 'argumentative writing'
			#elif var['prof_level'] == 'B2' and directory.endswith('semiformal_letters'):
					#var['text_type'] = 'formal letter'
            

			#Surface features
            
			#counting sentences
			var['sent_count'] = text[text.Index == 1].Index.count()
			#removing punctuation and counting words
			text = text[text.Xpos != 'Z']
			var['word_count'] = text.Word.count()
			#mean sentence length
			var['sent_length'] = var['word_count']/var['sent_count']
			#defining word lengths to calculate mean length
			text['WordLength'] = [len(str(word)) for word in text.Word]
			var['word_length'] = text.WordLength.mean()
			#proportion of long words and LIX index
			long_words = text[text.WordLength > 6].Word.count() / var['word_count']
			var['LIX'] = round(var['sent_length'] + 100 * long_words)
            
            #counting syllables and polysyllabic words
            #calculating SMOG and Flesch-Kincaid indices
			syllables = 0
			polysyllabic_words = 0
			text_word_list = text['Word'].tolist()
			for word in text_word_list:
				syllabified = lf.syllabify(word)
				if len(syllabified) > 0:
					word_syllables = syllabified[0].count('-') + 1
				syllables = syllables + word_syllables
				if word_syllables >= 3:
					polysyllabic_words += 1
			var['SMOG'] = 1.0430 * math.sqrt(polysyllabic_words * (30 / var['sent_count'])) + 3.1291
			var['FK'] = 0.39 * (var['word_count'] / var['sent_count']) + \
                11.8 * (syllables / var['word_count']) - 15.59
			var['syllable_count'] = syllables
			var['polysyllables'] = polysyllabic_words


			#Lexical features and parts-of-speech proportions
            
			#counting lemmas
			var['lemma_count'] = int(text.groupby('Lemma').Lemma.count().to_frame().count())
			#removing proper nouns, symbols, and numbers to measure lexical diversity
			cleaned_text = text[(text.Upos != 'PROPN')&(text.Upos != 'SYM')]
			cleaned_text = cleaned_text[~cleaned_text['Feats'].str.contains('NumForm=Digit',\
				na=False)]
			cleaned_lemmas = cleaned_text.groupby('Lemma').Lemma.count().count()
			cleaned_words = cleaned_text.groupby('Lemma').Lemma.count().sum()
			cleaned_word_list = cleaned_text['Word'].tolist()

			#lexical diversity indices
			var['TTR'] = cleaned_lemmas / cleaned_words
			var['RTTR'] = cleaned_lemmas / math.sqrt(cleaned_words)
			var['Uber'] = (math.log(cleaned_words))**2 / (math.log(cleaned_words)\
				- math.log(cleaned_lemmas))
			var['Maas'] = (math.log(cleaned_words) - math.log(cleaned_lemmas))\
				/ (math.log(cleaned_words))**2
			var['MTLD'] = lf.mtld(cleaned_word_list, ttr_threshold = 0.72)
			verb_count = text[text.Xpos == 'V'].groupby('Lemma').Lemma.count().sum()
			verb_lemmas = text[text.Xpos == 'V'].groupby('Lemma').Lemma.count().count()
			var['CVV'] = verb_lemmas / math.sqrt(2 * verb_count)
			
            #lexical density
			var['LD'] = lf.lexical_density(cleaned_text)
			
            #part-of-speech proportions
			pos_tags = ['A', 'D', 'I', 'J', 'K', 'N', 'P', 'S', 'V']
			for tag in pos_tags:
				var[tag] = lf.pos_ratio(text, tag, var['word_count'])
			var['S_Prop'] = lf.pos_ratio(text, 'PROPN', var['word_count'])

			#type-token ratio by part-of-speech
			for tag in pos_tags:
				var[tag+'_TTR'] = lf.pos_ttr(text, tag)
			
            #average noun abstractness on the scale of 1-3
			noun_data = text[text.Xpos == 'S']
			abstractness_data = lf.request_abstr_freq(noun_data)
			ab_sum = 0
			ab_count = 0
			#if the API does not respond, missing value is stored and file name is logged
			if abstractness_data:
				for word in abstractness_data['wordAnalysis']:
					lemma = word['lemmas'][0]
					if lemma['abstractness']:
						ab_sum += lemma['abstractness']
						ab_count += 1
			else:
				with open('Feature extraction/abstr_error_log.txt', 'a') as log:
					log.write(file_name+'\n')
			if ab_count > 0: 
				var['S_abstr'] = ab_sum / ab_count
			else:
				var['S_abstr'] = None

            #proportion of words not among the 1,000-5,000 most frequent words in Estonian
			freq_data = lf.request_abstr_freq(text)
			rare_boundaries = [5000, 4000, 3000, 2000, 1000]
			freq_boundaries = [220, 301, 447, 747, 1651]
			for i in range(len(rare_boundaries)):
				if freq_data:
					var['rare_'+str(rare_boundaries[i])] = \
						lf.rare_ratio(freq_data, freq_boundaries[i], var['word_count'])
				else:
					var['rare_'+str(rare_boundaries[i])] = None

			
            #Morphological features of nominal words
			
			#proportion of compound words
			var['compounds'] = text[text['Lemma'].str.contains('_', na=False)].Lemma.count()\
				/ var['word_count']            
            
            #nominal features
			text_nominals = text[(text.Xpos == 'S')|(text.Xpos == 'A')|(text.Xpos == 'P')|
				(text.Xpos == 'N')]
			nominal_count = text_nominals.Word.count()
			nominal_feats = lf.feats_table(text_nominals)
			var['n_cases'] = lf.case_count(nominal_feats)
			nominal_tags = ['Case=Nom', 'Case=Gen', 'Case=Par', 'Case=Add', 'Case=Ill',
				'Case=Ine', 'Case=Ela', 'Case=All', 'Case=Ade', 'Case=Abl', 'Case=Tra',
				'Case=Ter', 'Case=Ess', 'Case=Abe', 'Case=Com', 'Number=Sing','Number=Plur']
			for tag in nominal_tags:
				var['n_'+tag.split('=')[1]] = lf.feat_ratio(nominal_feats, tag, nominal_count)
			var['n_AddIll'] = var['n_Add'] + var['n_Ill']
			
            #noun features
			noun_count = lf.pos_freq(text, 'S')
			noun_feats = lf.feats_table(noun_data)
			var['S_cases'] = lf.case_count(noun_feats)
			for tag in nominal_tags:
				var['S_'+tag.split('=')[1]] = lf.feat_ratio(noun_feats, tag, noun_count)
			var['S_AddIll'] = var['S_Add'] + var['S_Ill']
			
            #adjective features
			adj_count = lf.pos_freq(text, 'A')
			adj_feats = lf.feats_table(text, 'A')
			var['A_cases'] = lf.case_count(adj_feats)
			for tag in nominal_tags:
				if tag != 'Case=Add':
					var['A_'+tag.split('=')[1]] = lf.feat_ratio(adj_feats, tag, adj_count)
			adj_tags = ['Degree=Pos', 'Degree=Cmp', 'Degree=Sup']
			for tag in adj_tags:
				var['A_'+tag.split('=')[1]] = lf.feat_ratio(adj_feats, tag, adj_count)
			
            #pronoun features
			pron_count = lf.pos_freq(text, 'P')
			pron_feats = lf.feats_table(text, 'P')
			var['P_cases'] = lf.case_count(pron_feats)
			for tag in nominal_tags:
				if tag != 'Case=Add':
					var['P_'+tag.split('=')[1]] = lf.feat_ratio(pron_feats, tag, pron_count)
			var['P_Reflex'] = lf.feat_ratio(pron_feats, 'Reflex=Yes', pron_count)\
				+ lf.feat_ratio(pron_feats, 'Poss=Yes', pron_count)
			var['P_Prs'] = lf.feat_ratio(pron_feats, 'PronType=Prs', pron_count)\
				- var['P_Reflex']
			var['P_Dem'] = lf.feat_ratio(pron_feats, 'PronType=Dem', pron_count)
			var['P_Ind'] = lf.feat_ratio(pron_feats, 'PronType=Ind', pron_count)\
					+ lf.feat_ratio(pron_feats, 'PronType=Tot', pron_count)
			var['P_IntRel'] = lf.feat_ratio(pron_feats, 'PronType=Int,Rel', pron_count)\
					+ lf.feat_ratio(pron_feats, 'PronType=Rel', pron_count)
			var['P_Rcp'] = lf.feat_ratio(pron_feats, 'PronType=Rcp', pron_count)

			#verb features
			verb_count = lf.pos_freq(text, 'V')
			verb_feats = lf.feats_table(text, 'V')
			verb_tags = ['VerbForm=Fin', 'Mood=Ind', 'Mood=Cnd', 'Mood=Imp', 'Tense=Pres',
				'Tense=Past', 'Number=Sing', 'Number=Plur', 'Polarity=Neg', 'Voice=Pass',
				'VerbForm=Inf', 'VerbForm=Part', 'VerbForm=Conv', 'Person=1', 'Person=2',
				'Person=3']
			for tag in verb_tags:
				if tag.startswith('Person'):
					var['V_Prs'+tag[-1]] = lf.feat_ratio(verb_feats, tag, verb_count)
				else:
					var['V_'+tag.split('=')[1]] = lf.feat_ratio(verb_feats, tag, verb_count)
			var['V_NonFin'] = var['V_Inf'] + var['V_Part'] + var['V_Conv']


			#Conjunction and adposition subtypes
			conj_count = lf.pos_freq(text, 'J')
			var['J_Crd'] = lf.pos_ratio(text, 'CCONJ', conj_count)
			var['J_Sub'] = lf.pos_ratio(text, 'SCONJ', conj_count)
			adp_count = lf.pos_freq(text, 'K')
			adp_feats = lf.feats_table(text, 'K')
			var['K_Prep'] = lf.feat_ratio(adp_feats, 'AdpType=Prep', adp_count)
			var['K_Post'] = lf.feat_ratio(adp_feats, 'AdpType=Post', adp_count)


			#Writing variables values into dataset rows
			datarow = next(iter(var.values()))
			for index, value in enumerate(var.values()):
				if index > 0:
					datarow = datarow + ',' + str(value)
			dataset.append(datarow+'\n')

#Writing the data into a csv-file
with open('Feature extraction/dataset.csv', 'w') as data_file:
	data_file.writelines(dataset)
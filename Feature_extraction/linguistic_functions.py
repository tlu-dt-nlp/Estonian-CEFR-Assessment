import os
import json
import requests
import pandas as pd
pd.options.mode.chained_assignment = None

def pos_freq(data: pd.DataFrame, pos: str) -> int:
	'''Calculate the frequency of a given part-of-speech.'''
	if len(pos) > 1:
		posFreq = data[data.Upos == pos].Word.count()
	else:
		posFreq = data[data.Xpos == pos].Word.count()
	return posFreq

def pos_ratio(data: pd.DataFrame, pos: str, textLength: int) -> float:
	'''Calculate the proportion of a given part-of-speech.'''
	if len(pos) > 1:
		posRatio = data[data.Upos == pos].Word.count() / textLength
	else:
		posRatio = data[data.Xpos == pos].Word.count() / textLength
	return posRatio

def pos_ttr(data: pd.DataFrame, pos: str) -> float:
	'''Calculate the type-token ratio of a given part-of-speech.'''
	pos_lemmas = data[data.Xpos == pos].groupby('Lemma').Lemma.count().to_frame()
	pos_count = pos_lemmas.Lemma.sum()
	if pos_count > 0:
		ttr = pos_lemmas.Lemma.count() / pos_lemmas.Lemma.sum()
	else:
		ttr = 0
	return ttr

def feats_table(data: pd.DataFrame, pos: str | None = None) -> pd.DataFrame:
	'''Compile a frequency table of morphological features. 
	A certain part-of-speech can be given as an optional argument.'''
	if pos:
		featsTable = data[data.Xpos == pos].groupby('Feats').Feats.count().to_frame()
	else:
		featsTable = data.groupby('Feats').Feats.count().to_frame()
	featsTable.rename(columns = {'Feats':'Freq'}, inplace=True)
	featsTable['Feats'] = featsTable.index
	return featsTable

def feat_ratio(data: pd.DataFrame, feature: str, wordCount: int) -> float:
	'''Calculate the proportion of a given morphological feature.'''
	if wordCount > 0:
		featRatio = data[data['Feats'].str.contains(feature)].Freq.sum() / wordCount
	else:
		featRatio = 0
	return featRatio

def case_count(data: pd.DataFrame) -> int:
	'''Calculate the number of different case forms represented.'''
	data = data[data['Feats'].str.contains('Case=')]
	data['Feats'] = data['Feats'].str.split('|')
	data['Case'] = [feat[0] for feat in data.Feats]
	cases = data.groupby('Case').Freq.sum().to_frame()
	caseCount = len(cases.index)
	return caseCount

def lexical_density(data: pd.DataFrame) -> float:
	'''Calculate lexical density, i.e., the proportion of content words.'''
	#Cleaning text lemmas and creating a lemma list.
	data['Lemma'] = data['Lemma'].str.replace('_', '')
	data['Lemma'] = data['Lemma'].str.replace('=', '')
	textLemmas = data['Lemma'].tolist()
	#Comparing text lemmas to the list of Estonian stopwords for lemmatized text
	#(Uiboaed 2018, https://datadoi.ee/handle/33/78)
	stopLemmas = pd.read_csv('Feature_extraction/estonian-stopwords-lemmas.txt')
	stopLemmas = set(stopLemmas['Stoplemma'].tolist())
	functionWords = 0
	for i in range(len(textLemmas)):
		search = textLemmas[i]
		if search in stopLemmas:
			functionWords += 1
	textWords = data.Word.count()
	lexDensity = (textWords - functionWords) / textWords
	return lexDensity

def request_abstr_freq(data: pd.DataFrame) -> dict[str, int] | None:
	'''Make a request to the API of the Speed reading software of University of Tartu 
	to analyze noun abstractness and rareness of words. 
	The abstractness rating is based on a 3-point scale (see Mikk et al., 2003: http://hdl.handle.net/10062/50110). 
	Word frequency data is based on the balanced subcorpus of the Estonian Reference Corpus.'''
	#Cleaning text lemmas and creating a lemma list for the request.
	chars_to_remove = ['_', '=', '\(', ')', '"', '&']
	for char in chars_to_remove:
		data['Lemma'] = data['Lemma'].str.replace(char, '', regex=True)
	data['Lemma'] = data['Lemma'].str.replace("'", "")
	lemmas = data['Lemma'].tolist()
	request="""
	curl --header 'Content-Type: application/json' \
		--request POST \
		--data '{"language": "estonian", "text": "%s"}' \
		https://kiirlugemine.keeleressursid.ee/api/analyze""" % lemmas
	response = os.popen(request).read()
	try:
		return json.loads(response)
	except:
		return None

def rare_ratio(data: dict[str, int], freqBoundary: int, textLength: int) -> float:
	'''Calculate the proportion of words that have a smaller frequency in the balanced corpus than the given threshold.
	The input data is based on the POST request to https://kiirlugemine.keeleressursid.ee/api/analyze.'''
	rareCount = 0
	for word in data['wordAnalysis']:
		lemma = word['lemmas'][0]
		if lemma['frequency']:
			if lemma['frequency'] < freqBoundary:
				rareCount += 1
	rareRatio = rareCount / textLength
	return rareRatio

def mtld(word_list: list[str], ttr_threshold: float) -> float:
	'''Calculate the lexical diversity index MTLD (McCarthy & Jarvis, 2010).
	The function is taken from the bachelor's thesis of Simon Berner (2022, Tallinn University).'''
	current_ttr = 1.0
	token_count = 0
	type_count = 0
	types = set()
	factors = 0.0
	for token in word_list:
		token = token.lower()
		token_count = token_count + 1
		if token not in types:
			type_count = type_count + 1
			types.add(token)
		current_ttr = type_count / token_count
		if current_ttr <= ttr_threshold:
			factors = factors + 1
			token_count = 0
			type_count = 0
			types = set()
			current_ttr = 1.0
	excess = 1.0 - current_ttr
	excess_val = 1.0 - ttr_threshold
	factors += excess / excess_val
	if factors != 0:
		return len(word_list) / factors
	return -1

def syllabify(text: str) -> list[str]:
	'''Syllabify a word or text. Syllabification is based on the bacherlor's thesis of Robin Kukke (2024, Tallinn University).'''
	response = requests.post("https://elle.tlu.ee/api/texts/silbid/", json={"tekst":text}, headers={"Content-Type": "application/json; charset=utf-8"})
	word_list = json.loads(response.text)
	return word_list


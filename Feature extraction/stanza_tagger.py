import stanza
nlp = stanza.Pipeline('et')
def tag_text(directory: str, filename: str) -> str:
    '''This function performs tokenization, sentence segmentation, lemmatization, and
    part-of-speech and morphological tagging of a given text.'''
    with open(directory+'/'+filename, 'r') as input_f:
        text = input_f.read().rstrip()
        doc = nlp(text)
        analysis = '\n'.join(
            [f'{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats}' for sent in doc.sentences for word in sent.words])
    return analysis
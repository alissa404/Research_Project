from __future__ import unicode_literals
import string
'''
import spacy
from spacy.tokenizer import Tokenizer

def normalize(text):
	#STOP_WORDS = ['the', 'a', 'an']
	nlp = spacy.load('/spacy/zh_model')
	nlp.add_pipe(nlp.create_pipe('sentencizer'))
	text = text.lower().strip()
	doc = nlp(text)
	filtered_sentences = []
	for sentence in doc.sents:
		filtered_tokens = list()
		for i, w in enumerate(sentence):
			s = w.string.strip()
			if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
				continue
			if s not in STOP_WORDS:
				s = s.replace(',', '.')
				filtered_tokens.append(s)
		filtered_sentences.append(' '.join(filtered_tokens))
	return filtered_sentences

normalized_text = normalize(text)
'''
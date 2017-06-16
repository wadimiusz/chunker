import sklearn_crfsuite, codecs, time, sys, itertools
from sklearn_crfsuite import metrics

import numpy as np
import scipy as sp

from random import shuffle

start = time.time()

def tag2feature(token, tag):
	tags = tag.split(',')
	result = {}
	
	result['upper'] = token.isupper()
	result['lower'] = token.islower()
	result['digit'] = token.isdigit()
	result['title'] = token.istitle()
	
	pos = [tag for tag in tags if tag.isupper()]
	result['POS'] = pos[0] if len(pos) > 0 else ''
	
	case = [tag for tag in tags if tag in ('nom', 'gen', 'dat', 'acc', 'abl', 'loc')]
	result['case'] = case[0] if len(case) > 0 else ''
	
	gender = [tag for tag in tags if tag in ('m', 'f', 'n')]
	result['gender'] = gender[0] if len(gender) > 0 else ''
	
	tense = [tag for tag in tags if tag in ('praet', 'praes', 'fut')]
	result['tense'] = tense[0] if len(tense) > 0 else ''
	
	number = [tag for tag in tags if tag in ('sg', 'pl')]
	result['number'] = number[0] if len(number) > 0 else ''
	
	person = [tag for tag in tags if tag in ('1s', '2s', '3s', '1p', '2p', '3p')]
	result['person'] = person[0] if len(person) > 0 else ''
	
	anim = [tag for tag in tags if tag in ('anim', 'inan')]
	result['anim'] = anim[0] if len(anim) > 0 else ''
	
	voice = [tag for tag in tags if tag in ('act', 'pass')]
	result['voice'] = voice[0] if len(voice) > 0 else ''
	
	return result

word2token = lambda word: [token for token, postag, label in word]
word2postag = lambda word: [(token, postag) for token, postag, label in word]
word2label = lambda word: [label for token, postag, label in word]

sent2token = lambda sentence: [word2token(word) for word in sentence]
sent2postag = lambda sentence: [word2postag(word) for word in sentence]
sent2label = lambda sentence: [word2label(word) for word in sentence]

sent2feature = lambda sentence: [tag2feature(token, tag) for token, tag in sentence]

f = codecs.open("chunking_data.txt", mode = "r", encoding = "utf-8")
sentences = [[word.split('\t') for word in sentence.split('\n') if len(word) > 1] for sentence in f.read().split('\n\n')]
shuffle(sentences)
X_raw = sent2postag(sentences)
y = sent2label(sentences)

border = len(X_raw) // 10

X_train_raw = X_raw[border:]
y_train = y[border:]
X_train = [sent2feature(sentence) for sentence in X_train_raw]
X_test_raw = X_raw[:border]
X_test = [sent2feature(sentence) for sentence in X_test_raw]
y_test = y[:border]

crf = sklearn_crfsuite.CRF(
algorithm='lbfgs',
c1=0.1,
c2=0.1,
max_iterations=200,
all_possible_transitions=True
)
crf.fit(X_train, y_train)

labels = list(crf.classes_)
labels.remove('O')
labels

y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]) if len(name) > 0 else (name,''))
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

#print(vectorizer.get_feature_names())

end = time.time()
duration = end - start
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print("The whole process took %d hours, %d minutes and %d seconds" % (hours, minutes, seconds))

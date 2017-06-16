import codecs, time, sys

import numpy as np
import scipy as sp

from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

start = time.time()

comma_split = lambda x: x.split(',')

f = codecs.open("chunking_data.txt", mode = "r", encoding = "utf-8")
table = [line.split('\t') for line in f.read().split('\n') if line != '']
table = [(a, b, c) for (a, b, c) in table if c != '']
f.close()

vectorizer = CountVectorizer(analyzer = comma_split)
X_raw = [b for (a, b, c) in table]
y = [c for (a, b, c) in table]
border = len(X_raw) // 10

X_train_raw = X_raw[border:]
y_train = y[border:]
X_test_raw = X_raw[:border]
y_test = y[:border]

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

l_word = X_train.shape[1]
zeros = np.zeros((l_word))

if len(sys.argv) > 2:
	window = int(sys.argv[1])
else:
	window = 3
context = lambda n, text: sp.sparse.hstack([sp.sparse.vstack([zeros] * i + [text] + [zeros] * (n - i)) for i in range(n+1)])
X_train = context(window * 2, X_train) #число должно быть чётным
y_train = [''] * window + y_train + [''] * window

X_test = context(window * 2, X_test) #число должно быть чётным
y_test = [''] * window + y_test + [''] * window

if len(sys.argv) > 2:
	max_iter = int(sys.argv[2])
else:
	max_iter = 200
clf = MLPClassifier(verbose = True, max_iter = max_iter)
print('Начинаю обучение...')
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

labels = [x for x in set(y_test)]
#можно смотреть качество распознавания только, допустим, начал чанков, если писать labels = [x for x in set(y_test) if 'B' in x]

print("Labels:")
print(*labels, sep = ', ')
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average = 'micro', labels = labels)
recall = recall_score(y_test, predicted, average = 'micro', labels = labels)
f1 = f1_score(y_test, predicted, average = 'micro', labels = labels)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

end = time.time()
duration = end - start
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)
print("The whole process took %d hours, %d minutes and %d seconds" % (hours, minutes, seconds))

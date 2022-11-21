import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
nltk.download('punkt')
import pandas as pd
import sys
data_path = '.'
train_data_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

etichete_unice = train_data_df['label'].unique()
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta] = idx
    id2label[idx] = eticheta

labels = []

for eticheta in train_data_df['label']:
    labels.append(label2id[eticheta])
labels = np.array(labels)


def proceseaza(text):
    text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
    text = text.lower()
    text = text.replace('\n', ' ').strip().lower()
    text_in_cuvinte = text.split(' ')
    # text_in_cuvinte = word_tokenize(text)
    list = []
    for word in text_in_cuvinte:
        if len(word) > 3:
            list.append(word)
    return list


# cuvintele rezultate din functia de preprocesare:
# exemple_italian = train_data_df[train_data_df['language'] == 'italiano']
# print(exemple_italian)
# text_italian = exemple_italian['text'].iloc[0]
data = train_data_df['text'].apply(lambda text: proceseaza(text))
data2 = test_data_df['text'].apply(lambda text: proceseaza(text))


# nr_test = int(35 / 100 * len(train_data_df))
nr_test = 13900
nr_ramase = len(data) - nr_test
nr_valid = 0
nr_train = len(train_data_df)
indici = np.arange(0, len(train_data_df))
np.random.shuffle(indici)
train_data = data[indici[:nr_train]]
train_labels = labels[indici[:nr_train]]

valid_data = data[indici[nr_train: nr_train + nr_valid]]
valid_labels = labels[indici[nr_train: nr_train + nr_valid]]

test_data = data2
test_labels = labels[indici[nr_train + nr_valid:]]


def count_most_common(how_many, texte_preprocesate):
    counter = Counter()
    for text in texte_preprocesate:
        counter.update(text)
    cuvinte_caracteristice = []
    for cuvant, frecventa in counter.most_common(how_many):
        if cuvant.strip():
            cuvinte_caracteristice.append(cuvant)
    return cuvinte_caracteristice


def build_id_word_dicts(cuvinte_caracteristice):
    word2id = {}
    id2word = {}
    for idx, cuv in enumerate(cuvinte_caracteristice):
        word2id[cuv] = idx
        id2word[idx] = cuv

    return word2id, id2word



def featurize(text_preprocesat, id2word):
    ctr = Counter(text_preprocesat)
    features = np.zeros(len(id2word))
    for idx in range(0, len(features)):
        cuvant = id2word[idx]
        features[idx] = ctr[cuvant]
    return features


def featurize_multi(texte, id2word):
    all_features = []
    for text in texte:
        all_features.append(featurize(text, id2word))
    return np.array(all_features)


cuvinte_caracteristice = count_most_common(5000, train_data)

word2id, id2word = build_id_word_dicts(cuvinte_caracteristice)

X_train = featurize_multi(data, id2word)
X_valid = featurize_multi(valid_data, id2word)
X_test = featurize_multi(test_data, id2word)

print(cuvinte_caracteristice)

from sklearn import svm

#model = svm.LinearSVC(C=0.1, max_iter=1000)
model = MultinomialNB(alpha=0.5,fit_prior=True,class_prior=None)

model.fit(X_train, train_data_df['label'])
tpreds = model.predict(X_test)
#print('Acuratete pe validare ', accuracy_score(valid_labels, vpreds))
#print('Acuratete pe test ', accuracy_score(test_labels, tpreds))

with open('submission.csv', 'w') as f:
    f.write('id,label\n')
    for id, label in enumerate(model.predict(X_test)):

        if id == 13860:
            break
        else:
            f.write(f'{id + 1},{label}\n')

print("gata")

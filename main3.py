import csv
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
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sys
nltk.download('punkt')
import pandas as pd

data_path = '.'
train_data_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
# print(train_data_df.head())
# print(len(train_data_df))
# print(len(train_data_df[train_data_df.label == 'England']))

train_data_df.loc[train_data_df['label']=='England', 'label'] = 1
train_data_df.loc[train_data_df['label']=='Scotland', 'label'] = 2
train_data_df.loc[train_data_df['label']=='Ireland', 'label'] = 3

df_x = train_data_df['language']
df_y = train_data_df['text']
df_z = train_data_df['label']

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(df_y,df_z, test_size=0.35, random_state=4)
x_train_cv = cv.fit_transform(x_train)
a = x_train_cv.toarray()
print(a)





sys.exit()



cv = CountVectorizer()
#x_train, x_test, y_train, y_test = train_test_split(test_data_df,test_data_df, test_size=0.2, random_state=4)
nr_test = 13800
nr_ramase = len(data) - nr_test
nr_valid = int(15 / 100 * nr_ramase)
nr_train = nr_ramase - nr_valid
x_train




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

#  https://www.youtube.com/watch?v=RZYjsw6P4nI&ab_channel=TheSemicolon
sys.exit()
def proceseaza(text):
    text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)

    text = text.lower()
    text = text.replace('\n', ' ').strip().lower()
    text_in_cuvinte = text.split(' ')
    # text_in_cuvinte = word_tokenize(text)
    for word in text_in_cuvinte:
        if (len(word) < 4):
            text_in_cuvinte.remove(word)
    return text_in_cuvinte




def vectorize(text):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2))
    print(vectorizer.get_feature_names())
    #  X = vectorizer.fit_transform(text)

    # df_bow_sklearn = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    # df_bow_sklearn.head()
    print(vectorizer)
    return vectorizer


# cuvintele rezultate din functia de preprocesare:
exemple_italian = train_data_df[train_data_df['language'] == 'italiano']

text_italian = exemple_italian['text'].iloc[0]
data = train_data_df['text'].apply(lambda text: proceseaza(text))
vectorize(data)
nr_test = 13800
nr_ramase = len(data) - nr_test
nr_valid = int(15 / 100 * nr_ramase)
nr_train = nr_ramase - nr_valid
indici = np.arange(0, len(train_data_df))
np.random.shuffle(indici)
train_data = data[indici[:nr_train]]
train_labels = labels[indici[:nr_train]]

valid_data = data[indici[nr_train: nr_train + nr_valid]]
valid_labels = labels[indici[nr_train: nr_train + nr_valid]]

test_data = data[indici[nr_train + nr_valid:]]
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


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

cuvinte_caracteristice = count_most_common(30, train_data)
word2id, id2word = build_id_word_dicts(cuvinte_caracteristice)

X_train = featurize_multi(train_data, id2word)
X_valid = featurize_multi(valid_data, id2word)
X_test = featurize_multi(test_data, id2word)

# model = BernoulliNB(alpha=2)
solutia = 0
max = 0.00;

for i in range(1, 2):
    model = svm.LinearSVC(C=0.1, loss='hinge', dual=True, tol=1e-5, multi_class='crammer_singer', fit_intercept=True,
                          intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=100)
    model.fit(X_train, train_labels)

    vpreds = model.predict(X_valid)
    tpreds = model.predict(X_test)

    a = accuracy_score(valid_labels, vpreds)

    b = accuracy_score(test_labels, tpreds)
    # print('Solutii:')
    # print(a)
    # print(b)
    if (a > max):
        max = a
        solutia = i
print('solutia maxima este', max, 'solutia', solutia)

# with open('submission.csv', 'w') as f:
#   f.write('id,label\n')
#   for id, label in enumerate(model.predict(X_test)):
#           if id == 13860:
#               break
#           else:
#               f.write(f'{id + 1},{id2label[label]}\n')

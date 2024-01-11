# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:35:59 2023

@author: David López
"""

import csv
import pandas as pd
import numpy as np

spam_or_ham = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
spam_or_ham.columns = ["label", "text"]
spam_or_ham.head()

spam_or_ham["label"].value_counts()

import string
punctuation = set(string.punctuation)
def tokenize(sentence):
    tokens = []
    for token in sentence.split():
        new_token = []
        for character in token:
            if character not in punctuation:
                new_token.append(character.lower())
        if new_token:
            tokens.append("".join(new_token))
    return tokens

spam_or_ham.head()["text"].apply(tokenize)

from sklearn.feature_extraction.text import CountVectorizer
demo_vectorizer = CountVectorizer(
    tokenizer = tokenize,
    binary = True
)

from sklearn.model_selection import train_test_split
train_text, test_text, train_labels, test_labels = train_test_split(spam_or_ham["text"], spam_or_ham["label"], stratify=spam_or_ham["label"])
print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")

real_vectorizer = CountVectorizer(tokenizer = tokenize, binary=True)
train_X = real_vectorizer.fit_transform(train_text)
test_X = real_vectorizer.transform(test_text)

from sklearn.svm import LinearSVC
classifier = LinearSVC()
classifier.fit(train_X, train_labels)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)

from sklearn.metrics import accuracy_score
predicciones = classifier.predict(test_X)
accuracy = accuracy_score(test_labels, predicciones)
print(f"Precisión: {accuracy:.4%}")

frases = [
  'Are you looking to redesign your website with new modern look and feel?',
  'Please send me a confirmation of complete and permanent erasure of the personal data',
  'You have been selected to win a FREE suscription to our service',
  'We’re contacting you because the webhook endpoint associated with your account in test mode has been failing',
  'Confirma tu cuenta de Facebook en el siguiente link',
  'You have been selected to participate in a free service'
]

frases_X = real_vectorizer.transform(frases)
predicciones = classifier.predict(frases_X)

for text, label in zip(frases, predicciones):
  print(f"{label:5} - {text}")
  


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import export_graphviz
import sys
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support
import nltk
nltk.download('wordnet')
#This will break if you do --mode or --input and do not supply an input, so dont do that
def get_args():
    args = sys.argv
    for i in range(len(args)):
        if args[i] == "--mode":
            mode = args[i+1]
        elif args[i] == "--input":
            _input = args[i+1]
    return mode, _input

def stem_text(text):
    #stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    return [" ".join([lemma.lemmatize(word) for word in sentence.split(" ")]) for sentence in text]

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
def main():
    mode, _input = get_args()
    input_data = pd.read_csv(_input)
    formatted_text = stem_text(input_data.text)
    print(formatted_text)

    tfidf = TfidfVectorizer()
    data = tfidf.fit_transform(formatted_text)

    train_feature, test_feature, train_class, test_class = train_test_split(data, input_data.label, stratify=input_data.label, random_state=0)

    svm = LinearSVC().fit(train_feature, train_class)
    print(svm.score(test_feature, test_class))

    scores = cross_val_score(svm, data, input_data.label, cv=5)
    pred = svm.predict(test_feature)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {:.2f}".format(scores.mean()))
    print(pd.crosstab(test_class, pred, rownames=['True'], colnames=['Predicted'], margins=True))

    #TODO: import trained model
    #TODO: Prediction and other one

main()

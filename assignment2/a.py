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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math

#This will break if you do --mode or --input and do not supply an input, so dont do that
def get_args():
    args = sys.argv
    for i in range(len(args)):
        if args[i] == "--mode":
            mode = args[i+1]
        elif args[i] == "--input":
            _input = args[i+1]
    return mode, _input

def remove_stopwords(tokenized_doc):
    stop = stopwords.words('english')
    filtered = []
    for token in tokenized_doc:
        if token not in stop:
            filtered.append(token)
    return filtered

def tokenize(string):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(string)
    return tokens

def get_stems(doc):
    stemmed = []
    stemmer = PorterStemmer()
    for tok in doc:
        stem = stemmer.stem(tok)
        if stem not in stemmed:
            stemmed.append(stem)
    return stemmed

def count_words(paragraph):
    words = {}
    for word in paragraph:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words

def get_features(example):
    example_words = count_words(get_stems(remove_stopwords(set(tokenize(example.lower())))))
    return example_words

#word_vector should be a vector with all words as keys and all values as the count as 0
def get_document_vector(vector, word_vector):
    return [vector[word] if word in vector else 0 for word in word_vector]
    # doc_vec = []
    # for word in word_vector:
    #     if word in vector:
    #         doc_vec.append(vector[word])
    #     else:
    #         doc_vec.append(0)
    # return word_vector

from nltk import NaiveBayesClassifier
import nltk
from sklearn.tree import DecisionTreeClassifier
def main():
    mode, _input = get_args()
    input_data = pd.read_csv(_input)

    bag_of_words = {}
    for sentence in input_data.text:
        features = get_features(sentence)
        for word in features:
            bag_of_words[word] = 0
    document_vectors = [get_document_vector(get_features(sentence), bag_of_words.keys()) for sentence in input_data.text]
    
    # input_set = list(zip(document_vectors, input_data.labels))

    train_feature, test_feature, train_class, test_class = train_test_split(document_vectors, input_data.label, stratify=input_data.label, random_state=0)

    nb = GaussianNB().fit(train_feature, train_class)
    print(nb.score(test_feature, test_class))

    tree = DecisionTreeClassifier().fit(train_feature, train_class)
    print(tree.score(test_feature, test_class))

    svm = LinearSVC().fit(train_feature, train_class)
    print(svm.score(test_feature, test_class))

    knn = KNeighborsClassifier().fit(train_feature, train_class)
    print(knn.score(test_feature, test_class))

    # nb = nltk.NaiveBayesClassifier.train(train_set)
    # print(nltk.classify.accuracy(nb, test_set))
    # nb.show_most_informative_features()

    #sklearn needs a vector containing counts of all words in the document, not sentence
    # tree = DecisionTreeClassifier().fit([x[0] for x in train_set], [x[1] for x in train_set])
    # print(tree.score([x[0] for x in test_set], [x[1] for x in test_set]))
    # svm = LinearSVC(random_state=0).fit(train_feature, train_class)
    # print(svm.score(test_feature, test_class))

main()








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
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import export_graphviz
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#This will break if you do --mode or --input and do not supply an input, so dont do that
def get_args():
    args = sys.argv
    for i in range(len(args)):
        if args[i] == "--mode":
            mode = args[i+1]
        elif args[i] == "--input":
            _input = args[i+1]
    return mode, _input

def remove_stopwords(tokenized_doc):
    stop = stopwords.words('english')
    filtered = []
    for token in tokenized_doc:
        if token not in stop:
            filtered.append(token)
    return filtered

def tokenize(string):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(string)
    return tokens

def get_stems(doc):
    stemmed = []
    stemmer = PorterStemmer()
    for tok in doc:
        stem = stemmer.stem(tok)
        if stem not in stemmed:
            stemmed.append(stem)
    return stemmed

def count_words(paragraph):
    words = {}
    for word in paragraph:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words

def get_features(example):
    example_words = count_words(get_stems(remove_stopwords(set(tokenize(example.lower())))))
    return example_words

from nltk import NaiveBayesClassifier
from nltk import classify
import sklearn.feature_extraction.text
from sklearn.tree import DecisionTreeClassifier
def main():
    mode, _input = get_args()
    input_data = pd.read_csv(_input)



    # nb = nltk.NaiveBayesClassifier.train(train_set)
    # print(nltk.classify.accuracy(nb, test_set))

    svm = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
    print(nltk.classify.accuracy(svm, test_set))

    # mnb = nltk.classify.SklearnClassifier(MultinomialNB()).train(train_set)
    # print(nltk.classify.accuracy(mnb, test_set))

    #sklearn needs
    # svm = LinearSVC(random_state=0).fit(train_feature, train_class)
    # print(svm.score(test_feature, test_class))

main()

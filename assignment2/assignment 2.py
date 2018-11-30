from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import nltk
import sys
import numpy as np
import pandas as pd

# This will break if you do --mode or --input and do not supply an input, so dont do that
def get_args():
    args = sys.argv
    mode = ""
    _input = ""
    for i in range(len(args)):
        if args[i] == "--mode":
            mode = args[i+1]
        elif args[i] == "--input":
            _input = args[i+1]
    return mode, _input

# Applys stemming and lemmatization on a list of strings
def stem_text(text):
    # Stemming actually makes it worse, perhaps because of the importance of the tense of some words
    #stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    return [" ".join([lemma.lemmatize(word) for word in sentence.split(" ")]) for sentence in text]

# Applys stemming and lemmatization on a single string
def stem_line(line):
    lemma = WordNetLemmatizer()
    return " ".join([lemma.lemmatize(word) for word in line.split(" ")])

def main():
    # Get the arguments from the command lines
    mode, _input = get_args()

    if mode == "train":
        # Read csv file and stem each line
        input_data = pd.read_csv(_input)
        formatted_text = stem_text(input_data.text)
        
        # Create tfidf vector for each row in the csv file
        tfidf = TfidfVectorizer()
        data = tfidf.fit_transform(formatted_text)

        # Split the training and testing data with 75/25 split
        train_feature, test_feature, train_class, test_class = train_test_split(data, input_data.label, stratify=input_data.label, random_state=0, test_size=0.25)

        # SVM had the best accuracy among all of the classifiers I tested (svm, mnb, nb, knn, tree, etc.)
        # Fit the training data and make predictions for the testing data
        svm = LinearSVC().fit(train_feature, train_class)
        pred = svm.predict(test_feature)

        # Printing out the classification report and confusion matrix
        print("Classification report:")
        print(classification_report(test_class, pred))
        print("Confusion Matrix")
        print(pd.crosstab(test_class, pred, rownames=['True'], colnames=['Predicted'], margins=True))

        # Export svm and tfidf data for prediction
        joblib.dump(svm, "svm.joblib")
        joblib.dump(tfidf, "tfidf.joblib")

    elif mode == "cross_val":
        # Read csv file and stem each line
        input_data = pd.read_csv(_input)
        formatted_text = stem_text(input_data.text)

        # Create tfidf vector for each row in the csv file
        tfidf = TfidfVectorizer()
        data = tfidf.fit_transform(formatted_text)

        # Split the training and testing data
        train_feature, test_feature, train_class, test_class = train_test_split(data, input_data.label, stratify=input_data.label, random_state=0)

        # SVM had the best accuracy among all of the classifiers I tested (svm, mnb, nb, knn, tree, etc.)
        # Fit the training data and create the cross-validation score list and find the mean
        svm = LinearSVC().fit(train_feature, train_class)
        scores = cross_val_score(svm, data, input_data.label, cv=10)
        print("Cross-validation scores: {}".format(scores))
        print("Average cross-validation score: {:.2f}".format(scores.mean()))

        # Export svm and tfidf data for prediction
        joblib.dump(svm, "svm.joblib")
        joblib.dump(tfidf, "tfidf.joblib")

    elif mode == "predict" :
        # Import the joblibs for svm and tfidf, this is assuming that this file will be run on either train or 
        # cross_val mode first so the file exists.
        svm = joblib.load("svm.joblib")
        tfidf = joblib.load("tfidf.joblib")

        # Input and stem the input string to be predicted
        input_string = _input
        formatted_text = stem_line(input_string)

        # Transform the input string to match the size of the trained vocabulary
        data = tfidf.transform([formatted_text])

        # Prints the prediction
        print(svm.predict(data)[0])

# Main is my entry point of the code, so I run the function here
main()
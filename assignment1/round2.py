import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math


def read_file(filename):
    with open(filename) as f:
        doc = f.read()
    return doc


def tokenize(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    return tokens


def remove_stopwords(tokenized_doc):
    stop = stopwords.words('english')
    filtered = []
    for token in tokenized_doc:
        if token not in stop:
            filtered.append(token)
    return filtered


def get_stems(doc):
    stemmed = []
    stemmer = PorterStemmer()
    for tok in doc:
        stem = stemmer.stem(tok)
        if stem not in stemmed:
            stemmed.append(stem)
    return stemmed


def get_stem(token):
    stemmer = PorterStemmer()
    return stemmer.stem(token)


def get_paragraphs(doc):
    paragraphs = [x for x in doc.splitlines() if x]
    return paragraphs


def count_words(paragraph):
    words = {}
    for word in paragraph:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words


def getidf(term):
    global paragraph_vectors
    N = len(paragraph_vectors)
    dft = 0
    for p in paragraph_vectors:
        if term in p:
            dft += p[term]

    if dft == 0:
        return -1

    return math.log10(N / dft)

def gettf(term, paragraph_vector):
    if paragraph_vector[term] == 0:
        return 0
    return math.log10(paragraph_vector[term])

def gettfidf(term, paragraph_vector):
    return (1 + gettf(term, paragraph_vector)) * getidf(term)

def getqvec(qstring):
    term_vect = get_stems(remove_stopwords(tokenize(qstring.lower())))
    return normalize_dict({term: getidf(term) for term in term_vect})

def cos_sim(query, paragraph, i):
    cosim = 0
    global counted_paragraphs
    for term in query:
        if term in paragraph:
            cosim += query[term] * paragraph[term] # * gettf(term, counted_paragraphs)[i]
    #cosim /= sum([a**2 for a in paragraph.values()])
    return cosim

def get_paragraph_vector(paragraph, bag_of_words):
    p_vec = {w: 0 for w in bag_of_words}
    for word in paragraph:
        p_vec[word] = paragraph[word]
    return p_vec


counted_paragraphs = []
paragraphs = []
paragraph_vectors = []

def main():
    global paragraphs, counted_paragraphs, paragraph_vectors
    # Read the document
    doc = read_file("debate.txt")
    # Split each paragraph
    bag_of_words = [a for a in get_stems(remove_stopwords(tokenize(str(doc).lower())))]    
    paragraphs = get_paragraphs(doc)
    # Lowercase all paragraphs
    paragraphs = [p.lower() for p in paragraphs]
    # Tokenize each paragraph
    tokens = [tokenize(p) for p in paragraphs]
    # Remove stopwords of each paragraph
    tokens = [remove_stopwords(p) for p in tokens]
    # Stem each paragraph
    tokens = [get_stems(p) for p in tokens]
    # count of each word in paragraph i
    counted_paragraphs = [count_words(p) for p in tokens]
    # each element is a vector containing all words from the document and their count in paragraph i
    paragraph_vectors = [get_paragraph_vector(p, bag_of_words) for p in counted_paragraphs]

    for term in tokens[0]:
        tf = gettf(term, paragraph_vectors[0])
        if tf > 0:
            print(term, tf, getidf(term))


main()

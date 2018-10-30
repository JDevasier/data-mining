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


def normalize(l):
    s = sum(l)
    if s == 0:
        return -1
    return [float(float(li) / float(s)) for li in l]


def normalize_dict(d):
    s = sum(d.values())
    if s == 0:
        return -1
    return {li: float(float(d[li]) / float(s)) for li in d}


def gettf(term, counted_paragraphs):
    tf_vector = []
    for p in counted_paragraphs:
        if term in p:
            tf_vector.append(math.log10(1 + p[term]))
        else:
            tf_vector.append(0)
    return tf_vector


def getidf(term):
    global counted_paragraphs
    N = len(counted_paragraphs)
    dft = 0
    for p in counted_paragraphs:
        if term in p:
            dft += 1

    if dft == 0:
        return -1

    return math.log10(N / dft)


def getqvec(qstring):
    term_vector = get_stems(remove_stopwords(tokenize(qstring.lower())))
    return normalize_dict({term: getidf(term) for term in term_vector})


def query(qstring):
    global counted_paragraphs
    qvec = getqvec(qstring)
    q = {term: gettfidf(term, counted_paragraphs) for term in qvec}
    d = counted_paragraphs

    best_p = ("", -1)
    for p in d:
        # print(p)
        print(cos_sim(qvec, p))

    return best_p


def cos_sim(query, paragraph):
    cosim = 0
    print(query, paragraph)
    for term in query:
        if term in paragraph:
            cosim += query[term]
    return cosim


counted_paragraphs = []


def main():
    # Read the document
    doc = read_file("debate.txt")
    # Get bag of words
    words = get_stems(remove_stopwords(tokenize(str(doc).lower())))
    # Split each paragraph
    paragraphs = get_paragraphs(doc)
    # Lowercase all paragraphs
    paragraphs = [p.lower() for p in paragraphs]
    # Tokenize each paragraph
    tokens = [tokenize(p) for p in paragraphs]
    # Remove stopwords of each paragraph
    tokens = [remove_stopwords(p) for p in tokens]
    # Stem each paragraph
    tokens = [get_stems(p) for p in tokens]

    paragraph_vectors = [{word: p.count(word)
                          for word in words} for p in tokens]

    # Count number of occurences for each word in each paragraph
    global counted_paragraphs
    counted_paragraphs = [count_words(p) for p in tokens]

    print(getqvec("vector entropy"))


main()

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


def normalize_dict(d):
    s = math.sqrt(sum([a**2 for a in d.values()]))
    if s == 0:
        return -1
    return {li: float(float(d[li]) / float(s)) for li in d}


# paragraph is a dictionary with each term and its count in a paragraph
def gettfidf(term, paragraph):
    return getidf(term) * gettf(term, paragraph)


def gettf(term, paragraph):
    if term in paragraph:
        return 1 + math.log10(paragraph[term])
    return 0


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
    term_vect = get_stems(remove_stopwords(tokenize(qstring.lower())))
    return normalize_dict({term: getidf(term) for term in term_vect})


def query(qstring):
    global weighted_paragraphs, paragraphs
    qvec = getqvec(qstring)
    d = weighted_paragraphs

    best_p = ("", 0)
    for i in range(len(d)):
        score = cos_sim(qvec, weighted_paragraphs[i])
        if score > best_p[1]:
            best_p = (paragraphs[i], score)

    return best_p


def cos_sim(query, paragraph):
    cosim = 0
    for term in query:
        if term in paragraph:
            cosim += query[term] * paragraph[term]
    return cosim


counted_paragraphs = []
paragraphs = []
weighted_paragraphs = []

def main():
    global counted_paragraphs, weighted_paragraphs, paragraphs
    # Read the document
    doc = read_file("debate.txt")
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

    # My idea here is rather than creating large 1200 element, mostly sparse, vectors, I will create
    # one vector containing the count of only the words in the paragraph. During the dot product step
    # for cos sim, I will add to the dot product value only if the word is in the paragraph. 
    # This works because if either value for the dot product operation is 0, it will not add anything the current value

    # Count number of occurences for each word in each paragraph
    counted_paragraphs = [count_words(p) for p in tokens]
    # Weight each counted paragraph by their tf-idf values
    weighted_paragraphs = [normalize_dict({term : gettfidf(term, p) for term in p}) for p in counted_paragraphs]

    # Run test code here:
    #print("%s%.4f" % query("clinton first amendment kavanagh"))
    print("%s%.4f" % query("The alternative, as cruz has proposed, is to deport 11 million people from this country"))



main()

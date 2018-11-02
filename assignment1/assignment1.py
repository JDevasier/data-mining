import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math

# Reads a file given the path 
def read_file(filename):
    with open(filename) as f:
        doc = f.read()
    return doc

# Tokenizes a string and returns the tokens
def tokenize(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    return tokens

# Removes all stopwords from a list of words
def remove_stopwords(tokenized_doc):
    stop = stopwords.words('english')
    filtered = []
    for token in tokenized_doc:
        if token not in stop:
            filtered.append(token)
    return filtered

# Stems all words in a list and returns the stemmed list
def get_stems(doc):
    stemmed = []
    stemmer = PorterStemmer()
    for tok in doc:
        stem = stemmer.stem(tok)
        if stem not in stemmed:
            stemmed.append(stem)
    return stemmed

# Stems a single word
def get_stem(token):
    stemmer = PorterStemmer()
    return stemmer.stem(token)

# Returns all paragraphs from a *file*
def get_paragraphs(doc):
    paragraphs = [x for x in doc.splitlines() if x]
    return paragraphs

# Returns a dictionary formatted like "term": N, where N is the number of times that word appears in the paragraph
def count_words(paragraph):
    words = {}
    for word in paragraph:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words

# Normalizes a dictionary
def normalize_dict(d):
    s = math.sqrt(sum([a**2 for a in d.values()]))
    if s == 0:
        return -1
    return {li: float(float(d[li]) / float(s)) for li in d}

# Returns the tf-idf vector of a paragraph
# paragraph is a dictionary with each term and its count in a paragraph
def gettfidf(term, paragraph):
    return getidf(term) * gettf(term, paragraph)

# Returns the tf value of a term in a paragraph
def gettf(term, paragraph):
    if term in paragraph:
        return 1 + math.log10(paragraph[term])
    return 0

# Returns the idf value of a term over all paragraphs in the document
def getidf(term):
    global counted_paragraphs
    N = len(counted_paragraphs)
    dft = 0

    for p in counted_paragraphs:
        if term in p:
            dft += 1

    if dft == 0:
        return 1

    return math.log10(N / dft)

# Returns the normalized tf-idf vector of the query string
def getqvec(qstring):
    term_vect = get_stems(remove_stopwords(tokenize(qstring.lower())))
    term_vect_pg = count_words(term_vect)
    return normalize_dict({term: gettfidf(term, term_vect_pg) for term in term_vect})

# Merges 2 dictionaries
def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

# Queries all paragraphs in the document to find the best paragraph
def query(qstring):
    global counted_paragraphs, paragraphs
    qvec = getqvec(qstring)

    # Create a list of tf values for each term in each paragraph
    # Although the merging is not necessary because of how I implemented the cos sim,
    # I still apply it for the sake of understanding what is happening
    d = [normalize_dict({term: gettf(term, p) for term in merge_dicts(p, qvec)})
         for p in counted_paragraphs]

    best_p = ("", 0)
    for i in range(len(d)):
        score = cos_sim(qvec, d[i])
        if score > best_p[1]:
            best_p = (paragraphs[i], score)

    return best_p

# Computes the cosine similarity of a query and a paragraph (or 2 paragraphs)
def cos_sim(query, paragraph):
    cosim = 0
    # Rather than storing very sparse matrices of terms and their frequencies,
    # I only store the non-zero values because they are the only ones that
    # change the cos sim value. 
    for term in query:
        if term in paragraph:
            cosim += query[term] * paragraph[term]
    return cosim

# Global list of dictionaries for each term and its word count in a paragraph
counted_paragraphs = []
# Global list of the paragraphs
paragraphs = []

# Entrance point of the program
def main():
    global counted_paragraphs, paragraphs
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

    # Count number of occurences for each word in each paragraph (unweighted tf vector of each paragraph)
    counted_paragraphs = [count_words(p) for p in tokens]

    # Run test code here:

main()
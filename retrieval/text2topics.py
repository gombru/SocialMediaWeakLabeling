# Gets an input text, an LDA model and its number of topics and outputs the topic distribution
# output is np array of dim num_topics

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import string
import numpy as np


def text2topics(text, ldamodel, num_topics):

    # Create English stop words list
    en_stop = get_stop_words('en')
    whitelist = string.letters + string.digits + ' '

    tokenizer = RegexpTokenizer(r'\w+')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    filtered_text = ''

    # Replace hashtags with spaces
    text = text.replace('#',' ')

    # Keep only letters and numbers
    for char in text:
        if char in whitelist:
            filtered_text += char

    filtered_text = filtered_text.lower()

    tokens = tokenizer.tokenize(filtered_text)

    # Remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # Handle stemmer error
    while "aed" in stopped_tokens:
        stopped_tokens.remove("aed")
        print "aed error"

    text = [p_stemmer.stem(i) for i in stopped_tokens]
    bow = ldamodel.id2word.doc2bow(text)
    r = ldamodel[bow]

    # Add zeros to topics without score
    topic_probs = np.zeros(num_topics)
    for t in range(0, num_topics):
        for topic in r:
            if topic[0] == t:
                topic_probs[t] = topic[1]
                break

    return topic_probs

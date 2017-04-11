from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import string
import numpy as np


def text2topics(text, ldamodel, num_topics):

    # create English stop words list
    en_stop = get_stop_words('en')
    whitelist = string.letters + string.digits + ' '

    # print "Loading LDA model ..."
    # ldamodel = models.ldamodel.LdaModel.load(model_path)
    tokenizer = RegexpTokenizer(r'\w+')

    # print "Computing topics for query ..."

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    filtered_text = ''

    # --Replace hashtags with spaces
    text = text.replace('#',' ')

    # -- Keep only letters and numbers
    for char in text:
        if char in whitelist:
            filtered_text += char

    filtered_text = filtered_text.lower()

    tokens = tokenizer.tokenize(filtered_text)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem token

    #Check this error
    while "aed" in stopped_tokens:
        stopped_tokens.remove("aed")
        print "aed error"

    text = [p_stemmer.stem(i) for i in stopped_tokens]
    bow = ldamodel.id2word.doc2bow(text)
    r = ldamodel[bow]

    topic_probs = np.zeros(num_topics)
    for t in range(0, num_topics):
        for topic in r:
            if topic[0] == t:
                topic_probs[t] = topic[1]
                break

    return topic_probs

# Gets an input text, an LDA model and its number of topics and outputs the topic distribution
# output is np array of dim num_topics

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import gensim


def LDA(text, ldamodel, num_topics):

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
        print("aed error")

    text = [p_stemmer.stem(i) for i in stopped_tokens]


    # bow = ldamodel.id2word.doc2bow(text)
    # r = ldamodel[bow]    # Warning, this uses a threshold of 0.01 on tropic probs, and usually returns only 1 max 2...

    if len(text) > 1:
        print("Warning: only using first word")
    if len(text) == 0:
        print("Not valid word")
        return np.zeros(num_topics)
    r = ldamodel.get_term_topics(text[0].__str__(),0) # #This 0 is changed to 1e-8 inside
    # Add zeros to topics without score
    topic_probs = np.zeros(num_topics)
    for t in range(0, num_topics):
        for topic in r:
            if topic[0] == t:
                topic_probs[t] = topic[1]
                break

    topic_probs = topic_probs - min(topic_probs)
    topic_probs = topic_probs / sum(topic_probs)

    return topic_probs



def doc2vec(text, model, num_topics):

    filtered_text = ''
    # Replace hashtags with spaces
    text = text.replace('#', ' ')

    en_stop = get_stop_words('en')
    whitelist = string.letters + string.digits + ' '
    # Keep only letters and numbers
    for char in text:
        if char in whitelist:
            filtered_text += char

    filtered_text = filtered_text.lower()
    # Gensim simple_preproces instead tokenizer
    tokens = gensim.utils.simple_preprocess(filtered_text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]

    embedding = model.infer_vector(tokens_filtered)
    embedding = embedding - min(embedding)
    embedding = embedding / sum(embedding)

    return embedding


def word2vec_mean(text, word_weights, model, num_topics):
    filtered_text = ''
    # Replace hashtags with spaces
    text = text.replace('#', ' ')

    en_stop = get_stop_words('en')
    whitelist = string.letters + string.digits + ' '
    # Keep only letters and numbers
    for char in text:
        if char in whitelist:
            filtered_text += char

    filtered_text = filtered_text.lower()
    # Gensim simple_preproces instead tokenizer
    tokens = gensim.utils.simple_preprocess(filtered_text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]
    word_weights = [float(x) for x in word_weights.split()]
    print word_weights

    embedding = np.zeros(num_topics)
    c = 0
    for e,tok in enumerate(tokens_filtered):
        try:

            embedding += model[tok] * word_weights[e]
            c += 1
        except:
            # print "Word not in model: " + tok
            continue

    if sum(word_weights) > 0:
        embedding /= sum(word_weights)

    embedding = embedding - min(embedding)
    embedding = embedding / max(embedding)

    #embedding = embedding / sum(embedding)

    return embedding




def word2vec_tfidf(text, model, num_topics, tfidf_model, tfidf_dictionary):

    whitelist = string.ascii_letters + string.digits + ' '
    # Keep only letters and numbers
    filtered_text = ''
    text = text.replace('#', ' ')
    for char in text:
        if char in whitelist:
            filtered_text += char
    filtered_text = filtered_text.lower()

    # Gensim simple_preproces instead tokenizer
    en_stop = get_stop_words('en')
    tokens = gensim.utils.simple_preprocess(filtered_text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]

    embedding = np.zeros(num_topics)
    c = 0

    vec = tfidf_dictionary.doc2bow(tokens_filtered)
    vec_tfidf = tfidf_model[vec]
    for tok in vec_tfidf:
        word_embedding = model[tfidf_dictionary[tok[0]]]
        embedding += word_embedding * tok[1]

    # if c > 0:
    #     embedding /= c

    embedding = embedding - min(embedding)
    if max(embedding) > 0:
        embedding = embedding / max(embedding)

    #embedding = embedding / sum(embedding)

    return embedding



def glove(text, word_weights, model, num_topics):

    filtered_text = ''
    # Replace hashtags with spaces
    text = text.replace('#', ' ')

    en_stop = get_stop_words('en')
    whitelist = string.ascii_letters + string.digits + ' '
    # Keep only letters and numbers
    for char in text:
        if char in whitelist:
            filtered_text += char

    filtered_text = filtered_text.lower()
    # Gensim simple_preproces instead tokenizer
    tokens = gensim.utils.simple_preprocess(filtered_text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    embedding = np.zeros(num_topics)
    word_weights = [float(x) for x in word_weights.split()]

    c = 0
    for e,tok in enumerate(stopped_tokens):
        try:
            embedding += model.word_vectors[model.dictionary[tok]] * word_weights[e]
            # print model.word_vectors[model.dictionary[tok]]
            c += 1
        except:
            # print "Word not in model: " + tok
            continue
    if sum(word_weights) > 0:
        embedding /= sum(word_weights)


    embedding = embedding - min(embedding)
    embedding = embedding / sum(embedding)

    return embedding


def glove_tfidf(text,  model, num_topics):

    filtered_text = ''
    # Replace hashtags with spaces
    text = text.replace('#', ' ')

    en_stop = get_stop_words('en')
    whitelist = string.ascii_letters + string.digits + ' '
    # Keep only letters and numbers
    for char in text:
        if char in whitelist:
            filtered_text += char

    filtered_text = filtered_text.lower()
    # Gensim simple_preproces instead tokenizer
    tokens = gensim.utils.simple_preprocess(filtered_text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    embedding = model.transform_paragraph(stopped_tokens, 50, True)
    embedding /= len(stopped_tokens)
    embedding = embedding - min(embedding)
    embedding = embedding / sum(embedding)

    return embedding
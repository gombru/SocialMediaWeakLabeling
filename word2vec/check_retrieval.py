from stop_words import get_stop_words
import gensim
import numpy as np
import operator


# Load data and model
model_path = '../../../datasets/WebVision/models/word2vec/word2vec_model_webvision.model'
model = gensim.models.Word2Vec.load(model_path)
en_stop = get_stop_words('en')

captions = ['ocean','boat sailing in the sea','sea','beach','man and woman with a kid','having a pizza for dinner','king','queen','royalty','pizza']
database = {}
database_words = {}


def get_embedding(text):
    tokens = gensim.utils.simple_preprocess(text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    embedding = np.zeros(400)
    c = 0
    for tok in stopped_tokens:
        try:
            embedding += model[tok]
            c += 1
        except:
            # print "Word not in model: " + tok
            continue
    if c > 0:
        embedding /= c

    embedding = embedding - min(embedding)
    embedding = embedding / sum(embedding)
    return embedding

for n,caption in enumerate(captions):
    database[n] = get_embedding(caption)

queries = ['people','nature','food''king','queen','sea','boat sayling in the sea','man and woman with a kid','with the family and the kid my sister','eating some meat at the restaurant','family','food','pizza','water','beach','boat']

for q in queries:
    distances1 = {}

    print q + "------------> \n"
    q_embedding = get_embedding(q)

    for n, caption in enumerate(database):
        distances1[n] = np.dot(database[n], q_embedding)

    # # Sort dictionary
    distances1 = sorted(distances1.items(), key=operator.itemgetter(1), reverse=True)

    # Get elements with min distances
    for idx, id in enumerate(distances1):
        print captions[id[0]]

    print '__________________ \n'




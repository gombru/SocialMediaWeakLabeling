# Retrieves nearest images given a text query and saves them in an given folder

import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models
import gensim
import glove


f = open('food_word_embeddings.txt','w')

model_name = 'word2vec_model_InstaCities1M.model'
num_topics = 400 # Num LDA model topics

#-----------> if tfidf
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)


model_path = '../../../datasets/SocialMedia/models/word2vec/' + model_name
embedding = 'word2vec_mean' #'word2vec_mean' 'doc2vec' 'LDA' 'word2vec_tfidf' 'glove' 'glove_tfidf'


# Load LDA model
print("Loading " +embedding+ " model ...")
if embedding == 'LDA': model = models.ldamodel.LdaModel.load(model_path)
elif embedding == 'word2vec_mean' or embedding == 'word2vec_tfidf': model = models.Word2Vec.load(model_path)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_path)
elif embedding == 'glove' or embedding == 'glove_tfidf': model = glove.Glove.load(model_path)


def get_embedding_complex(text, word_weights):

    words = text.split(' ')
    topics = np.zeros(num_topics)

    if embedding == 'LDA':
        for w in words:
            w_topics = text2topics.LDA(w, model, num_topics)
            topics = topics + w_topics
        topics = topics / len(words)

    elif embedding == 'word2vec_mean':
        topics = text2topics.word2vec_mean(text, word_weights, model, num_topics)


    elif embedding == 'word2vec_tfidf':
        topics = text2topics.word2vec_tfidf(text, model, num_topics, tfidf_model, tfidf_dictionary)

    elif embedding == 'doc2vec':
        topics = text2topics.doc2vec(text, model, num_topics)


    elif embedding == 'glove':
        topics = text2topics.glove(text, word_weights, model, num_topics)

    elif embedding == 'glove_tfidf':
        topics = text2topics.glove_tfidf(text, model, num_topics)

    return topics


# Do default queryes
q = []

search_words = ['food','eat','drink','dinner','breakfast','lunch','wine','beer','cocktail','restaurant','bar','cook','meat','fish','hamburguer','sushi','salad','fruit','vegetables']
for w in search_words:
    q.append(w)

# # Simple
# q.append('car')
# q.append('skyline')
# q.append('bike')
#
# q.append('sunrise')
# q.append('snow')
# q.append('rain')
#
# q.append('icecream')
# q.append('cake')
# q.append('pizza')
#
# q.append('woman')
# q.append('man')
# q.append('kid')
#
# # Complex
# q.append('yellow car')
# q.append('skyline night')
# q.append('bike park')
#
# q.append('sunrise beach')
# q.append('snow ski')
# q.append('rain umbrella')
#
# q.append('icecream beach')
# q.append('chocolate cake')
# q.append('pizza wine')
#
# q.append('woman bag')
# q.append('man boat')
# q.append('kid dog')



for cur_q in q:
    cur_w = '1 1'

    if len(cur_q.split(' ')) == 1:

        if embedding == 'LDA': topics = text2topics.LDA(cur_q,  model, num_topics)
        elif embedding == 'word2vec_mean': topics = text2topics.word2vec_mean(cur_q, cur_w, model, num_topics)
        elif embedding == 'doc2vec': topics = text2topics.doc2vec(cur_q, model, num_topics)
        elif embedding == 'word2vec_tfidf': topics = text2topics.word2vec_tfidf(cur_q, model, num_topics, tfidf_model, tfidf_dictionary)
        elif embedding == 'glove': topics = text2topics.glove(cur_q, cur_w, model, num_topics)
        elif embedding == 'glove_tfidf': topics = text2topics.glove_tfidf(cur_q, model, num_topics)

    else:
        topics = get_embedding_complex(cur_q, cur_w)

    # Normalize by max to compare with net
    topics = topics - min(topics)
    topics = topics / max(topics)

    f.write('word_images/' + cur_q.replace(' ','_'))
    for t in topics:
        f.write(',' + str(t))
    f.write('\n')










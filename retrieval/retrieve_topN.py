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

data = 'WebVision_Inception_frozen_word2vec_tfidfweighted_divbymax_iter_460000'
model_name = 'word2vec_model_webvision.model'
num_topics = 400 # Num LDA model topics
num_results = 4 # Num retrival results we want to take into accountnt

#-----------> if tfidf
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/WebVision/regression_output/' + data +'/test.txt'
model_path = '../../../datasets/WebVision/models/word2vec/' + model_name
embedding = 'word2vec_mean' #'word2vec_mean' 'doc2vec' 'LDA' 'word2vec_tfidf' 'glove' 'glove_tfidf'
test_dataset = 'webvision' #'instacities1m' #webvision


# Load LDA model
print("Loading " +embedding+ " model ...")
if embedding == 'LDA': model = models.ldamodel.LdaModel.load(model_path)
elif embedding == 'word2vec_mean' or embedding == 'word2vec_tfidf': model = models.Word2Vec.load(model_path)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_path)
elif embedding == 'glove' or embedding == 'glove_tfidf': model = glove.Glove.load(model_path)



# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
for id in database:
    database[id] = database[id] / sum(database[id])


def get_results(database, topics, num_results, results_path):
    # Create empty dict for distances
    distances = {}

    #Compute distances
    for id in database:
        distances[id] = np.dot(database[id],topics)

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)

    # Get elements with min distances
    for idx,id in enumerate(distances):
        # Copy image results
        if test_dataset == 'webvision':
            copyfile('../../../datasets/WebVision/test_images_256/' + id[0] , results_path + id[0].replace('/', '_'))
        else:
            copyfile('../../../datasets/SocialMedia/img_resized_1M/cities_instagram/' + id[0] + '.jpg', results_path + id[0].replace('/', '_') + '.jpg')
        if idx == num_results - 1: break

def get_results_complex(database, text, word_weights, num_results, results_path):

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
        # topics = topics + w_topics
        # topics = topics / len(words)

    elif embedding == 'doc2vec':
        topics = text2topics.doc2vec(text, model, num_topics)


    elif embedding == 'glove':
        topics = text2topics.glove(text, word_weights, model, num_topics)

    elif embedding == 'glove_tfidf':
        topics = text2topics.glove_tfidf(text, model, num_topics)


    # Create empty dict for distances
    distances = {}

    # Compute distances
    for id in database:
        # distances[id] = np.linalg.norm(database[id] - topics)
        distances[id] = np.dot(database[id], topics)

    # Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)

    # Get elements with min distances
    for idx, id in enumerate(distances):
        # Copy image results
        if test_dataset == 'webvision':
            copyfile('../../../datasets/WebVision/test_images_256/' + id[0] , results_path + id[0].replace('/', '_'))
        else:
            copyfile('../../../datasets/SocialMedia/img_resized_1M/cities_instagram/' + id[0] + '.jpg', results_path + id[0].replace('/', '_') + '.jpg')
        if idx == num_results - 1: break

# Do one query
# text= 'losangeles'
# results_path = "../../../datasets/SocialMedia/retrieval_results/" + data + "/" + text.replace(' ','_') + '/'
# if not os.path.exists(results_path):
#     os.makedirs(results_path)
# # Get topic distribution from text query
# topics = text2topics(text, ldamodel, num_topics)
# get_results(database,topics,num_results,results_path)

# Do default queryes
q = []
w = [] # Weights per word (can be negative)

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

word_list = ['beach building','bike car','fish food']

for words in word_list:
    q.append(words)
    w.append('1 0')
    q.append(words)
    w.append('0 1')
    q.append(words)
    w.append('0.5 0.5')
    q.append(words)
    w.append('0.55 0.45')
    q.append(words)
    w.append('0.45 0.55')
    q.append(words)
    w.append('0.52 0.48')
    q.append(words)
    w.append('0.48 0.52')
    q.append(words)
    w.append('0.54 0.46')
    q.append(words)
    w.append('0.53 0.47')
    q.append(words)
    w.append('0.56 0.44')
    q.append(words)
    w.append('0.1 0.9')
    q.append(words)
    w.append('0.9 0.1')
    q.append(words)
    w.append('0.2 0.8')
    q.append(words)
    w.append('0.8 0.2')
    q.append(words)
    w.append('0.6 0.4')
    q.append(words)
    w.append('0.63 0.37')
    q.append(words)
    w.append('0.65 0.35')
    q.append(words)
    w.append('0.7 0.4')
    q.append(words)
    w.append('0.4 0.6')
    q.append(words)
    w.append('0.7 0.3')
    q.append(words)
    w.append('0.3 0.7')









for e,cur_q in enumerate(q):
    print(cur_q)
    cur_w = w[e]
    if test_dataset == 'webvision': results_path = "../../../datasets/WebVision/rr/" + data + "/" + cur_q.replace(' ', '_') + '-' + cur_w.replace(' ', '_') + '/'
    else: results_path = "../../../datasets/SocialMedia/retrieval_results/" + data + "/" + cur_q.replace(' ', '_') + '-' + cur_w.replace(' ', '_') + '/'
    if not os.path.exists(results_path):
        print("Creating dir: " + results_path)
        os.makedirs(results_path)


    if len(cur_q.split(' ')) == 1:

        if embedding == 'LDA': topics = text2topics.LDA(cur_q,  model, num_topics)
        elif embedding == 'word2vec_mean': topics = text2topics.word2vec_mean(cur_q, cur_w, model, num_topics)
        elif embedding == 'doc2vec': topics = text2topics.doc2vec(cur_q, model, num_topics)
        elif embedding == 'word2vec_tfidf': topics = text2topics.word2vec_tfidf(cur_q, model, num_topics, tfidf_model, tfidf_dictionary)
        elif embedding == 'glove': topics = text2topics.glove(cur_q, cur_w, model, num_topics)
        elif embedding == 'glove_tfidf': topics = text2topics.glove_tfidf(cur_q, model, num_topics)


        get_results(database, topics, num_results,results_path)

    else:
        get_results_complex(database, cur_q, cur_w, num_results, results_path)









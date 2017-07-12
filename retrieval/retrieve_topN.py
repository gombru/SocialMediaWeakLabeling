# Retrieves nearest images given a text query and saves them in an given folder

import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models

data = 'InstaCities1M_Inception_frozen_doc2vec_iter_50000'
model_name = 'doc2vec_model_instacities1M.model'
num_topics = 400 # Num LDA model topics
num_results = 5 # Num retrival results we want to take into accountnt



# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/retrieval_results/' + data +'/test.txt'
model_path = '../../../datasets/SocialMedia/models/doc2vec/' + model_name
embedding = 'doc2vec' #'word2vec_mean' 'doc2vec'
test_dataset = 'instacities1m' #'instacities1m'


# Load LDA model
print "Loading " +embedding+ " model ..."
if embedding == 'LDA': model = models.ldamodel.LdaModel.load(model_path)
elif embedding == 'word2vec_mean': model = models.Word2Vec.load(model_path)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_path)


# Load dataset
database = load_regressions_from_txt(database_path, num_topics)


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

def get_results_complex(database, text, num_results, results_path):

    words = text.split(' ')
    topics = np.zeros(num_topics)

    if embedding == 'LDA':
        for w in words:
            w_topics = text2topics.LDA(w, model, num_topics)
            topics = topics + w_topics
        topics = topics / len(words)

    elif embedding == 'word2vec_mean':
        for w in words:
            w_topics = text2topics.word2vec_mean(w, model, num_topics)
            topics = topics + w_topics
        topics = topics / len(words)

    elif embedding == 'doc2vec':
        topics = text2topics.doc2vec(text, model, num_topics)


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

# Simple

q.append('car')
q.append('skyline')
q.append('bike')

q.append('sunrise')
q.append('snow')
q.append('rain')

q.append('icecream')
q.append('cake')
q.append('pizza')

q.append('woman')
q.append('man')
q.append('kid')

# Complex
q.append('yellow car')
q.append('skyline night')
q.append('bike park')

q.append('sunrise beach')
q.append('snow ski')
q.append('rain umbrella')

q.append('icecream beach')
q.append('chocolate cake')
q.append('pizza wine')

q.append('woman bag')
q.append('man boat')
q.append('kid dog')







for cur_q in q:
    print cur_q
    if test_dataset == 'webvision': results_path = "../../../datasets/WebVision/rr/" + data + "/" + cur_q.replace(' ', '_') + '/'
    else: results_path = "../../../datasets/SocialMedia/regression_output/" + data + "/" + cur_q.replace(' ', '_') + '/'
    if not os.path.exists(results_path):
        print "Creating dir: " + results_path
        os.makedirs(results_path)
    if len(cur_q.split(' ')) == 1:
        if embedding == 'LDA': topics = text2topics.LDA(cur_q, model, num_topics)
        elif embedding == 'word2vec_mean': topics = text2topics.word2vec_mean(cur_q, model, num_topics)
        elif embedding == 'doc2vec': topics = text2topics.doc2vec(cur_q, model, num_topics)

        get_results(database, topics, num_results,results_path)

    else:
        get_results_complex(database, cur_q, num_results, results_path)









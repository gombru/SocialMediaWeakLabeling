# Retrieves nearest images given a text query and saves them in an given folder

from text2topics import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models

data = 'instagram_cities_1M_Inception_frozen_500_chunck_multiGPU_iter_500000'
lda_model = 'lda_model_cities_instagram_1M_500_5000chunck.model'
num_topics = 500 # Num LDA model topics
num_results = 5 # Num retrival results we want to take into accountnt



# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/regression_output/' + data +'/test.txt'
LDA_model_path = '../../../datasets/SocialMedia/models/LDA/' + lda_model


# Load LDA model
print "Loading LDA model ..."
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)



# Load dataset
database = load_regressions_from_txt(database_path, num_topics)


def get_results(database, topics, num_results, results_path):
    # Create empty dict for distances
    distances = {}

    #Compute distances
    for id in database:    distances[id] = np.linalg.norm(database[id] - topics)

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    # Get elements with min distances
    for idx,id in enumerate(distances):
        # Copy image results
        copyfile('../../../datasets/SocialMedia/img_resized_1M/cities_instagram/' + id[0] + '.jpg',
                 results_path + id[0].replace('/', '_') + '.jpg')
        if idx == num_results - 1: break

def get_results_complex(database, text, num_results, results_path):

    words = text.split(' ')
    topics = np.zeros(num_topics)

    for w in words:
        w_topics = text2topics(w, ldamodel, num_topics)
        # print w_topics
        for i,t in enumerate(w_topics):
            if t > 0:
                topics[i] = topics[i] + t
        topics = topics / len(words)

    #topics = topics / len(words)

    # Create empty dict for distances
    distances = {}

    # Compute distances
    for id in database:    distances[id] = np.linalg.norm(database[id] - topics)

    # Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    # Get elements with min distances
    for idx, id in enumerate(distances):
        # Copy image results
        copyfile('../../../datasets/SocialMedia/img_resized_1M/cities_instagram/' + id[0] + '.jpg',
                 results_path + id[0].replace('/', '_') + '.jpg')
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
    results_path = "../../../datasets/SocialMedia/retrieval_results/" + data + "/" + cur_q.replace(' ', '_') + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if len(cur_q.split(' ')) == 1:
        topics = text2topics(cur_q, ldamodel, num_topics)
        get_results(database, topics, num_results,results_path)

    else:
        get_results_complex(database, cur_q, num_results, results_path)









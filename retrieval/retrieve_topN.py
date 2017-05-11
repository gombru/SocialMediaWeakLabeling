# Retrieves nearest images given a text query and saves them in an given folder

from text2topics import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models

data = 'instagram_cities_1M_Inception_frozen_200_20passes_iter_170000'
lda_model = 'lda_model_cities_instagram_1M_200_20passes.model'
num_topics = 200 # Num LDA model topics
num_results = 5 # Num retrival results we want to take into account


# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/regression_output/' + data +'/test.txt'
LDA_model_path = '../../../datasets/SocialMedia/models/LDA/' + lda_model


# Load LDA model
print "Loading LDA model ..."
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)



# Load dataset
database = load_regressions_from_txt(database_path, num_topics)


def get_results(database, topics, num_results,results_path):
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


# Do one query
# text= 'basketball nba jazz'
# results_path = "../../../datasets/SocialMedia/retrieval_results/" + data + "/" + text.replace(' ','_') + '/'
# if not os.path.exists(results_path):
#     os.makedirs(results_path)
# # Get topic distribution from text query
# topics = text2topics(text, ldamodel, num_topics)
# get_results(database,topics,num_results,results_path)

# Do default queryes
q = []
q.append('taxi london')
q.append('newyork appartment')
q.append('chicago sports')
q.append('sunrise beach')
q.append('snow london')
q.append('rain rainbow')
q.append('icecream beach')
q.append('cocktail night')
q.append('food wine')
q.append('shirt girl')
q.append('man beard')
q.append('man dog')

q.append('taxi')
q.append('appartment')
q.append('bridge')
q.append('sunrise')
q.append('snow')
q.append('rain')
q.append('icecream')
q.append('cocktail')
q.append('sushi')
q.append('shirt')
q.append('beard')
q.append('kid')

q.append('dog')
q.append('cat')




for cur_q in q:
    print cur_q
    results_path = "../../../datasets/SocialMedia/retrieval_results/" + data + "/" + cur_q.replace(' ', '_') + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    topics = text2topics(cur_q, ldamodel, num_topics)
    get_results(database, topics, num_results,results_path)









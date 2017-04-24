# Retrieves nearest images given a text query and saves them in an given folder

from text2topics import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/regression_output/intagram_cities_CaffeNet2Heads_100_iter_70000/testCitiesClassification.txt'
# intagram_cities_CaffeNet_100_iter_40000  intagram_cities_CaffeNet2Heads_100_iter_70000
LDA_model_path = '../../../datasets/SocialMedia/models/LDA/lda_model_cities_instagram.model'

num_topics = 100 # Num LDA model topics
num_results = 8 # Num retrival results we want save

# text = "palmtre crew gift chicago" # Query text
text= 'snow winter white'
results_path = "../../../datasets/SocialMedia/retrieval_results/" + text.replace(' ','_') + '/'

if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load LDA model
print "Loading LDA model ..."
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)

# Get topic distribution from text query
topics = text2topics(text, ldamodel, num_topics)

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)

# Create empty dict for distances
distances = {}

# Compute distances parallel SLOWER!
# def compute_distances(id):
#     dist = np.linalg.norm(database[id]-topics)
#     return dist, id

# Compute distances in parallel
# parallelizer = Parallel(n_jobs=4)
# tasks_iterator = (delayed(compute_distances)(id) for id in database)
# r = parallelizer(tasks_iterator)
# # merging the output of the jobs
# distances = np.vstack(r)
# # Compute distances
print "Computing distances"
for id in database:    distances[id] = np.linalg.norm(database[id]-topics)

# Get elements with min distances
for n in range(0,num_results):

    # If dictionary (non-paralel)
    id = min(distances.iteritems(), key=operator.itemgetter(1))[0]
    print id + " -- " + str(distances[id])
    distances.pop(id)

    # If array (parallel) SLOWER!
    # el = np.argmin(distances[:, 0])
    # dist = distances[el,0]
    # id = distances[el,1]
    # distances[el, 0] = 1000
    # print id + " -- " + str(dist)

    # Copy image results
    copyfile('../../../datasets/SocialMedia/img_resized/cities_instagram/' + id + '.jpg', results_path + id.replace('/','_') + '.jpg')







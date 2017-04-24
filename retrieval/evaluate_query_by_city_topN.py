# Evaluates the TOPN retrieval precision using as queries the city names.

from text2topics import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models


# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/regression_output/instagram_cities_Inception_100_iter_150000_corrected/testCitiesClassification.txt'
LDA_model_path = '../../../datasets/SocialMedia/models/LDA/lda_model_cities_instagram.model'
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

num_topics = 100 # Num LDA model topics
num_results = 100 # Num retrival results we want to take into account

precisions = {}


# Compute distances parallel
# def compute_distances(id):
#     dist = np.linalg.norm(database[id]-topics)
#     return dist, id


# Load LDA model
print "Loading LDA model ..."
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)

for city in cities:

    results_path = "../../../datasets/SocialMedia/retrieval_results/" + city + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Get topic distribution from text query
    topics = text2topics(city, ldamodel, num_topics)

    # Create empty dict for ditances
    distances = {}

    # Compute distances)
    for id in database:    distances[id] = np.linalg.norm(database[id]-topics)

    # Compute distances in parallel SLOWER!
    # parallelizer = Parallel(n_jobs=4)
    # tasks_iterator = (delayed(compute_distances)(id) for id in database)
    # r = parallelizer(tasks_iterator)
    # # merging the output of the jobs
    # distances = np.vstack(r)

    # Get elements with min distances
    correct = 0
    for n in range(0,num_results):
        # If dictionrary
        id = min(distances.iteritems(), key=operator.itemgetter(1))[0]
        # print id + " -- " + str(distances[id])
        distances.pop(id)

        # If array (parallel) SLOWER! and different results..
        # el = np.argmin(distances[:, 0])
        # dist = distances[el, 0]
        # id = distances[el, 1]
        # distances[el, 0] = 1000

        # Compute retrieval precision
        if id.split('/')[0] == city:
            correct += 1
        # Copy resulting images
        copyfile('../../../datasets/SocialMedia/img_resized/cities_instagram/' + id + '.jpg', results_path + id.replace('/','_') + '.jpg')

    precisions[city] = (float(correct) / num_results) * 100
    print city + ' --> ' + str(precisions[city])

# Compute mean precision
sum = 0
for c in precisions:
    sum = precisions[c] + sum
mean_precision = sum / len(cities)

print "Mean precision: " + str(mean_precision)








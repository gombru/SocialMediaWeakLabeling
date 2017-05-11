# Evaluates the TOPN retrieval precision using as queries the city names.

from text2topics import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models
import time
import operator

data = 'instagram_cities_1M_Inception_frozen_200_20passes_iter_170000'
lda_model = 'lda_model_cities_instagram_1M_200_20passes.model'
num_topics = 200 # Num LDA model topics


# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/regression_output/' + data +'/test.txt'
LDA_model_path = '../../../datasets/SocialMedia/models/LDA/' + lda_model
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']


map = {}

# Load LDA model
print "Loading LDA model ..."
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)

for city in cities:

    # Get topic distribution from text query
    topics = text2topics(city, ldamodel, num_topics)

    # Create empty dict for ditances
    distances = {}

    # Compute distances)
    for id in database:    distances[id] = np.linalg.norm(database[id]-topics)

    city_test_images = 0
    # Compute number of tes timages per city
    for id in distances:
        if city == id.split('/')[0]:
            city_test_images+=1

    print 'Test images for ' + city + ': ' + str(city_test_images)

    # Get elements with min distances
    correct = 0
    precisions = []

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    for idx,id in enumerate(distances):

        if id[0].split('/')[0] == city:
            correct += 1
            precisions.append(float(correct)/(idx + 1))

        if correct == city_test_images: break

    map_city = 0
    for p in precisions: map_city += p
    map_city /=len(precisions)
    map[city] = map_city * 100
    print city + ': map --> ' + str(map[city])

# Compute mean precision
sum = 0
for c in map:
    sum = map[c] + sum
mean_precision = sum / len(cities)

print "Mean map: " + str(mean_precision)








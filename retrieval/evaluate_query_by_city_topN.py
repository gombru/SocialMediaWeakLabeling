# Evaluates the TOPN retrieval precision using as queries the city names.

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
num_results = 100 # Num retrival results we want to take into account



# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/SocialMedia/regression_output/' + data +'/test.txt'
LDA_model_path = '../../../datasets/SocialMedia/models/LDA/' + lda_model
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']


precisions = {}

# Load LDA model
print "Loading LDA model ..."
ldamodel = models.ldamodel.LdaModel.load(LDA_model_path)

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)

for city in cities:

    results_path = '../../../datasets/SocialMedia/retrieval_results/' + data + '/' + city + '/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Get topic distribution from text query
    topics = text2topics(city, ldamodel, num_topics)

    # Create empty dict for ditances
    distances = {}

    # Compute distances)
    for id in database:    distances[id] = np.linalg.norm(database[id]-topics)

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    # Get elements with min distances
    correct = 0
    for idx,id in enumerate(distances):

        if id[0].split('/')[0] == city:
            correct += 1
        # Copy resulting images
        copyfile('../../../datasets/SocialMedia/img_resized_1M/cities_instagram/' + id[0] + '.jpg', results_path + id[0].replace('/','_') + '.jpg')
        if idx == num_results - 1: break

    precisions[city] = (float(correct) / num_results) * 100
    print city + ' --> ' + str(precisions[city])

# Compute mean precision
sum = 0
for c in precisions:
    sum = precisions[c] + sum
mean_precision = sum / len(cities)

print "Mean precision: " + str(mean_precision)








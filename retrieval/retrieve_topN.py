from text2topics import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
from joblib import Parallel, delayed
import operator
import os
from shutil import copyfile
from gensim import corpora, models


database_path = '../../../datasets/SocialMedia/regression_output/intagram_cities_CaffeNet_40_iter_40000/testCitiesClassification.txt'

LDA_model_path = '../../../datasets/SocialMedia/models/LDA/lda_model_cities_instagram_40.model'
num_topics = 40

num_results = 6

text = "sun rise sunrise"
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

# Create empty dict for ditances
distances = {}

# def compute_distances(id):
#     dist = np.linalg.norm(database[id]-topics)
#     distances[id] = dist
#
# # Compute distances
# Parallel(n_jobs=10)(delayed(compute_distances)(id) for id in database)

print "Computing distances"
for id in database:    distances[id] = np.linalg.norm(database[id]-topics)

# Get elements with min ditances
results = []
for n in range(0,num_results):
    id = min(distances.iteritems(), key=operator.itemgetter(1))[0]
    print id + " -- " + str(distances[id])
    distances.pop(id)
    results.append(id)

    copyfile('../../../datasets/SocialMedia/img_resized/cities_instagram/' + id + '.jpg', results_path + id.replace('/','_') + '.jpg')







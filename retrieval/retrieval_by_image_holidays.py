# Evaluates the TOPN retrieval precision using as queries the city names.

from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
import operator

data = 'SocialMedia_Inception_frozen_word2vec_mean_divbymax_iter_360000'
num_topics = 400

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/Holidays/regression_output/' + data +'/test.txt'
queries_embedding_path = '../../../datasets/Holidays/regression_output/' + data +'/queries.txt'

map = {}

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
queries = load_regressions_from_txt(queries_embedding_path, num_topics + 1)


def belongs2query(id,query):
    num_id = id[:-2]
    num_query = query[:-2]
    if num_id == num_query: return True
    return False


for q in queries:

    q_name = q
    q_num_test_img = queries[q][0]
    q_embedding = queries[q][1:]


    # Create empty dict for ditances
    distances = {}

    # Compute distances)
    for id in database:    distances[id] = np.linalg.norm(database[id]-q_embedding)


    print 'Test images for ' + q_name + ': ' + str(q_num_test_img)

    # Get elements with min distances
    correct = 0
    precisions = []

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    for idx,id in enumerate(distances):

        if belongs2query(id[0].split('/')[2][:-4], q_name.split('/')[2][:-4]):
            correct += 1
            precisions.append(float(correct)/(idx + 1))

        if correct == q_num_test_img: break

    map_q = 0
    for p in precisions: map_q += p
    map_q /=len(precisions)
    map[q] = map_q * 100
    print q + ': map --> ' + str(map[q])

# Compute mean precision
sum = 0
for c in map:
    sum = map[c] + sum
mean_precision = sum / len(queries)

print "Mean map: " + str(mean_precision)








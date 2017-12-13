# Retrieves nearest images given a text query and saves them in an given folder

import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
import operator
from shutil import copyfile
import glove
import get_NN_txt_embedding
import os

txt_embeddings_path = '../../../datasets/SocialMedia/models/glove/txt_embeddings/top5_SM.txt'
txt_embeddings = open(txt_embeddings_path,'r')
data = 'triplet_frozen_glove_tfidf_SM_iter_100000'
database_path = '../../../datasets/SocialMedia/regression_output/' + data +'/test.txt'
test_dataset = 'SocialMedia'
num_topics = 400 # Num LDA model topics
num_results = 5 # Num retrival results we want to take into accountnt

# # FC text layers
# model_path = '../../../datasets/SocialMedia/models/saved/' + data + '.caffemodel'
# prototxt = '../googlenet_contrastive/prototxt/deploy_txt_FC.prototxt'
# text_NN = get_NN_txt_embedding.load_net(model_path,prototxt)

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
for id in database:
    database[id] = database[id] / sum(database[id])

# Load txt embeddings
for q in txt_embeddings:

    q = q.split(',')
    cur_q = q[0]
    topics = np.zeros(num_topics)
    for t in range(0, len(q)-1):
        topics[t] = q[t + 1]
    print topics

    if test_dataset == 'webvision': results_path = "../../../datasets/WebVision/rr/" + data + "/" + cur_q.replace(' ', '_')  + '/'
    else: results_path = "../../../datasets/SocialMedia/rr/" + data + "/" + cur_q.replace(' ', '_')  + '/'
    if not os.path.exists(results_path):
        print("Creating dir: " + results_path)
        os.makedirs(results_path)

    # Get FC txt embeddings
    topics = topics - min(topics)
    if max(topics) > 0:
        topics = topics / max(topics)
    # topics = get_NN_txt_embedding.get_NN_txt_embedding(text_NN,topics)
    topics = topics - min(topics)
    topics = topics / sum(topics)

    # Create empty dict for distances
    distances = {}


    # Compute distances
    for id in database:
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


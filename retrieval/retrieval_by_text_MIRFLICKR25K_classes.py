# Evaluates the TOPN retrieval precision using as queries the city names.

import text2topics
import numpy as np
import operator
from gensim import corpora, models
import gensim
import glove
import glob
from shutil import copyfile


def load_regressions_from_txt(path, num_topics):

    database = {}

    file = open(path, "r")

    print("Loading data ...")
    print(path)

    for line in file:
        d = line.split(',')
        regression_values = np.zeros(num_topics)
        for t in range(0,num_topics):
            # regression_values[t-1] = d[t+1]  #-1 in regression values should not be there. So I'm skipping using topic 0 somewhere
            regression_values[t] = d[t + 1]
        database[d[0]] = regression_values

    return database


data = 'WebVision_Inception_frozen_word2vec_tfidfweighted_divbymax_iter_460000'
num_topics = 400

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/MIRFLICKR25K/regression_output/' + data +'/test.txt'
filtered_topics = '../../../datasets/MIRFLICKR25K/filtered_topics/'

model_name = 'word2vec_model_webvision.model'
num_topics = 400 # Num LDA model topics
embedding = 'word2vec_tfidf'
model_path = '../../../datasets/WebVision/models/word2vec/' + model_name

# Load LDA model
print("Loading " +embedding+ " model ...")
if embedding == 'LDA': model = models.ldamodel.LdaModel.load(model_path)
elif embedding == 'word2vec_mean' or embedding == 'word2vec_tfidf': model = models.Word2Vec.load(model_path)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_path)
elif embedding == 'glove' or embedding == 'glove_tfidf': model = glove.Glove.load(model_path)

#-----------> if tfidf
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

queries = ['animals','baby','bird','car','female','lake','sea','tree','clouds','dog','sky','structures','sunset','transport','water','flower','food','indoor','plant_life','portrait','river','male','night','people']

map_classes = {}
for q in queries:
    map_classes[q] = []

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
for id in database:
    database[id] = database[id] / sum(database[id])
# queries = load_regressions_from_txt(queries_embedding_path, num_topics + 1)


img_topics = {}
# Load topics of all images
for file_name in glob.glob("/home/raulgomez/datasets/MIRFLICKR25K/filtered_topics/*.txt"):
    file = open(file_name, "r")
    lines = []
    for line in file:
        line = line.replace('\n','').replace('\t','').replace('\r','')
        lines.append(line)
    img_topics[file_name.split('/')[-1][:-4]] = lines[0].split(','), lines[1].split(',')
    file.close()


for q in queries:

    text_query = q
    if text_query == 'plant_life':
        text_query = 'plant'
    words = text_query.split(' ')
    topics = np.zeros(num_topics)

    if embedding == 'LDA':
        for w in words:
            w_topics = text2topics.LDA(w, model, num_topics)
            topics = topics + w_topics
        topics = topics / len(words)

    elif embedding == 'word2vec_mean':
        num = 0
        for w in words:
            w_topics = text2topics.word2vec_mean(w, model, num_topics)
            if np.isnan(w_topics).any():
                continue
            topics = topics + w_topics
            num += 1
        topics = topics / num

    elif embedding == 'word2vec_tfidf':
        topics = text2topics.word2vec_tfidf(text_query, model, num_topics, tfidf_model, tfidf_dictionary)
        # topics = topics + w_topics
        # topics = topics / len(words)

    elif embedding == 'doc2vec':
        topics = text2topics.doc2vec(text_query, model, num_topics)


    elif embedding == 'glove':
        topics = text2topics.glove(text_query, model, num_topics)

    elif embedding == 'glove_tfidf':
        topics = text2topics.glove_tfidf(text_query, model, num_topics)


    # Create empty dict for ditances
    distances = {}

    # Compute distances)
    for id in database:
        distances[id] = np.linalg.norm(database[id]-topics)


    # Get elements with min distances
    correct = 0

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    for idx,id in enumerate(distances):

        if img_topics[id[0]][0].__contains__(str(q)) or img_topics[id[0]][1].__contains__(str(q)):
            correct += 1
            map_classes[q].append(float(correct)/(idx + 1))

        # if correct == (numxcat[q_cat-1]): break

    map_q = 0
    for p in map_classes[q]: map_q += p
    map_q /=len(map_classes[q])
    print(q + ': map --> ' + str(map_q))
    map_classes[q] = map_q

map = 0
for q,v in map_classes.iteritems():
    map = map + v
map = map / len(queries)
print "Mean map: " + str(map)









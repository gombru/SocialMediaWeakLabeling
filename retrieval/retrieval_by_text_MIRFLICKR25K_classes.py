# Evaluates the TOPN retrieval precision using as queries the city names.

import text2topics
import numpy as np
import operator
from gensim import corpora, models
import gensim
import glove
import glob
from shutil import copyfile
from scipy.misc import imshow, imread



def load_regressions_from_txt(path, num_topics):

    database = {}

    file = open(path, "r")

    print("Loading data ...")
    print(path)

    for line in file:
        d = line.split(',')
        d = line.split(',')
        regression_values = np.zeros(num_topics)
        for t in range(0,num_topics):
            # regression_values[t-1] = d[t+1]  #-1 in regression values should not be there. So I'm skipping using topic 0 somewhere
            regression_values[t] = d[t + 1]
        database[d[0]] = regression_values

    return database

path_to_dataset = "/home/raulgomez/datasets/MIRFLICKR25K/"

data = 'mirflickr_Inception_frozen_word2vec_mean_finetuned_5000lrdecrease_half_iter_3000'
num_topics = 400

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = path_to_dataset+ 'regression_output/' + data +'/test_half.txt'
filtered_topics = path_to_dataset+ 'filtered_topics/'

model_name = 'word2vec_model_MIRFlickr_finetuned_half.model'
embedding = 'word2vec_mean'
model_path = '../../../datasets/MIRFLICKR25K/models/word2vec/' + model_name

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
strong_topics_names = ['baby','bird','car','clouds','dog','female','flower','male','night','people','portrait','river','sea','tree']

map_classes = {}
for q in queries:
    map_classes[q] = []

map_strong_classes = {}
for q in strong_topics_names:
    map_strong_classes[q] = []


# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
for id in database:
    database[id] = database[id] / sum(database[id])
# queries = load_regressions_from_txt(queries_embedding_path, num_topics + 1)


img_topics = {}
# Load topics of all images
for file_name in glob.glob(path_to_dataset+ "filtered_topics/*.txt"):
    file = open(file_name, "r")
    lines = []
    for line in file:
        line = line.replace('\n','').replace('\t','').replace('\r','')
        lines.append(line)
    img_topics[file_name.split('/')[-1][:-4]] = lines[0].split(','), lines[1].split(',')
    file.close()


strong_topics = {}
# Load indices or string topics
for file_name in glob.glob(path_to_dataset+ "annotations_r/*.txt"):
    file = open(file_name, "r")
    class_name = file_name.split('/')[-1][:-7]
    lines = []
    for line in file:
        lines.append(str(int(line)))
    strong_topics[class_name] = lines
    file.close()




for q in queries:

    text_query = q
    if text_query == 'plant_life':
        text_query = 'plant'
    words = text_query.split(' ')
    topics = np.zeros(num_topics)

    if embedding == 'LDA':
        used_words = 0
        for w in words:
            if w == '' or w == []: continue
            w_topics = text2topics.LDA(w, model, num_topics)
            if sum(w_topics) > 0:
                topics = topics + w_topics
                used_words += 1
        topics = topics / used_words

    elif embedding == 'word2vec_mean':
        word_weights = 0
        topics = text2topics.word2vec_mean(text_query, word_weights, model, num_topics)

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
    strong_correct = 0

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    for idx,id in enumerate(distances):

        # if idx < 2:
        #     img_path = "/home/raulgomez/datasets/MIRFLICKR25K/img/im" + str(id[0]) + ".jpg"
        #     imshow(imread(img_path))

        if img_topics[id[0]][0].__contains__(str(q)) or img_topics[id[0]][1].__contains__(str(q)):
            correct += 1
            map_classes[q].append(float(correct)/(idx + 1))

        if strong_topics_names.__contains__(str(q)):
            if strong_topics[str(q)].__contains__(str(int(id[0]))):
                strong_correct += 1
                map_strong_classes[q].append(float(correct)/(idx + 1))
        # if correct == (numxcat[q_cat-1]): break

    map_q = 0
    for p in map_classes[q]: map_q += p
    map_q /=len(map_classes[q])
    print(q + ': map --> ' + str(map_q))
    map_classes[q] = map_q

    if strong_topics_names.__contains__(str(q)):
        map_q = 0
        for p in map_strong_classes[q]: map_q += p
        map_q /= len(map_strong_classes[q])
        print(q + ': map strong --> ' + str(map_q))
        map_strong_classes[q] = map_q

map = 0
for q,v in map_classes.iteritems():
    map = map + v
map = map / len(queries)
print("Mean map all topics toguether: " + str(map))

map = 0
for q,v in map_classes.iteritems():
    map = map + v
for q,v in map_strong_classes.iteritems():
    map = map + v
map = map / (len(queries) + len(strong_topics_names))
print("Mean considering separate topics: " + str(map))










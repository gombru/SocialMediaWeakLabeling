# Evaluates the TOPN retrieval precision using as queries the city names.

import text2topics
import numpy as np
import operator
from gensim import corpora, models
import gensim
import glove
from load_regressions_from_txt import load_regressions_from_txt
from shutil import copyfile

cats = ['art','biology','geography','history','literature','media','music','royalty','sport','warfare']

data = 'WebVision_Inception_frozen_glove_tfidf_weighted_iter_630000'
num_topics = 400

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/Wikipedia/regression_output/' + data +'/test.txt'
test_indices_fname = '../../../datasets/Wikipedia/testset_txt_img_cat.list'


model_name = 'glove_model_WebVision.model'
num_topics = 400 # Num LDA model topics
embedding = 'glove_tfidf'
model_path = '../../../datasets/WebVision/models/glove/' + model_name

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

with open(test_indices_fname) as f:
    test_indices = f.readlines()

map = {}

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
# queries = load_regressions_from_txt(queries_embedding_path, num_topics + 1)


def belongs2query(id,query):
    num_id = id[:-2]
    num_query = query[:-2]
    if num_id == num_query: return True
    return False

# Count number of articles per category

test_img_cats = {}
numxcat = np.zeros((10,1))
for l in test_indices:
    cat = int(l.split('\t')[2])
    numxcat[cat-1] += 1
    test_img_cats[l.split('\t')[1]] = cat

print "Num images per category"
print numxcat
print sum(numxcat)

for q in test_indices:

    #Load textual query
    text_query_fname = '../../../datasets/Wikipedia/texts/' + q.split('\t')[0] + '.xml'
    q_cat = int(q.split('\t')[2])

    file = open(text_query_fname, "r")
    text_query = ""
    for line in file:
        text_query = text_query + line
    text_query = text_query.split('text>')[1][:-3]
    # text_query = "cat"
    words = text_query.split(' ')
    topics = np.zeros(num_topics)

    if embedding == 'LDA':
        for w in words:
            w_topics = text2topics.LDA(w, model, num_topics)
            topics = topics + w_topics
        topics = topics / len(words)

    elif embedding == 'word2vec_mean':
        num= 0
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
    precisions = []

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    print q_cat
    print text_query
    for idx,id in enumerate(distances):
        if idx < 6:
            copyfile('../../../datasets/Wikipedia/images/' + cats[test_img_cats[id[0]] - 1] + '/' + id[0] + '.jpg', '../../../datasets/Wikipedia/rr/' + id[0] + '.jpg')
        print test_img_cats[id[0]]
        if test_img_cats[id[0]] == q_cat:
            correct += 1
            precisions.append(float(correct)/(idx + 1))

        if correct == (numxcat[q_cat-1]): break

    map_q = 0
    for p in precisions: map_q += p
    map_q /=len(precisions)
    map[q] = map_q * 100
    print q + ': map --> ' + str(map[q])

# Compute mean precision
sum = 0
for c in map:
    sum = map[c] + sum
mean_precision = sum / len(test_indices)

print "Mean map: " + str(mean_precision)








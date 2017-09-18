# Evaluates the TOPN retrieval precision using as queries the city names.

import text2topics
import numpy as np
import operator
from gensim import corpora, models
import gensim
import glove
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



data = 'SocialMedia_Inception_frozen_glove_tfidf_iter_460000'
num_topics = 400

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/PascalVOC2007/regression_output/' + data +'/test.txt'
test_labels_fname = '../../../datasets/PascalVOC2007/labels.txt'
test_captions_fname = '../../../datasets/PascalVOC2007/tags.txt'

model_name = 'glove_model_InstaCities1M.model'
num_topics = 400 # Num LDA model topics
embedding = 'glove_tfidf'
model_path = '../../../datasets/SocialMedia/models/glove/' + model_name

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

with open(test_labels_fname) as f:
    test_indices = f.readlines()
with open(test_captions_fname) as f:
    test_captions = f.readlines()

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
numxcat = np.zeros((20,1))
for l in test_indices:
    cats = l.split()[1:]
    for c in range(0,20):
        if cats.__contains__(str(c)):
            numxcat[c] += 1
    test_img_cats[l.split()[0]] = cats

# Load captions
test_img_captions = {}
for l in test_captions:
    caption = l.split()[1:]
    all_caption = ""
    for w in caption:
        all_caption = all_caption + ' ' + w
    test_img_captions[l.split()[0]] = all_caption

print("Num images per category")
print(numxcat)
print(sum(numxcat))

for q in test_indices:

    q = q.split()[0]
    text_query = test_img_captions[q]

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

    for idx,id in enumerate(distances):

        query_labels =  test_img_cats[q]

        # print query_labels
        # print test_img_cats[id[0]]
        # print "-------\n"

        # print text_query
        #
        # if idx < 6:
        #     copyfile('../../../datasets/PascalVOC2007/JPEGImages/' + id[0], '../../../datasets/PascalVOC2007/rr/' + id[0])

        for label in query_labels:

            if test_img_cats[id[0]].__contains__(str(label)):
                correct += 1
                precisions.append(float(correct)/(idx + 1))
                break

        # if correct == (numxcat[q_cat-1]): break

    map_q = 0
    for p in precisions: map_q += p
    map_q /=len(precisions)
    map[q] = map_q * 100
    print(q + ': map --> ' + str(map[q]))

# Compute mean precision
sum = 0
for c in map:
    sum = map[c] + sum
mean_precision = sum / len(test_indices)

print("Mean map: " + str(mean_precision))








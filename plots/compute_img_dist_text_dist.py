# Evaluates the TOPN retrieval precision using as queries the city names.

import sys
sys.path.insert(0, '../retrieval/')
import text2topics
import numpy as np
import operator
from gensim import corpora, models
import gensim
import glove
import glob
import os
from shutil import copyfile
from scipy.misc import imshow, imread


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

path_to_dataset = "/home/Imatge/hd/datasets/MIRFLICKR25K/"

data = 'SocialMedia_Inception_frozen_glove_tfidf_iter_460000'
num_topics = 400
out_file = path_to_dataset+'both_embeddings/'+ data + '/data.txt'
if not os.path.exists(path_to_dataset+'both_embeddings/'+ data):
    os.makedirs(path_to_dataset+'both_embeddings/'+ data)
out_file = open(out_file, "w")


# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = path_to_dataset+'regression_output/' + data +'/test.txt'
filtered_topics = path_to_dataset+'filtered_topics/'
database_fname = path_to_dataset+'retrieval_list.txt'

model_name = 'glove_model_InstaCities1M.model'
embedding = 'glove'
model_path = '../../../datasets/SocialMedia/models/glove/' + model_name


# Load LDA model
print("Loading " +embedding+ " model ...")
if embedding == 'LDA': model = models.ldamodel.LdaModel.load(model_path)
elif embedding == 'word2vec_mean' or embedding == 'word2vec_tfidf': model = models.Word2Vec.load(model_path)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_path)
elif embedding == 'glove' or embedding == 'glove_tfidf': model = glove.Glove.load(model_path)

#-----------> if tfidf
# tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
# tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'
# tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
# tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

tfidf_model_path = '../../../datasets/SocialMedia/models/tfidf/tfidf_model_instaCities1M.model'
tfidf_dictionary_path = '../../../datasets/SocialMedia/models/tfidf/docs.dict'
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)


with open(database_fname) as f:
    queries_indices = f.readlines()

map = {}

# Load CNN embeddings
database = load_regressions_from_txt(database_path, num_topics)
#for id in database:
#    database[id] = database[id] / sum(database[id])
# queries = load_regressions_from_txt(queries_embedding_path, num_topics + 1)


# Load text
img_topics = {}
for file_name in glob.glob(path_to_dataset+'filtered_topics/*.txt'):
    file = open(file_name, "r")
    lines = []
    for line in file:
        line = line.replace('\n','').replace('\t','').replace('\r','')
        lines.append(line)
    img_topics[file_name.split('/')[-1][:-4]] = lines[0].split(','), lines[1].split(',')
    file.close()

# Compute text embeddings
text_embeddings = {}
for q in queries_indices:

    topics = img_topics[str(int(q))][0] + img_topics[str(int(q))][1]
    text_query = ""
    for l in topics:
        for cat in topics:
            if cat == 'plant_life':
                cat = 'plant'
            text_query = text_query + ' ' + cat

    #text_query = 'car yellow sunset sun'
    words = text_query.split(' ')
    print(words)
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

    else:
        print("Select a correct embedding")
        raise SystemExit(0)

    text_embeddings[str(int(q))] = topics

    # For each image save its CNN embedding and its text embedding
    img_embedding_str = ""
    text_embedding_str = ""
    for n in database[str(int(q))]:
        img_embedding_str = img_embedding_str + ' ' + str(n)
    for n in topics:
        text_embedding_str = text_embedding_str + ' ' + str(n)

    out_file.write(str(int(q)) + ',' + img_embedding_str + ',' + text_embedding_str + '\n')

out_file.close()








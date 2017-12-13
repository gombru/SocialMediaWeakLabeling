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
        regression_values = np.zeros(num_topics)
        for t in range(0,num_topics):
            # regression_values[t-1] = d[t+1]  #-1 in regression values should not be there. So I'm skipping using topic 0 somewhere
            regression_values[t] = d[t + 1]
        database[d[0]] = regression_values
    return database


data = 'triplet_softNegativeBatch_m50_notNormalize_frozen_glove_tfidf_SM_iter_260000'
num_topics = 400

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/MIRFLICKR25K/regression_output/' + data +'/retrieval_list.txt'
filtered_topics = '../../../datasets/MIRFLICKR25K/filtered_topics/'
queries_fname = '../../../datasets/MIRFLICKR25K/query_list.txt'

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
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)


# tfidf_model_path = '../../../datasets/SocialMedia/models/tfidf/tfidf_model_instaCities1M.model'
# tfidf_dictionary_path = '../../../datasets/SocialMedia/models/tfidf/docs.dict'
# tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
# tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

# FC text layers
FC = False
if FC:
    model_path = '../../../datasets/SocialMedia/models/CNNContrastive/triplet_withFC_frozen_glove_tfidf_SM_iter_60000.caffemodel'
    prototxt = '../googlenet_contrastive/prototxt/deploy_txt_FC.prototxt'
    text_NN = get_NN_txt_embedding.load_net(model_path,prototxt)




with open(queries_fname) as f:
    queries_indices = f.readlines()

map = {}

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
for id in database:
    database[id] = database[id] / sum(database[id])
# queries = load_regressions_from_txt(queries_embedding_path, num_topics + 1)


img_topics = {}
# Load topics of all images
for file_name in glob.glob("/home/Imatge/hd/datasets/MIRFLICKR25K/filtered_topics/*.txt"):
    file = open(file_name, "r")
    lines = []
    for line in file:
        line = line.replace('\n','').replace('\t','').replace('\r','')
        lines.append(line)
    img_topics[file_name.split('/')[-1][:-4]] = lines[0].split(','), lines[1].split(',')
    file.close()

# Count number of articles per category
num_per_cat = {}
for i, topics in img_topics.items():
    for cat in topics[1]:
        if num_per_cat.has_key(cat):
            num_per_cat[cat] += 1
        else:
            num_per_cat[cat] = 1

print("Num images per category")
print(num_per_cat)

for q in queries_indices:

    topics = img_topics[str(int(q))][0] + img_topics[str(int(q))][1]
    text_query = ""
    for l in topics:
        for cat in topics:
            if cat == 'plant_life':
                cat = 'plant'
            text_query = text_query + ' ' + cat

    # text_query = 'car'
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

    else:
        print("Select a correct embedding")
        raise SystemExit(0)
    #
    #
    # Create empty dict for ditances
    distances = {}


    if FC:
        topics = topics - min(topics)
        if max(topics) > 0:
            topics = topics / max(topics)
        topics = get_NN_txt_embedding.get_NN_txt_embedding(text_NN,topics)
    topics = topics - min(topics)
    topics = topics / sum(topics)


    # print topics
    # Compute distances)
    for id in database:
        distances[id] = np.linalg.norm(database[id]-topics)


    # Get elements with min distances
    correct = 0
    precisions = []

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    query_labels = img_topics[str(int(q))][0] + img_topics[str(int(q))][1]
    # query_labels = img_topics[str(int(q))][1]
    for idx,id in enumerate(distances):

        # if idx < 2:
        #     img_path = "/home/raulgomez/datasets/MIRFLICKR25K/img/im" + str(id[0]) + ".jpg"
        #     imshow(imread(img_path))

        for label in query_labels:
            if img_topics[id[0]][0].__contains__(str(label)) or img_topics[id[0]][1].__contains__(str(label)):
            # if img_topics[id[0]][1].__contains__(str(label)):
                correct += 1
                precisions.append(float(correct)/(idx + 1))
                break

        # if correct == (numxcat[q_cat-1]): break

    map_q = 0
    for p in precisions: map_q += p
    map_q /=len(precisions)
    map[q] = map_q * 100
    print(': map --> ' + str(map[q]))

# Compute mean precision
sum = 0
for c in map:
    sum = map[c] + sum
mean_precision = sum / len(queries_indices)

print("Mean map: " + str(mean_precision))







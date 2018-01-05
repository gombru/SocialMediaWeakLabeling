# Retrieves nearest images given a text query and saves them in an given folder

import text2topics
from load_regressions_from_txt import load_regressions_from_txt
import numpy as np
import operator
import json
from shutil import copyfile
from gensim import corpora, models
import gensim
import glove


data = 'SocialMedia_Inception_frozen_word2vec_tfidfweighted_divbymax_iter_150000'
model_name = 'word2vec_model_InstaCities1M.model'
num_topics = 400 # Num LDA model topics
num_results = 5 # Num retrival results we want to take into accountnt
split = 'test1k'

#-----------> if tfidf
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/WebVision/models/tfidf/docs.dict'
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

anns = json.load(open('../../../datasets/COCO/annotations/'+ split +'.json'))

# Topic distribution given by the CNN to test images. .txt file with format city/{im_id},score1,score2 ...
database_path = '../../../datasets/COCO/regression_output/' + data + '/' + split + '.txt'
model_path = '../../../datasets/SocialMedia/models/word2vec/' + model_name
embedding = 'word2vec_mean' #'word2vec_mean' 'doc2vec' 'LDA' 'word2vec_tfidf' 'glove' 'glove_tfidf'

# Load LDA model
print("Loading " +embedding+ " model ...")
if embedding == 'LDA': model = models.ldamodel.LdaModel.load(model_path)
elif embedding == 'word2vec_mean' or embedding == 'word2vec_tfidf': model = models.Word2Vec.load(model_path)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_path)
elif embedding == 'glove' or embedding == 'glove_tfidf': model = glove.Glove.load(model_path)

# FC text layers
FC = False
if FC:
    model_path = '../../../datasets/SocialMedia/models/CNNContrastive/triplet_withFC_frozen_glove_tfidf_SM_iter_60000.caffemodel'
    prototxt = '../googlenet_contrastive/prototxt/deploy_txt_FC.prototxt'
    text_NN = get_NN_txt_embedding.load_net(model_path,prototxt)

# Load dataset
database = load_regressions_from_txt(database_path, num_topics)
for id in database:
    database[id] = database[id] / sum(database[id])

def get_results_complex(database, text, word_weights, im_id, num_results):

    words = text.split(' ')
    topics = np.zeros(num_topics)

    if embedding == 'LDA':
        for w in words:
            w_topics = text2topics.LDA(w, model, num_topics)
            topics = topics + w_topics
        topics = topics / len(words)

    elif embedding == 'word2vec_mean':
        topics = text2topics.word2vec_mean(text, word_weights, model, num_topics)

    elif embedding == 'word2vec_tfidf':
        topics = text2topics.word2vec_tfidf(text, model, num_topics, tfidf_model, tfidf_dictionary)

    elif embedding == 'doc2vec':
        topics = text2topics.doc2vec(text, model, num_topics)


    elif embedding == 'glove':
        topics = text2topics.glove(text, word_weights, model, num_topics)

    elif embedding == 'glove_tfidf':
        topics = text2topics.glove_tfidf(text, model, num_topics)

    if FC:
        topics = topics - min(topics)
        if max(topics) > 0:
            topics = topics / max(topics)
            topics = get_NN_txt_embedding.get_NN_txt_embedding(text_NN,topics)


    topics = topics - min(topics)
    topics = topics / max(topics)

    # Create empty dict for distances
    distances = {}

    # Compute distances
    for id in database:
        distances[id] = np.dot(database[id], topics)

    # Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)

    # Get elements with min distances
    results = []
    for idx, id in enumerate(distances):
        copyfile('../../../datasets/COCO/val2014/' + id[0], '../../../datasets/COCO/rr/' + id[0].replace('/', '_'))
        results.append(int(id[0][13:-4]))
        if idx == num_results - 1: break

    if im_id in results:
        return True
    else:
        return False

recall = .0
for id in anns:
    cur_w = '1'
    for el in anns[id]['caption'].split(' '):
        cur_w = cur_w + ' 1'
    print anns[id]['caption']
    if get_results_complex(database, anns[id]['caption'], cur_w, anns[id]['image_id'], num_results):
        recall += 1
recall = 100* recall / len(anns)
print "Recall: " + str(recall)






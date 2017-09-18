# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN regression training
# It also creates the splits train/val/test randomly

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
import string
from joblib import Parallel, delayed
import numpy as np
from random import randint
import json
import gensim
import multiprocessing

# Load data and model
indices_fname = '../../../datasets/Wikipedia/trainset_txt_img_cat.list'
model_path = '../../../datasets/Wikipedia/models/word2vec/word2vec_model_wikipedia.model'
tfidf_weighted = False
# tfidf_model_path = '../../../datasets/Wikipedia/models/tfidf/tfidf_model_wikipedia.model'
# tfidf_dictionary_path = '../../../datasets/Wikipedia/models/tfidf/docs.dict'

cats = ['art','biology','geography','history','literature','media','music','royalty','sport','warfare']


# Create output files
dir = "word2vec_mean_gt"
if tfidf_weighted: dir = "word2vec_tfidf_weighted_gt"

train_gt_path = '../../../datasets/Wikipedia/' + dir + '/' + 'train_wikipedia.txt'
train_file = open(train_gt_path, "w")
val_gt_path = '../../../datasets/Wikipedia/' + dir + '/' + 'val_wikipedia.txt'
val_file = open(val_gt_path, "w")

model = gensim.models.Word2Vec.load(model_path)
# tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
# tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

size = 400 # vector size
cores = 8#multiprocessing.cpu_count()

whitelist = string.letters + string.digits + ' '
words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']
# create English stop words list
en_stop = get_stop_words('en')


def infer_LDA(d):

        caption = d[1]
        tokens = gensim.utils.simple_preprocess(caption)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]

        embedding = np.zeros(size)

        if not tfidf_weighted:
            c = 0
            for tok in tokens_filtered:
                try:
                    embedding += model[tok]
                    c += 1
                except:
                    #print "Word not in model: " + tok
                    continue
            if c > 0:
                embedding /= c

        if tfidf_weighted:
            vec = tfidf_dictionary.doc2bow(tokens_filtered)
            vec_tfidf = tfidf_model[vec]
            for tok in vec_tfidf:
                word_embedding = model[tfidf_dictionary[tok[0]]]
                embedding += word_embedding * tok[1]

        embedding = embedding - min(embedding)
        if max(embedding) > 0:
            embedding = embedding / max(embedding)


        # Add zeros to topics without score
        out_string = ''
        for t in range(0,size):
            out_string = out_string + ',' + str(embedding[t])

        # print id + topic_probs
        return d[0] + out_string


with open(indices_fname) as f:
    indices = f.readlines()

strings = []
for l in indices:
    file = l.split('\t')[0]
    text_file = '../../../datasets/Wikipedia/texts/' + l.split('\t')[0] + '.xml'

    caption = ""
    filtered_caption = ""
    file = open(text_file, "r")
    for line in file:
        caption = caption + line
    # Replace hashtags with spaces
    caption = caption.replace('#', ' ')
    caption = caption.split('text>')[1][:-3]
    # Keep only letters and numbers
    for char in caption:
        if char in whitelist:
            filtered_caption += char
    img_name = cats[int(l.split('\t')[2])-1] + '/' + l.split('\t')[1]
    strings.append(infer_LDA([img_name, filtered_caption]))


# print "Number of elements " + str(len(data))
# parallelizer = Parallel(n_jobs=cores)
# print "Infering word2vec scores"
# tasks_iterator = (delayed(infer_LDA)(d) for d in data)
# r = parallelizer(tasks_iterator)
# # merging the output of the jobs
# strings = np.vstack(r)

print "Resulting number of elements " + str(len(strings))

print "Saving results"
for s in strings:
    # Create splits random
    try:
        split = randint(0,25)
        if split < 1:
            val_file.write(s + '\n')
        else: train_file.write(s + '\n')
    except:
        print "Error writing to file: "
        print s
        continue


train_file.close()
val_file.close()

print "Done"

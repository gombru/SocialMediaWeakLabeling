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
import glove

# Load data and model
model_path = '../../../datasets/SocialMedia/models/glove/glove_model_InstaCities1M.model'

tfidf_weighted = True
print("TFIDF weighted: " + str(tfidf_weighted))
# tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
# tfidf_dictionary_path = '../../../datasets/SocialMedia/models/tfidf/docs.dict'

# Create output files
dir = "glove_mean_gt"
if tfidf_weighted: dir = "glove_tfidf_weighted_gt"

train_gt_path = '../../../datasets/MIRFLICKR25K/' + dir + '/' + 'train.txt'
train_file = open(train_gt_path, "w")
val_gt_path = '../../../datasets/MIRFLICKR25K/' + dir + '/' + 'val.txt'
val_file = open(val_gt_path, "w")

model = glove.Glove.load(model_path)
# tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
# tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

size = 400 # vector size

whitelist = string.ascii_letters + string.digits + ' '
words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']
# create English stop words list
en_stop = get_stop_words('en')


def infer_glove(d):

        caption = d[1]
        filtered_caption = ""

        # Replace hashtags with spaces
        caption = caption.replace('#',' ')

        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        filtered_caption = filtered_caption.lower()
        #Gensim simple_preproces instead tokenizer
        tokens = gensim.utils.simple_preprocess(filtered_caption)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        tokens_filtered = stopped_tokens
        #tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]


        embedding = np.zeros(size)

        if not tfidf_weighted:
            c = 0
            for tok in tokens_filtered:
                try:
                    embedding += model.word_vectors[model.dictionary[tok]]
                    # print model.word_vectors[model.dictionary[tok]]
                    c += 1
                except:
                    # print "Word not in model: " + tok
                    continue
            if c > 0:
                embedding /= c

        if tfidf_weighted:
            # vec = tfidf_dictionary.doc2bow(tokens_filtered)
            # vec_tfidf = tfidf_model[vec]
            # for tok in vec_tfidf:
            #     word_embedding = model[tfidf_dictionary[tok[0]]]
            #     embedding += word_embedding * tok[1]

            # print("Using GLOVE paragraph embedding (similar to tfidf weighting)")
            # print(tokens_filtered)
            embedding = model.transform_paragraph(tokens_filtered, 50, True)

        embedding = embedding - min(embedding)
        if max(embedding) > 0:
            embedding = embedding / max(embedding)

        if np.isnan(embedding).any():
            embedding = np.zeros(size)

        # Add zeros to topics without score
        out_string = ''
        for t in range(0,size):
            out_string = out_string + ',' + str(embedding[t])

        # print id + topic_probs
        return d[0] + out_string



print("Loading MIRFlickr data")
strings = []
retreival_indices_ints = []
# Read topics for only retrieval images
retrieval_list_fname = '../../../datasets/MIRFLICKR25K/retrieval_list.txt'
with open(retrieval_list_fname) as f:
    retrieval_indices = f.readlines()
for q in retrieval_indices:
    retreival_indices_ints.append(int(q))
for file_name in glob.glob("/home/raulgomez/datasets/MIRFLICKR25K/filtered_topics/*.txt"):
    if int(file_name.split('/')[-1][:-4]) not in retreival_indices_ints:
        continue
    file = open(file_name, "r")
    lines = []
    for line in file:
        line = line.replace('\n', '').replace('\t', '').replace('\r', '')
        line = line.replace('plant_life','plant')
        lines.append(line)

    filtered_caption = lines[0].replace(',',' ') + ' ' + lines[1].replace(',',' ')
    strings.append(infer_glove([file_name.split('/')[-1][:-4], filtered_caption]))


# print "Number of elements " + str(len(data))
# parallelizer = Parallel(n_jobs=cores)
# print "Infering word2vec scores"
# tasks_iterator = (delayed(infer_LDA)(d) for d in data)
# r = parallelizer(tasks_iterator)
# # merging the output of the jobs
# strings = np.vstack(r)

print("Resulting number of elements " + str(len(strings)))

print("Saving results")
for s in strings:
    # Create splits random
    try:
        split = randint(0,25)
        if split < 1:
            val_file.write(s + '\n')
        else: train_file.write(s + '\n')
    except:
        print("Error writing to file: ")
        print s
        continue


train_file.close()
val_file.close()

print("Done")

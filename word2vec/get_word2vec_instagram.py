# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN regression training
# It also creates the splits train/val/test randomly

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import os
from random import randint
import string
from joblib import Parallel, delayed
import numpy as np
import gensim
import multiprocessing
import json
import glob


# Load data and model
base_path = '../../../hd/datasets/instaBarcelona/instaMiro/'
data_path = base_path + 'data/'
model_path = base_path + 'models/word2vec/instaMiro_word2vec.model'

# Create output files
dir = "word2vec_l2norm_gt"
gt_path_train = base_path + dir + '/train_l2norm.txt'
gt_path_val = base_path + dir + '/val_l2norm.txt'
# gt_path_test = base_path + dir + '/test_l2norm.txt'
train_file = open(gt_path_train, "w")
val_file = open(gt_path_val, "w")
# test_file = open(gt_path_test, "w")

words2filter = ['miro','joanmiro','rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']


model = gensim.models.Word2Vec.load(model_path)

size = 300 # vector size
cores = multiprocessing.cpu_count()


def img_exists(path):
    im_path = base_path + "img_resized/" + path + ".jpg"
    return os.path.isfile(im_path)

def get_data():
    data = {}
    without_caption = 0
    print "Loading instagram data"
    for i, file_name in enumerate(glob.glob(data_path + "/*.json")):
        # Preprocess text: Here I only filter to be able to look for cities. The text processing will be done when training text models, because I want to save the captions as they are
        try:
            data = json.load(open(file_name, "r"))
            caption = data['edge_media_to_caption']['edges'][0]['node']['text']
        except:
            without_caption += 1
            print("Caption not found")
            continue
        data[file_name.split['/'][-1].split['.'][0]] = caption

    print "Number of posts without_caption: " + str(without_caption) + " out of " + str(
        without_caption + len(data))

data = get_data()

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
es_stop = get_stop_words('es')
ca_stop = get_stop_words('ca')
for w in es_stop:
    en_stop.append(w)
for w in ca_stop:
    en_stop.append(w)

# add own stop words
for w in words2filter:
    en_stop.append(w)

whitelist = string.letters + string.digits + ' '


def infer_word2vec(id, caption):

    embedding = np.zeros(size)
    filtered_caption = ""
    caption = caption.replace('#', ' ')
    for char in caption:
        if char in whitelist:
            filtered_caption += char
    filtered_caption = filtered_caption.decode('utf-8').lower()
    #Gensim simple_preproces instead tokenizer
    tokens = gensim.utils.simple_preprocess(filtered_caption)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]

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


    if min(embedding) < 0:
        embedding = embedding - min(embedding)

    # L2 normalized
    if sum(embedding) > 0:
        embedding = embedding / np.linalg.norm(embedding)

    return id, embedding



parallelizer = Parallel(n_jobs=cores)
tasks_iterator = (delayed(infer_word2vec)(id,caption) for id, caption in data.iteritems())
results = parallelizer(tasks_iterator)
count = 0
skipped = 0
for r in results:
    # Create splits random
    if sum(r[1]) == 0:
        print "Continuing, sum = 0"
        skipped += 1
        continue

    # Check if image file exists
    if not img_exists(str(r[0])):
        print "Img file does not exist"
        continue

    try:
        out = str(r[0])
        for v in r[1]:
            out = out + ',' + str(v)
        out = out + '\n'
        split = randint(0,19)
        if split < 19:
            train_file.write(out)
        else: val_file.write(out)
        # elif split == 19: val_file.write(out)
        # else: test_file.write(out)
        count += 0
    except:
        print "Error writing to file: "
        print r[0]
        continue


train_file.close()
val_file.close()
# test_file.close()

print "Done. Skipped: " + str(skipped)  + " Saved: " + str(count)

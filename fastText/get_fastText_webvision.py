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
text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/WebVision/models/fastText/fastText_model_WebVision.model'
tfidf_weighted = False
tfidf_model_path = '../../../datasets/WebVision/models/tfidf/tfidf_model_webvision.model'
tfidf_dictionary_path = '../../../datasets/SocialMedia/models/tfidf/docs.dict'

# Create output files
dir = "fastText_mean_gt"
if tfidf_weighted: dir = "fastText_tfidf_weighted_gt"

train_gt_path = '../../../datasets/WebVision/' + dir + '/' + 'train_webvision.txt'
train_file = open(train_gt_path, "w")
val_gt_path = '../../../datasets/WebVision/' + dir + '/' + 'myval_webvision.txt'
val_file = open(val_gt_path, "w")

model = gensim.models.FastText.load(model_path)
tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)

size = 400 # vector size
cores = 8#multiprocessing.cpu_count()

whitelist = string.letters + string.digits + ' '
words2filter = ['wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']
# create English stop words list
en_stop = get_stop_words('en')


def infer_LDA(d):

        caption = d[2]
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
        return d[0] + ',' + str(d[1]) + out_string



sources=['google','flickr']
former_filename = ' '
for s in sources:
    data = []
    print 'Loading data from ' + s
    data_file = open(text_data_path + 'info/train_meta_list_' + s + '.txt', "r")
    img_list_file = open(text_data_path + 'info/train_filelist_' + s + '.txt', "r")

    img_names = []
    img_classes = []
    for line in img_list_file:
        img_names.append(line.split(' ')[0])
        img_classes.append(int(line.split(' ')[1]))

    for i,line in enumerate(data_file):

        filename = line.split(' ')[0].replace(s,s+'_json')
        idx = int(line.split(' ')[1])

        if filename != former_filename:
            # print filename
            json_data = open(text_data_path + filename)
            d = json.load(json_data)
            former_filename = filename

        caption = ''

        if d[idx - 1].has_key('description'): caption = caption + d[idx - 1]['description'] + ' '
        if d[idx - 1].has_key('title'): caption = caption + d[idx - 1]['title'] + ' '
        if d[idx - 1].has_key('tags'):
            for tag in d[idx-1]['tags']:
                caption = caption + tag + ' '

        data.append([img_names[i],img_classes[i],caption])


    print "Number of elements for " + s + ": " + str(len(data))
    parallelizer = Parallel(n_jobs=cores)
    print "Infering fastText scores"
    tasks_iterator = (delayed(infer_LDA)(d) for d in data)
    r = parallelizer(tasks_iterator)
    # merging the output of the jobs
    strings = np.vstack(r)

    print "Resulting number of elements for " + s + ": " + str(len(strings))

    print "Saving results"
    for s in strings:
        # Create splits random
        try:
            split = randint(0,19)
            if split < 1:
                val_file.write(s[0] + '\n')
            else: train_file.write(s[0] + '\n')
        except:
            print "Error writing to file: "
            print s[0]
            continue

    data_file.close()
    img_list_file.close()

train_file.close()
val_file.close()

print "Done"

# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN regression training
# It also creates the splits train/val/test randomly

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
from random import randint
import string
from joblib import Parallel, delayed
import numpy as np
import gensim
import multiprocessing
import glove
import json

# Load data and model
text_data_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/'
model_path = '../../../datasets/SocialMedia/models/glove/glove_model_InstaCities1M.model'

# Create output files
dir = "glove_perWord_gt"
gt_path_train = '../../../datasets/SocialMedia/' + dir + '/train_InstaCities1M_divbymax_perWord.txt'
gt_path_val = '../../../datasets/SocialMedia/' + dir + '/val_InstaCities1M_divbymax_perWord.txt'
gt_path_test = '../../../datasets/SocialMedia/' + dir + '/test_InstaCities1M_divbymax_perWord.txt'
train_file = open(gt_path_train, "w")
val_file = open(gt_path_val, "w")
test_file = open(gt_path_test, "w")

# Output json with per word embeddings
dir_word_embeddings = "../../../datasets/SocialMedia/glove_perWord_gt/word_embeddings/"

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

model = glove.Glove.load(model_path)
# tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
# tfidf_dictionary = gensim.corpora.Dictionary.load(tfidf_dictionary_path)


size = 400 # vector size
cores = multiprocessing.cpu_count()

num_images_per_city = 100000
num_val = num_images_per_city * 0.05
num_test = num_images_per_city * 0.15

words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog']

# create English stop words list
en_stop = get_stop_words('en')

# add own stop words
for w in words2filter:
    en_stop.append(w)

whitelist = string.ascii_letters + string.digits + ' '


def infer_glove(file_name):

    id = file_name.split('/')[-1][:-4]

    with open(file_name, 'r') as file:

        caption = ""
        filtered_caption = ""

        for line in file:
            caption = caption + line

        # Replace hashtags with spaces
        caption = caption.replace('#',' ')

        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        filtered_caption = filtered_caption.lower()
        #Gensim simple_preproces instead tokenizer
        tokens = gensim.utils.simple_preprocess(filtered_caption)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        tokens_filtered = stopped_tokens

        embeddings = {}

        for tok in tokens_filtered:
            try:
                embedding = model.word_vectors[model.dictionary[tok]]
                embedding = embedding - min(embedding)
                if max(embedding) > 0:
                    embedding = embedding / max(embedding)
                if np.isnan(embedding).any():
                    continue
                embeddings[tok] = embedding.tolist()
            except:
                continue

        # Save json with the embeddings of each word as a dict
        if len(embeddings.keys()) == 0:
            return ""
        else:
            with open(dir_word_embeddings + id + '.json','w') as f:
                json.dump(embeddings,f)

        return city + '/' + id


for city in cities:
        print(city)
        count = 0

        parallelizer = Parallel(n_jobs=cores)
        tasks_iterator = (delayed(infer_glove)(file_name) for file_name in glob.glob(text_data_path + city + "/*.txt"))
        r = parallelizer(tasks_iterator)
        # merging the output of the jobs
        strings = np.vstack(r)

        for s in strings:
            # Create splits same number of images per class in each split
            try:
                if len(s) < 1: continue
                if count < num_test:
                    test_file.write(s[0] + '\n')
                elif count < num_test + num_val:
                    val_file.write(s[0] + '\n')
                else:
                    train_file.write(s[0] + '\n')
                count += 1
            except:
                print("Error writing to file: ")
                print(s[0])
                continue


train_file.close()
val_file.close()
test_file.close()

print("Done")

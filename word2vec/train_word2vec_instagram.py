from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import gensim
import string
import glob
import multiprocessing
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

cores = multiprocessing.cpu_count()

whitelist = string.letters + string.digits + ' '
data_path = '../../../hd/datasets/instaMiro/data/'
model_path = '../../../hd/datasets/instaMiro/models/word2vec/instaMiro_word2vec.model'
words2filter = ['miro','joanmiro','rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']

size = 300 # vector size
min_count = 5 # minimum word count to 2 in order to give higher frequency words more weighting
iter = 25 # iterating over the training corpus x times
window = 8

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
# Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()
texts = [] #List of lists of tokens


def get_data():
    # -- LOAD DATA FROM INSTAGRAM --
    posts_text = []
    print "Loading instagram data"
    for i, file_name in enumerate(glob.glob(data_path + "/*.json")):
        caption = ""
        filtered_caption = ""
        data = json.load(open(file_name, "r"))

        # Check if post has caption
        if 'caption' not in data:
            continue

        # Preprocess text: Here I only filter to be able to look for cities. The text processing will be done when training text models, because I want to save the captions as they are
        caption = data['caption']

        caption = caption.replace('#', ' ')
        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char
        posts_text.append(filtered_caption.decode('utf-8').lower())

    return posts_text


posts_text = get_data()

print "Number of posts: " + str(len(posts_text))

print "Creating tokens"
c= 0

for t in posts_text:

    c += 1
    if c % 10000 == 0:
        print c

    try:
        t = t.lower()
        #Gensim simple_preproces instead tokenizer
        tokens = gensim.utils.simple_preprocess(t)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token
        # text = [p_stemmer.stem(i) for i in stopped_tokens]
        # add proceced text to list of lists
        texts.append(stopped_tokens)
    except:
        continue

posts_text = []

#Train the model
print "Training ..."
model = gensim.models.Word2Vec(texts, size=size, min_count=min_count, workers=cores, iter=iter, window=window)
model.save(model_path)
print "DONE"
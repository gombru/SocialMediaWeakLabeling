import gensim
from pyLDAvis.gensim import prepare
import pyLDAvis
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
import string
import numpy as np
import pandas as pd


model_path = '../../../datasets/SocialMedia/models/LDA/lda_model_cities_instagram_1M_500_5000chunck.model'

# Create training corpus again (it is needed)
whitelist = string.letters + string.digits + ' '
instagram_text_data_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/'
words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog']

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

num_topics = 500
repetition_threshold = 20

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

posts_text = []
texts = [] #List of lists of tokens

# -- LOAD DATA FROM INSTAGRAM --
for city in cities:
    c=0
    print "Loading data from " + city
    for file_name in glob.glob(instagram_text_data_path + city + "/*.txt"):
        caption = ""
        filtered_caption = ""
        file = open(file_name, "r")
        for line in file:
            caption =  caption + line

        # Replace hashtags with spaces
        capion = caption.replace('#', ' ')
        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        posts_text.append(filtered_caption.decode('utf-8').lower())
        # print filtered_caption.decode('utf-8')
        if c == 50: break
        c+=1


print "Number of posts: " + str(len(posts_text))

print "Creating tokens"
c= 0

for t in posts_text:

    c += 1
    if c % 10000 == 0:
        print c

    try:
        t = t.lower()
        tokens = tokenizer.tokenize(t)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token
        text = [p_stemmer.stem(i) for i in stopped_tokens]
        # add proceced text to list of lists
        texts.append(text)
    except:
        continue
    #Remove element from list if memory limitation TODO
    #del tweets_text[0]
posts_text = []

# Remove words that appear less than N times
print "Removing words appearing less than: " + str(repetition_threshold)
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > repetition_threshold] for text in texts]

# Construct a document-term matrix to understand how frewuently each term occurs within each document
# The Dictionary() function traverses texts, assigning a unique integer id to each unique token while also collecting word counts and relevant statistics.
# To see each token unique integer id, try print(dictionary.token2id)
dictionary = corpora.Dictionary(texts)
dictionary.compactify()
dictionary.save('dict.dict')

# Convert dictionary to a BoW
# The result is a list of vectors equal to the number of documents. Each document containts tumples (term ID, term frequency)
corpus = [dictionary.doc2bow(text) for text in texts]

texts = []

#Randomize training elements
corpus = np.random.permutation(corpus)
gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)

# Create csc matrix of corpus (speed up if calling multiple times prepare)
#corpus_csc = gensim.matutils.corpus2csc(corpus)


dictionary = gensim.corpora.Dictionary.load('dict.dict')
corpus = gensim.corpora.MmCorpus('corpus.mm')
lda = gensim.models.ldamodel.LdaModel.load(model_path)

vis_data = prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis_data,'visualization.html')
pyLDAvis.display(vis_data)


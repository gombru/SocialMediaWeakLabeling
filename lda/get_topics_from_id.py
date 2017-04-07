# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN training

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
from random import randint
import string

# It also creates the splits train/val/test randomly

id = '1480255059174847010'
city = 'sydney'
file_name =  '../../../datasets/SocialMedia/captions_resized/cities_instagram/' + city + '/' + id + '.txt'
model_path = '../../../datasets/SocialMedia/models/LDA/lda_model_cities_instagram.model'


num_topics = 100
words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog']
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)

whitelist = string.letters + string.digits + ' '

ldamodel = models.ldamodel.LdaModel.load(model_path)
tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

caption = ""
filtered_caption = ""

with open(file_name, 'r') as file:
    for line in file:
        caption = caption + line

# --Replace hashtags with spaces
capion = caption.replace('#',' ')

# -- Keep only letters and numbers
for char in caption:
    if char in whitelist:
        filtered_caption += char

filtered_caption = filtered_caption.lower()

tokens = tokenizer.tokenize(filtered_caption)
# remove stop words from tokens
stopped_tokens = [i for i in tokens if not i in en_stop]
# stem token
print stopped_tokens
#Check this error
while "aed" in stopped_tokens:
    stopped_tokens.remove("aed")
    print "aed error"

text = [p_stemmer.stem(i) for i in stopped_tokens]
bow = ldamodel.id2word.doc2bow(tokens)
r = ldamodel[bow]
print r

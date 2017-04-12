# Trains and saves an LDA model with the given text files.

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
import string


whitelist = string.letters + string.digits + ' '

# tweets_text_data_path = '../../../datasets/SocialMedia/text/text_cities_1day.txt'
instagram_text_data_path = '../../../datasets/SocialMedia/captions_resized/cities_instagram/'

model_path = '../../../datasets/SocialMedia/models/lda_model_cities_instagram.model'

words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog']

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

num_topics = 100
threads = 6
passes = 20

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

#ids = []
posts_text = []
texts = [] #List of lists of tokens

# -- LOAD DATA FROM TWITTER --
# posts_text_file = open(tweets_text_data_path, "r")

# print "Loading data ..."
# for line in tweets_text_file:
#     info = line.split(',')
#     #ids.append(info[0])
#     tweets_text.append(info[1].decode('utf-8'))
#
# tweets_text_file.close()


# -- LOAD DATA FROM INSTAGRAM --

for city in cities:
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
# Construct a document-term matrix to understand how frewuently each term occurs within each document
# The Dictionary() function traverses texts, assigning a unique integer id to each unique token while also collecting word counts and relevant statistics.
# To see each token unique integer id, try print(dictionary.token2id)
dictionary = corpora.Dictionary(texts)

# Convert dictionary to a BoW
# The result is a list of vectors equal to the number of documents. Each document containts tumples (term ID, term frequency)
corpus = [dictionary.doc2bow(text) for text in texts]

texts = []

# Generate an LDA model
print "Creating LDA model"
#ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
ldamodel = models.LdaMulticore(corpus, num_topics=num_topics, id2word = dictionary, passes=passes, workers=threads)
ldamodel.save(model_path)
# Our LDA model is now stored as ldamodel

print(ldamodel.print_topics(num_topics=8, num_words=10))









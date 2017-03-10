from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

tweets_text_data_path = '../../datasets/SocialMedia/text/text_trump_test.txt'
model_path = '../../datasets/SocialMedia/models/lda_model_trump_test.model'

words2filter = ['rt', 'http', 't', 'trump', 'gt', 'co', 's', 'https', 'http','tweet', 'pictur','markars_']

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


# -- LOAD DATA --
tweets_text_file = open(tweets_text_data_path, "r")
ids = []
tweets_text = []
texts = [] #List of lists of tokens

for line in tweets_text_file:
    info = line.split(',')
    ids.append(info[0])
    tweets_text.append(info[1].decode('utf-8'))

print "Number of tweets: " + str(len(ids))

print "Creating tokens"
c= 0

for t in tweets_text:

    c += 1
    if c % 100 == 0:
        print c

    t = t.lower()
    tokens = tokenizer.tokenize(t)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem token
    text = [p_stemmer.stem(i) for i in stopped_tokens]
    # add proceced text to list of lists
    texts.append(text)

# Construct a document-term matrix to understand how frewuently each term occurs within each document
# The Dictionary() function traverses texts, assigning a unique integer id to each unique token while also collecting word counts and relevant statistics.
# To see each token unique integer id, try print(dictionary.token2id)
dictionary = corpora.Dictionary(texts)

# Convert dictionary to a BoW
# The result is a list of vectors equal to the number of documents. Each document containts tumples (term ID, term frequency)
corpus = [dictionary.doc2bow(text) for text in texts]

# Generate an LDA model
print "Creating LDA model"
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
ldamodel.save(model_path)
# Our LDA model is now stored as ldamodel

print(ldamodel.print_topics(num_topics=3, num_words=10))









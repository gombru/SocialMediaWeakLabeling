# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN training

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
from random import randint

# It also creates the splits train/val/test randomly

tweets_text_data_path = '../../../datasets/SocialMedia/tweets_info/cities_1day_unique/'
model_path = '../../../datasets/SocialMedia/models/lda_model_cities_1day.model'

gt_path_train = '../../../datasets/SocialMedia/lda_gt/cities_1day_unique/train.txt'
gt_path_val = '../../../datasets/SocialMedia/lda_gt/cities_1day_unique/val.txt'
gt_path_test = '../../../datasets/SocialMedia/lda_gt/cities_1day_unique/test.txt'

num_topics = 50
words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog']
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)

cities = ['paris','istanbul','rome','prague','milan','barcelona','amsterdam','vienna','moscow','berlin','madrid']

ldamodel = models.ldamodel.LdaModel.load(model_path)
tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

topics = ldamodel.print_topics(num_topics=num_topics, num_words=10)
print topics

file = open('topics.txt', 'w')

for item in topics:
    file.write("%s\n" % item[1])

c= 0

train_file = open(gt_path_train, "w")
val_file = open(gt_path_val, "w")
test_file = open(gt_path_test, "w")

for city in cities:
    for file in glob.glob(tweets_text_data_path + city + "/*.txt"):

        with open(file, 'r') as fin:
            lines = fin.readlines()
            id = lines[0][:-1]
            tweet_city = lines[-1]

            t = ""
            try:
                t = lines[-1][:-1].decode('utf-8').lower()
            except:
                print "Error decoding utf-8"
                continue

            c += 1
            if c % 100 == 0:
                print c

            tokens = tokenizer.tokenize(t)
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            # stem token
            text = [p_stemmer.stem(i) for i in stopped_tokens]
            bow = ldamodel.id2word.doc2bow(tokens)
            r = ldamodel[bow]
            # print r

            #Save txt per tweet TODO depending on net gt format

            #To make a fast test I can use one-hot (classification) wiht caffe
            # top_topic = 0
            # top_value = 0
            # for topic in r:
            #     if topic[1] > top_value:
            #         top_topic = topic[0]
            #         top_value = topic[1]
            #
            # split = randint(0,9)
            # if split < 8:
            #     train_file.write(id + ',' + str(top_topic) + '\n')
            # elif split == 8: val_file.write(id + ',' + str(top_topic) + '\n')
            # else: test_file.write(id + ',' + str(top_topic) + '\n')


            # GT for regression

            topic_probs = ''

            for t in range(0,num_topics):
                assigned = False
                for topic in r:
                        if topic[0] == t:
                            topic_probs = topic_probs + ',' + str(topic[1])
                            assigned = True
                            continue
                if not assigned:
                    topic_probs = topic_probs + ',' + '0'

            # print id + topic_probs

            split = randint(0,9)
            if split < 8:
                train_file.write(tweet_city + '/' + id + topic_probs + '\n')
            elif split == 8: val_file.write(tweet_city + '/' + id + topic_probs + '\n')
            else: test_file.write(tweet_city + '/' + id + topic_probs + '\n')




train_file.close()
val_file.close()
test_file.close()

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob

tweets_text_data_path = '../../../datasets/SocialMedia/weak_ann/test_random'
model_path = '../../../datasets/SocialMedia/models/lda_model_trump_weekend.model'
gt_path = '../../../datasets/SocialMedia/lda_gt/test_random/train.txt'

words2filter = ['rt','http','t','trump','gt','co','s','https','http','tweet','markars_','photo','donald','pictur','picture','say','dondald','photo','much','tweet']
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)


ldamodel = models.ldamodel.LdaModel.load(model_path)
tokenizer = RegexpTokenizer(r'\w+')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

topics = ldamodel.print_topics(num_topics=8, num_words=20)
print topics

file = open('topics.txt', 'w')

for item in topics:
    file.write("%s\n" % item[1])

c= 0

with open(gt_path, "w") as text_file:

    for file in glob.glob(tweets_text_data_path + "/*.txt"):
        with open(file, 'r') as fin:
            lines = fin.readlines()
            id = lines[0][:-1]
            t = lines[-1][:-1].decode('utf-8').lower()

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

            #Save txt per tweet TODO depending on net gt format

            #To make a fast test I can use one-hot (classification) wiht caffe
            top_topic = 0
            top_value = 0
            for topic in r:
                if topic[1] > top_value:
                    top_topic = topic[0]
                    top_value = topic[1]

            text_file.write(id + ',' + str(top_topic) + '\n')

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

tweets_text_data_path = '../../datasets/SocialMedia/text/text_trump_test.txt'
model_path = '../../datasets/SocialMedia/models/lda_model_trump_test.model'
gt_path = '../../datasets/SocialMedia/gt/train.txt'


ldamodel = models.ldamodel.LdaModel.load(model_path)
tokenizer = RegexpTokenizer(r'\w+')

print(ldamodel.print_topics(num_topics=3, num_words=10))

tweets_text_file = open(tweets_text_data_path, "r")
c= 0

with open(gt_path, "w") as text_file:

    for line in tweets_text_file:

        c += 1
        if c % 100 == 0:
            print c

        info = line.split(',')
        id = info[0]
        t = info[1].decode('utf-8').lower()
        tokens = tokenizer.tokenize(t)
        bow = ldamodel.id2word.doc2bow(tokens)
        r = ldamodel[bow]

        #Save txt per tweet TODO depending on net gt format
        #To make a fast test I can use one-hot (classification) wiht caffe
        topic = max(r)
        text_file.write(id + ',' + topic)

tweets_text_file.close()

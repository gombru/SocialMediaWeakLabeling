# Load trained LDA model and infer topics for unseen text.
# Make the train/val/test splits for CNN training

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import glob
from random import randint

# It also creates the splits train/val/test randomly

tweets_text_data_path = '../../../datasets/SocialMedia/weak_ann/trumpUnique'
model_path = '../../../datasets/SocialMedia/models/lda_model_trump_weekend.model'

gt_path_train = '../../../datasets/SocialMedia/lda_gt/trumpUnique/train.txt'
gt_path_val = '../../../datasets/SocialMedia/lda_gt/trumpUnique/val.txt'
gt_path_test = '../../../datasets/SocialMedia/lda_gt/trumpUnique/test.txt'


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

train_file = open(gt_path_train, "w")
val_file = open(gt_path_val, "w")
test_file = open(gt_path_test, "w")


for file in glob.glob(tweets_text_data_path + "/*.txt"):

    # Assign split

    with open(file, 'r') as fin:
        lines = fin.readlines()
        id = lines[0][:-1]

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

        #Save txt per tweet TODO depending on net gt format

        #To make a fast test I can use one-hot (classification) wiht caffe
        top_topic = 0
        top_value = 0
        for topic in r:
            if topic[1] > top_value:
                top_topic = topic[0]
                top_value = topic[1]

        split = randint(0,9)
        if split < 8:
            train_file.write(id + ',' + str(top_topic) + '\n')
        elif split == 8: val_file.write(id + ',' + str(top_topic) + '\n')
        else: test_file.write(id + ',' + str(top_topic) + '\n')

train_file.close()
val_file.close()
test_file.close()

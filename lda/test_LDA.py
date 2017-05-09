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

# Load data and model
text_data_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/'
model_path = '../../../datasets/SocialMedia/models/LDA/lda_model_cities_instagram_1M_200_chunck.model'

# Create output files
gt_path_train = '../../../datasets/SocialMedia/lda_gt/cities_instagram/trainCitiesInstagram_1M_200_chunck.txt'
gt_path_val = '../../../datasets/SocialMedia/lda_gt/cities_instagram/valCitiesInstagram_1M_200_chunck.txt'
gt_path_test = '../../../datasets/SocialMedia/lda_gt/cities_instagram/testCitiesInstagram_1M_200_chunck.txt'
train_file = open(gt_path_train, "w")
val_file = open(gt_path_val, "w")
test_file = open(gt_path_test, "w")

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

num_topics = 200
threads = 6

num_images_per_city = 100000
num_val = num_images_per_city * 0.05
num_test = num_images_per_city *0.15

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

topics = ldamodel.print_topics(num_topics=num_topics, num_words=20)

print topics

# Save a txt with the topics and the weights
file = open('topics.txt', 'w')
i = 0
for item in topics:
    file.write(str(i) + " - ")
    file.write("%s\n" % item[1])
    i+=1
file.close()


def infer_LDA(file_name):

    id = file_name.split('/')[-1][:-4]

    with open(file_name, 'r') as file:

        caption = ""
        filtered_caption = ""

        for line in file:
            caption = caption + line

        # Replace hashtags with spaces
        capion = caption.replace('#',' ')

        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        filtered_caption = filtered_caption.lower()

        tokens = tokenizer.tokenize(filtered_caption)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token

        # Handle stemmer error
        while "aed" in stopped_tokens:
            stopped_tokens.remove("aed")
            print "aed error"

        try:
            text = [p_stemmer.stem(i) for i in stopped_tokens]
            bow = ldamodel.id2word.doc2bow(text)
            r = ldamodel[bow]
            # print r
        except:
            print "Tokenizer error"
            print stopped_tokens
            return


        # GT for classification

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

        # Add zeros to topics without score
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
        return city + '/' + id + topic_probs


for city in cities:
        print city
        count = 0
        parallelizer = Parallel(n_jobs=threads)
        tasks_iterator = (delayed(infer_LDA)(file_name) for file_name in glob.glob(text_data_path + city + "/*.txt"))
        r = parallelizer(tasks_iterator)
        # merging the output of the jobs
        strings = np.vstack(r)

        for s in strings:
        #     # Create splits random
        #     try:
        #         split = randint(0,9)
        #         if split < 8:
        #             train_file.write(s[0] + '\n')
        #         elif split == 8: val_file.write(s[0] + '\n')
        #         else: test_file.write(s[0] + '\n')
        #     except:
        #         print "Error writing to file: "
        #         print s[0]
        #         continue

            # Create splits same number of images per class in each split
            try:
                if count < num_test:
                    test_file.write(s[0] + '\n')
                elif count < num_test + num_val:
                    val_file.write(s[0] + '\n')
                else:
                    train_file.write(s[0] + '\n')
                count += 1
            except:
                print "Error writing to file: "
                print s[0]
                continue


train_file.close()
val_file.close()
test_file.close()

print "Done"

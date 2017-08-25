import itertools
from gensim.models.word2vec import Text8Corpus
from stop_words import get_stop_words
from glove import Corpus, Glove
import string
import glob
import gensim
import json
import collections

whitelist = string.ascii_letters + string.digits + ' '
instagram_text_data_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/'
webvision_text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/SocialMedia/models/glove/glove_model_InstaCities1M.model'
words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']
en_stop = get_stop_words('en')

dim = 400
threads = 8
epochs = 30
lr = 0.05

def read_corpus(filename):

    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)

    with open(filename, 'r') as datafile:
        for line in datafile:
            yield line.lower().translate(None, delchars).split(' ')


def get_instacities1m():
    # -- LOAD DATA FROM INSTAGRAM --
    posts_text = []
    for city in cities:
        print("Loading InstaCities1M data from " + city)
        for i, file_name in enumerate(glob.glob(instagram_text_data_path + city + "/*.txt")):
            caption = ""
            filtered_caption = ""
            file = open(file_name, "r")
            for line in file:
                caption = caption + line
            # Replace hashtags with spaces
            caption = caption.replace('#', ' ')
            # Keep only letters and numbers
            for char in caption:
                if char in whitelist:
                    filtered_caption += char

            posts_text.append(filtered_caption.lower()) #.decode('utf-8')

    return posts_text


def get_webvision():
    # -- LOAD DATA FROM WEBVISION --
    posts_text = []
    former_filename = ' '
    print("Loading WebVision data")
    file = open(webvision_text_data_path + 'info/train_meta_list_all.txt', "r")

    for line in file:

        filename = line.split(' ')[0]
        filename = filename.replace('google', 'google_json')
        filename = filename.replace('flickr', 'flickr_json')
        idx = int(line.split(' ')[1])

        if filename != former_filename:
            print(filename)
            json_data = open(webvision_text_data_path + filename)
            d = json.load(json_data)
            former_filename = filename

        caption = ''
        filtered_caption = ''

        if 'description' in d[idx - 1]: caption = caption + d[idx - 1]['description'] + ' '
        if 'title' in d[idx - 1]: caption = caption + d[idx - 1]['title'] + ' '
        if 'tags' in d[idx - 1]:
            for tag in d[idx - 1]['tags']:
                caption = caption + tag + ' '

        # Replace hashtags with spaces
        caption = caption.replace('#', ' ')
        # Keep only letters and numbers
        for char in caption:
            if char in whitelist:
                filtered_caption += char

        posts_text.append(filtered_caption.decode('utf-8').lower())

    return posts_text

print('Creating Corpus')
#sentences = read_corpus(filename)
posts_text = get_instacities1m()
c=0
sentences = []
for t in posts_text:
    c += 1
    if c % 10000 == 0:
        print(c)
    try:
        t = t.lower()
        #Gensim simple_preproces instead tokenizer
        tokens = gensim.utils.simple_preprocess(t)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token
        # text = [p_stemmer.stem(i) for i in stopped_tokens]
        # add proceced text to list of lists
        sentences.append(stopped_tokens)
    except:
        continue

posts_text = []

corpus = Corpus()
corpus.fit(sentences, window=10)

print('Dict size: %s' % len(corpus.dictionary))
print('Collocations: %s' % corpus.matrix.nnz)

print('Training the GloVe model')
glove = Glove(no_components=dim, learning_rate=lr)
glove.fit(corpus.matrix, epochs=epochs, no_threads=threads, verbose=True)
glove.add_dictionary(corpus.dictionary)


glove.save(model_path)

# Default Pickle fails with large models, so I go with cPickle
# with open(model_path, 'wb') as savefile:
#     cPickle.dump(glove.__dict__,
#                 savefile) #protocol=cPickle.HIGHEST_PROTOCOL

print('Model Saved')

#
# model = Glove.load(model_path)
#

print(glove.most_similar('man'))
print(glove.most_similar('biology'))
print(glove.most_similar('dog'))

# Get a word embedding
print(len(glove.word_vectors[corpus.dictionary['london']]))

# Get a paragraph embedding
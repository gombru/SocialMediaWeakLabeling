from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import gensim
import string
import glob
import multiprocessing
import json

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"


whitelist = string.letters + string.digits + ' '
instagram_text_data_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/'
webvision_text_data_path = '../../../datasets/WebVision/'
model_path = '../../../datasets/SocialMedia/models/doc2vec/doc2vec_model_webvision.model'
words2filter = ['rt','http','t','gt','co','s','https','http','tweet','markars_','photo','pictur','picture','say','photo','much','tweet','now','blog','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube','wikipedia','google', 'flickr', 'figure', 'photo', 'image', 'homepage', 'url', 'youtube', 'images', 'blog', 'pinterest']

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

size = 400 # vector size
min_count = 25 # minimum word count to 2 in order to give higher frequency words more weighting
iter = 10 # iterating over the training corpus x times
window = 8

#Initialize Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)
# Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()
texts = [] #List of lists of tokens


def get_instacities1m():
    # -- LOAD DATA FROM INSTAGRAM --
    posts_text = []
    for city in cities:
        print "Loading InstaCities1M data from " + city
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

            posts_text.append(filtered_caption.decode('utf-8').lower())

    return posts_text


def get_webvision():
    # -- LOAD DATA FROM WEBVISION --
    former_filename = ' '
    print "Loading WebVision data"
    file = open(webvision_text_data_path + 'info/train_meta_list_all.txt', "r")

    for line in file:

        filename = line.split(' ')[0]
        filename.replace('google', 'google_json')
        filename.replace('flickr', 'flickr_json')
        idx = int(line.split(' ')[1])

        if filename != former_filename:
            print filename
            json_data = open(webvision_text_data_path + filename)
            d = json.load(json_data)
            former_filename = filename

        caption = ''
        filtered_caption = ''

        if d[idx - 1].has_key('description'): caption = caption + d[idx - 1]['description'] + ' '
        if d[idx - 1].has_key('title'): caption = caption + d[idx - 1]['title'] + ' '
        if d[idx - 1].has_key('tags'):
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

posts_text = get_webvision()

print "Number of posts: " + str(len(posts_text))

print "Creating tokens"
c= 0

for t in posts_text:

    c += 1
    if c % 10000 == 0:
        print c

    try:
        t = t.lower()
        #Gensim simple_preproces instead tokenizer
        tokens = gensim.utils.simple_preprocess(t)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem token
        # text = [p_stemmer.stem(i) for i in stopped_tokens]
        # add proceced text to list of lists
        texts.append(stopped_tokens)
    except:
        continue

posts_text = []


#For training data, add tags
for i in range(0,len(texts)):
    texts[i] = gensim.models.doc2vec.TaggedDocument(texts[i], [i])

#Train the model
print "Training ..."
model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=min_count, iter=iter, window=window, workers=cores)
model.build_vocab(texts)
model.train(texts, total_examples=model.corpus_count, epochs=model.iter) # use BLAS if you value your time
print "Training DONE"
model.save(model_path)


# Print similar docs for debugging
def print_similar(doc_id):
    inferred_vector = model.infer_vector(texts[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    print('\nDocument ({}): --{}--\n'.format(doc_id, ' '.join(texts[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    # for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #     print(u'%s %s: -%s-\n' % (label, sims[index], ' '.join(texts[sims[index][0]].words)))

    for label, index in [('MOST2', 1), ('MOST3', 2), ('MOST4', 3)]:
        print(u'%s %s: -%s-\n' % (label, sims[index], ' '.join(texts[sims[index][0]].words)))

docs_id = [1,10000,3000,10000]
for doc_id in docs_id: print_similar(doc_id)
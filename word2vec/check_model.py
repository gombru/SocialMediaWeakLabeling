from gensim import models

model_name = 'word2vec_model_InstaCities1M.model'
model_path = '../../../datasets/SocialMedia/models/word2vec/' + model_name

print "Loading model ... "
model = models.FastText.load(model_path)

print model.wv.most_similar(positive=['amusement'])
print model.wv.most_similar(positive=['anger'])
print model.wv.most_similar(positive=['awe'])
print model.wv.most_similar(positive=['contentment'])
print model.wv.most_similar(positive=['disgusting'])
print model.wv.most_similar(positive=['exiting'])
print model.wv.most_similar(positive=['fear'])
print model.wv.most_similar(positive=['sadness'])



print model.wv.most_similar(positive=['woman', 'king'], negative=['man'])

print model.wv.most_similar(positive=['beach', 'sea'], negative=['mountain'])

print model.wv.doesnt_match("breakfast cereal dinner lunch".split())

print model.wv.doesnt_match("man woman kid dog".split())


emotions = ['amusement','anger','awe','contentment','disgusting','exiting','fear','sadness']

for m in emotions:
    print m
    print model.wv.similarity(m, 'baby')
    print model.wv.similarity(m, 'beach')
    print model.wv.similarity(m, 'winter')
    print model.wv.similarity(m, 'night')
    print model.wv.similarity(m, 'terrorism')
    print model.wv.similarity(m, 'icecream')
    print model.wv.similarity(m, 'sun')
    print model.wv.similarity(m, 'laugh')
    print model.wv.similarity(m, 'dead')
    print model.wv.similarity(m, 'tears')
    print model.wv.similarity(m, 'dark')




print 'DONE'
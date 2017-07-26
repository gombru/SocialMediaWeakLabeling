from gensim import models

model_name = 'word2vec_model_webvision.model'
model_path = '../../../datasets/WebVision/models/word2vec/' + model_name

print "Loading model ... "
model = models.Word2Vec.load(model_path)

print model.wv.most_similar(positive=['woman', 'king'], negative=['man'])

print model.wv.most_similar(positive=['beach', 'sea'], negative=['mountain'])

print model.wv.doesnt_match("breakfast cereal dinner lunch".split())

print model.wv.doesnt_match("man woman kid dog".split())

print model.wv.similarity('woman', 'man')

print 'DONE'
from gensim import models
import re

model_name = 'doc2vec_model_instacities1M.model'
model_path = '../../../datasets/SocialMedia/models/doc2vec/' + model_name

print "Loading model ... "
model = models.Doc2Vec.load(model_path)

print "Vocabulary size: " + str(len(model.wv))

print model.wv.most_similar(positive=['woman', 'king'], negative=['man'])

print model.wv.most_similar(positive=['sailing'])

print model.wv.most_similar(positive=['pizza'])

print model.wv.most_similar(positive=['bear'])


print model.wv.most_similar(positive=['beach', 'sea'], negative=['mountain'])

print model.wv.doesnt_match("breakfast cereal dinner lunch".split())

print model.wv.doesnt_match("man woman kid dog".split())

print model.wv.similarity('woman', 'man')

print 'DONE'
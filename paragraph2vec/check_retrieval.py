from stop_words import get_stop_words
import gensim
import numpy as np
import operator


# Load data and model
model_path = '../../../datasets/SocialMedia/models/doc2vec/doc2vec_model_instacities1M.model'
model = gensim.models.Doc2Vec.load(model_path)
en_stop = get_stop_words('en')

captions = ['a man in a boat sailing in the Mediterranean sea','a family having a pizza and wine for dinner', 'a bear and a lion in a zoo']
database = {}
database_words = {}

for n,caption in enumerate(captions):
    tokens = gensim.utils.simple_preprocess(caption)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    tokens_filtered = [token for token in stopped_tokens if token in model.wv.vocab]
    print tokens_filtered
    model.random.seed(0)
    embedding = model.infer_vector(tokens_filtered, alpha=0.025, min_alpha=0.0001, steps=20)
    database[n] = embedding
    database_words[n] = tokens_filtered

queries = ['i like the ocean and i love to sail', 'we ate pasta with tomato and cheese','i fed the dog and the cat']

for q in queries:
    # distances1 = {}
    distances2 = {}
    distances3 = {}


    print "Query: " +  q + " ---> \n"
    tokens = gensim.utils.simple_preprocess(q)
    tokens_filtered = [token for token in tokens if token in model.wv.vocab]

    q_embedding = model.infer_vector(tokens)

    for id, caption in enumerate(database):
        # distances1[id] = np.dot(database[id], q_embedding)
        model.random.seed(0)
        # distances2[id] = np.dot((database[id] - min(database[id])) / sum(database[id] - min(database[id])) , (q_embedding - min(q_embedding)) / sum(q_embedding - min(q_embedding)))
        model.random.seed(0)
        distances3[id] = model.docvecs.similarity_unseen_docs(model, database_words[id], tokens_filtered, steps=100, alpha=0.025)

    # # Sort dictionary
    # distances1 = sorted(distances1.items(), key=operator.itemgetter(1), reverse=True)
    # distances2 = sorted(distances2.items(), key=operator.itemgetter(1), reverse=True)
    distances3 = sorted(distances3.items(), key=operator.itemgetter(1), reverse=True)



    # Get elements with min distances
    # for idx, id in enumerate(distances1):
    #     print captions[id[0]]
    # print '__________________'
    # for idx, id in enumerate(distances2):
    #     print captions[id[0]]
    # print '__________________'
    for idx, id in enumerate(distances3):
        print id
        print captions[id[0]]
    print '__________________ \n'




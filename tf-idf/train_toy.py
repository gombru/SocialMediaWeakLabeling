import gensim

docs = [['sea','ocean','ocean'],['a','a','a'],['ocean','ocean','ocean', 'beach']]

dictionary = gensim.corpora.Dictionary(docs)
dictionary.save('/tmp/docs.dict')
raw_corpus = [dictionary.doc2bow(t) for t in docs]
gensim.corpora.MmCorpus.serialize('/tmp/docs.mm', raw_corpus)
dictionary = gensim.corpora.Dictionary.load('/tmp/docs.dict')
corpus = gensim.corpora.MmCorpus('/tmp/docs.mm')
tfidf = gensim.models.TfidfModel(corpus)


doc = "beach sea ocean oce2an pepe"
dictionary = gensim.corpora.Dictionary.load('/tmp/docs.dict')
vec = dictionary.doc2bow(doc.lower().split())
print (vec)
vec_tfidf=tfidf[vec]
print vec_tfidf

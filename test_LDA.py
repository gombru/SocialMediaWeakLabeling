from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

model_path = '../../datasets/SocialMedia/models/lda_model_trump_test.model'

ldamodel = models.ldamodel.LdaModel.load(model_path)

print(ldamodel.print_topics(num_topics=3, num_words=10))



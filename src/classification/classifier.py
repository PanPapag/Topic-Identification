import gensim
import numpy as np

from classification.mean_embedding_vectorizer import *

from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold


class Classifier:

    def __init__(self, path, train_df, test_df, feature):

        self.path = path
        self.feature = feature
        self.steps = []
        # Encode labels with value the different articles' categories
        self.le = preprocessing.LabelEncoder()
        self.le.fit(train_df['Category'])
        # define x_train and y_train
        self.x_train = train_df['Content']
        self.y_train = self.le.transform(train_df['Category'])
        # define x_test
        self.x_test = test_df['Content'] if not test_df is None else None
        self.test_ids = test_df['Id'] if not test_df is None else None


    def define_features(self):

        if self.feature == "W2V":
             # let X be a list of tokenized texts (i.e. list of lists of tokens)
             X =
             # train a Word2Vec model from scratch with gensim
             model = gensim.models.Word2Vec(X, size=100)
             w2v = dict(zip(model.wv.index2word, model.wv.syn0))
             self.steps.append(('w2v', MeanEmbeddingVectorizer(w2v)))
        elif self.feature == "TF-IDF":
            self.steps.append(('tf-idf', TfidfTransformer(stop_words='english')))
        else:
            self.steps.append(('vect', CountVectorizer(stop_words='english')))

        # perform Lantent-Semantic-Indexing (LSI)
        svd = TruncatedSVD(n_components=5000)
        self.steps.append(('svd', svd))

        return self.steps


    def predict(self, pipeline):
        print(self.steps)
        return None


    def predict_kfold(self, pipeline):
        print(self.steps)
        return 1

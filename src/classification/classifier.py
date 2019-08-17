import gensim
import numpy as np

from classification.mean_embedding_vectorizer import *
from data_preprocessing.preprocessor import *

from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
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
             X = Preprocessor().tokenize_doc(self.x_train)
             # train a Word2Vec model from scratch with gensim
             model = gensim.models.Word2Vec(X, size=100)
             w2v = dict(zip(model.wv.index2word, model.wv.syn0))
             self.steps.append(('w2v', MeanEmbeddingVectorizer(w2v)))
        elif self.feature == "TF-IDF":
            self.steps.append(('tf-idf', TfidfTransformer()))
        else:
            self.steps.append(('vect', CountVectorizer(stop_words='english')))

        # perform Lantent-Semantic-Indexing (LSI)
        svd = TruncatedSVD(n_components=5000)
        self.steps.append(('svd', svd))

        return self.steps


    def predict(self, pipeline):

        # fit model
        pipeline.fit(self.x_train, self.y_train)
        # Output a pickle file for the model
        joblib.dump(pipeline, dump_path)
        # make predictions and export them to csv file
        predicted = pipeline.predict(self.x_test)
        y_pred = self.le.inverse_transform(predicted)
        self.export_to_csv(y_pred)

        return None


    def predict_kfold(self, pipeline):

        score_array = []
        accuracy_array = []
        # apply 10-fold cross validation
        kf = KFold(n_splits=10)

        for train_index, test_index in kf.split(self.x_train):
            # use iloc which is label based indexing
            cv_train_x = self.x_train.iloc[train_index]
            cv_test_x = self.x_train.iloc[test_index]
            cv_train_y = self.y_train[train_index]
            cv_test_y = self.y_train[test_index]
            # fit model
            pipeline.fit(cv_train_x, cv_train_y)
            # predict label in form of numbers {0,1,2,3,4}
            predicted = pipeline.predict(cv_test_x)
            # apply inverse transform to get labels
            y_labels = self.le.inverse_transform(cv_test_y)
            y_pred = self.le.inverse_transform(predicted)
            # append score and accuracy values to corresponding arrays
            score_array.append(precision_recall_fscore_support(y_labels, y_pred, average=None))
            accuracy_array.append(accuracy_score(y_labels, y_pred))

        # compute classification report metrics
        avg_accuracy = round(np.mean(accuracy_array), 4)
        avg_scores = np.mean(np.mean(score_array, axis=0), axis=1)
        avg_precision = round(avg_scores[0], 4)
        avg_recall = round(avg_scores[1], 4)
        avg_f1 = round(avg_scores[2], 4)

        return (avg_accuracy, avg_precision, avg_recall, avg_f1)


    def export_to_csv(self, predicted_values):

        with open(self.path + 'testSet_categories.csv', 'w') as f:
            sep = '\t'
            f.write('Test_Document_ID')
            f.write(sep)
            f.write('Predicted Category')
            f.write('\n')

            for Id, predicted_value in zip(self.test_ids, predicted_values):
                f.write(str(Id))
                f.write(sep)
                f.write(predicted_value)
                f.write('\n')

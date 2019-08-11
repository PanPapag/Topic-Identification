import gensim
import numpy as np

from sklearn import preprocessing


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
        return self.steps


    def predict(self, pipeline):
        print(self.steps)
        return None


    def predict_kfold(self, pipeline):
        print(self.steps)
        return 1

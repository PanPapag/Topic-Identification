import gensim
import numpy as np

from sklearn import preprocessing


class Classifier:

    def __init__(self, path, train_df, test_df, feature):
        # pass info from arguments
        self.path = path
        self.feature = feature
        # Encode labels with value the different articles' categories
        self.le = preprocessing.LabelEncoder()
        self.le.fit(train_df['Category'])
        # define x_train and y_train
        self.x_train = train_df['Content']
        self.y_train = self.le.transform(train_df['Category'])
        # define x_test
        self.x_test = test_df['Content']
        self.test_ids = test_df['Id']

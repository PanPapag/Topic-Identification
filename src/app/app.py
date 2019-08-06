import pandas as pd
import os

class App:

    def __init__(self, datasets, outputs, threshold=None, preprocess=False,
                 wordcloud=False, classification=None, features=None, kfold=False, cache=False):

        # pass info from arguments
        self.datasets = datasets
        self.outputs = outputs
        self.threshold = threshold
        self.preprocess = preprocess
        self.wordcloud = wordcloud
        self.classification = classification
        self.features = features
        self.kfold = kfold
        self.cache = cache

        # define csv train and test file names
        if not self.cache:
            self.csv_train_file = "/".join([datasets,'train_set.csv'])
            self.csv_test_file = "/".join([datasets,'test_set.csv'])
        else:
            self.csv_train_file = "/".join([datasets,'processed_train_set.csv'])
            self.csv_test_file = "/".join([datasets,'processed_test_set.csv'])

        # read csv train and test files using pandas
        self.train_df = pd.read_csv(self.csv_train_file, sep='\t')
        self.test_df = pd.read_csv(self.csv_test_file, sep='\t') if not self.kfold else None

        # define output directory names
        self.wordcloud_out_dir = "/".join([outputs,'wordcloud_out_dir/']) if self.wordcloud else None
        self.duplicates_out_dir = "/".join([outputs,'duplicates_out_dir/']) if self.threshold else None
        self.classification_out_dir = "/".join([outputs,'classification_out_dir/']) if self.classification else None

        # create output directories (if not already exist)
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)

        if self.wordcloud:
            if not os.path.exists(self.wordcloud_out_dir):
                os.makedirs(self.wordcloud_out_dir)

        if self.threshold:
            if not os.path.exists(self.duplicates_out_dir):
                os.makedirs(self.duplicates_out_dir)

        if self.classification:
            if not os.path.exists(self.classification_out_dir):
                os.makedirs(self.classification_out_dir)


    def run(self):
        print("App running")
        print(self.train_df)

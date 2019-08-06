import pandas as pd
import logging
import math
import os

from data_preprocessing.preprocessor import *

class App:

    def __init__(self, datasets, outputs, threshold=None, preprocess=None,
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

        # discard RowNum
        self.train_df.drop('RowNum', axis=1, inplace=True)
        self.test_df.drop('RowNum', axis=1, inplace=True)

        # get unique categories using training set
        self.categories = self.train_df['Category'].unique()

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


    def preprocess_data(self):

        print("Data preprocessing..")
        # create preprocessor object
        preprocessor = Preprocessor(self.train_df, self.preprocess)
        # define processed training set
        processed_train_set =  "/".join([self.datasets,'processed_train_set.csv'])
        # Title preprocessing

        # Content preprocessing
        
        print("Data preprocessing completed.")


    def run(self):

        print("App running..")

        # if data has not been preprocessed before and preprocess flag is True
        if not self.cache and self.preprocess:
            self.preprocess_data()

        print("App completed.")

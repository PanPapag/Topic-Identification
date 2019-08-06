import logging
import math
import pandas as pd
import os
import time

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

        data_start = time.time()
        print("Data preprocessing..")
        # create preprocessor object
        preprocessor = Preprocessor(self.train_df, self.preprocess)
        # define processed training set
        processed_train_set =  "/".join([self.datasets,'processed_train_set.csv'])
        train_start = time.time()
        print("\t Train set preprocesssing..")
        # Title preprocessing
        title_start = time.time()
        print("\t \t Title preprocesssing..")
        self.train_df = preprocessor.preprocess(col='Title')
        title_end = time.time()
        print("\t \t Title preprocesssing completed. Time elapsed: {:.3f} seconds"
              .format(title_end - title_start))
        # Content preprocessing
        content_start = time.time()
        print("\t \t Content preprocesssing..")
        self.train_df = preprocessor.preprocess(col='Content')
        content_end = time.time()
        print("\t \t Content preprocesssing completed. Time elapsed: {:.3f} seconds"
              .format(content_end - content_start))
        # Train set preprocessing completed
        train_end = time.time()
        print("\t Train set preprocesssing completed. Time elapsed: {:.3f} seconds"
              .format(train_end - train_start))
        # Data preprocessing completed
        data_end = time.time()
        print("Data preprocessing completed. Time elapsed: {:.3f} seconds\n"
              .format(data_end - data_start))
        # save processed train set to csv
        preprocessor.save_to_csv(self.train_df, processed_train_set)


    def run(self):

        print("App running..\n")

        # if data has not been preprocessed before and preprocess flag is True
        if not self.cache and self.preprocess:
            self.preprocess_data()

        print("App completed.")

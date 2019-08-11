import math
import pandas as pd
import os
import time

from classification.support_vector_machine import *

from data_preprocessing.preprocessor import *
from duplicates.duplicate import *
from word_cloud.wordcloud import *

class App:

    def __init__(self, datasets, outputs, dupl_threshold=None, preprocess=None,
                 wordcloud=False, classification=None, feature=None, kfold=False, cache=False):

        # pass info from arguments
        self.datasets = datasets
        self.outputs = outputs
        self.dupl_threshold = dupl_threshold
        self.preprocess = preprocess
        self.wordcloud = wordcloud
        self.classification = classification
        self.feature = feature
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

        # get unique categories using training set
        self.categories = self.train_df['Category'].unique()

        # define output directory names
        self.wordcloud_out_dir = "/".join([outputs,'wordcloud_out_dir/']) if self.wordcloud else None
        self.duplicates_out_dir = "/".join([outputs,'duplicates_out_dir/']) if self.dupl_threshold else None
        self.classification_out_dir = "/".join([outputs,'classification_out_dir/']) if self.classification else None

        # create output directories (if not already exist)
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)

        if self.wordcloud:
            if not os.path.exists(self.wordcloud_out_dir):
                os.makedirs(self.wordcloud_out_dir)

        if self.dupl_threshold:
            if not os.path.exists(self.duplicates_out_dir):
                os.makedirs(self.duplicates_out_dir)

        if self.classification:
            if not os.path.exists(self.classification_out_dir):
                os.makedirs(self.classification_out_dir)

    def clean_data(self):
        # discard column RowNum and delte rows with Nan values
        print("Cleaning training set..")
        print("\t Discarding column RowNum..")
        try:
            self.train_df.drop('RowNum', axis=1, inplace=True)
            print("\t Discarding column RowNum completed.")
        except KeyError:
            print("\t File {} has no column RowNum.".format(self.csv_train_file))
        print("\t Deleting rows with NaN values..")
        try:
            self.train_df.dropna(inplace=True)
            print("\t Deleting rows with NaN values completed.")
        except KeyError:
            print("\t File {} has rows with NaN values.".format(self.csv_train_file))
        print("Cleaning training set completed.\n")

        if not self.kfold:
            print("Cleaning test set..")
            print("\t Discarding column RowNum..")
            try:
                self.test_df.drop('RowNum', axis=1, inplace=True)
                print("\t Discarding column RowNum completed.")
            except KeyError:
                print("\t File {} has no column RowNum.".format(self.csv_test_file))
            print("\t Deleting rows with NaN values..")
            try:
                self.test_df.dropna(inplace=True)
                print("\t Deleting rows with NaN values completed.")
            except KeyError:
                print("\t File {} has rows with NaN values.".format(self.csv_test_file))
            print("Cleaning test set completed.\n")

    def preprocess_data(self):

        data_start = time.time()
        print("Data preprocessing..")
        # create preprocessor object for training set
        train_preprocessor = Preprocessor(self.train_df, self.preprocess)
        # define processed training set
        processed_train_set =  "/".join([self.datasets,'processed_train_set.csv'])
        train_start = time.time()
        print("\t Train set preprocesssing..")
        # Title preprocessing
        title_start = time.time()
        print("\t \t Title preprocesssing..")
        self.train_df = train_preprocessor.preprocess(col='Title')
        title_end = time.time()
        print("\t \t Title preprocesssing completed. Time elapsed: {:.3f} seconds"
              .format(title_end - title_start))
        # Content preprocessing
        content_start = time.time()
        print("\t \t Content preprocesssing..")
        self.train_df = train_preprocessor.preprocess(col='Content')
        content_end = time.time()
        print("\t \t Content preprocesssing completed. Time elapsed: {:.3f} seconds"
              .format(content_end - content_start))
        # Train set preprocessing completed
        train_end = time.time()
        print("\t Train set preprocesssing completed. Time elapsed: {:.3f} seconds"
              .format(train_end - train_start))
        # save processed train set to csv and define proc
        train_preprocessor.save_to_csv(self.train_df, processed_train_set)
        if not self.kfold:
            # create preprocessor object for test set
            test_preprocessor = Preprocessor(self.test_df, self.preprocess)
            # define processed training set
            processed_test_set =  "/".join([self.datasets,'processed_test_set.csv'])
            test_start = time.time()
            print("\t Test set preprocesssing..")
            # Title preprocessing
            title_start = time.time()
            print("\t \t Title preprocesssing..")
            self.test_df = test_preprocessor.preprocess(col='Title')
            title_end = time.time()
            print("\t \t Title preprocesssing completed. Time elapsed: {:.3f} seconds"
                  .format(title_end - title_start))
            # Content preprocessing
            content_start = time.time()
            print("\t \t Content preprocesssing..")
            self.test_df = test_preprocessor.preprocess(col='Content')
            content_end = time.time()
            print("\t \t Content preprocesssing completed. Time elapsed: {:.3f} seconds"
                  .format(content_end - content_start))
            # Test set preprocessing completed
            test_end = time.time()
            print("\t Test set preprocesssing completed. Time elapsed: {:.3f} seconds"
                  .format(test_end - test_start))
            # save processed test set to csv and define proc
            test_preprocessor.save_to_csv(self.test_df, processed_test_set)
        # Data preprocessing completed
        data_end = time.time()
        print("Data preprocessing completed. Time elapsed: {:.3f} seconds\n"
              .format(data_end - data_start))


    def generate_wordclouds(self):
        wc_start = time.time()
        print("Generating wordcloud per category..")
        # create WordCloud object and pass appropriate info
        wc = WordCloudGen(self.wordcloud_out_dir)
        # create a preprocessor object to handle processed training set
        filter = Preprocessor(self.train_df)
        # iterate over each label
        for label in self.categories:
            print("\t Generating wordcloud for category {}..".format(label))
            gen_start = time.time()
            text = filter.join_spec_rows_of_spec_column_value(label, ['Title','Content'], 'Category')
            wc.generate_wordcloud(label, text)
            gen_end = time.time()
            print("\t Wordcloud generating for category {} completed. Time elapsed: {:.3f} seconds"
                  .format(label, gen_end - gen_start))
        wc_end = time.time()
        print("Wordcloud generating completed. Time elapsed: {:.3f} seconds\n"
              .format(wc_end - wc_start))


    def find_similar_articles(self):

        start = time.time()
        print("Finding similar articles..")
        # create duplicate object
        dup = Duplicate(self.duplicates_out_dir, self.train_df, self.dupl_threshold, self.categories)
        dup.detect_duplicates()
        end = time.time()
        print("Finding similar articles completed. Time elapsed: {:.3f} seconds\n"
              .format(end - start))


    def classify(self):

        start = time.time()
        # define Bag of Words as default feature
        self.feature = "BoW" if self.feature is None else self.feature
        print("Running {} classifier with the selected feature {}..".format(self.classification, self.feature))
        # determine the classifier that gonna be used
        '''
        if self.classification == "NB":
            clf = NaiveBayes
        elif self.classification == 'RF':
            clf = RandomForest
        elif self.classification == 'SVM':
            clf = SupportVectorMachine
        elif self.classification == "KNN":
            clf = KNN '''
        if self.classification == 'SVM':
            clf = SupportVectorMachine

        classifier = clf(self.classification_out_dir, self.train_df, self.test_df, self.feature)
        score = classifier.run_kfold() if self.kfold else classifier.run_predict()

        end = time.time()

        print(score) if self.kfold else None
        print("Running {} classifier with the selected feature {} completed. Time elapsed: {:.3f} seconds\n"
              .format(self.classification, self.feature, end - start))


    def run(self):

        print("App running..\n")

        self.clean_data()

        if not self.cache and self.preprocess:
            self.preprocess_data()

        if self.wordcloud:
            self.generate_wordclouds()

        if self.dupl_threshold:
            self.find_similar_articles()

        if self.classification:
            self.classify()

        print("App completed.")

import logging
import numpy as np
import pandas as pd
import re
import spacy

from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class Preprocessor:

    def __init__(self, input_df=None, transformation=None):
        # pass info from arguments
        self.input_df = input_df
        self.transformation = transformation
        # define the corresponding filter given the preprocess transformation
        if self.transformation == "LEM":
            self.filter = WordNetLemmatizer()
        elif self.transformation == "STEM":
            self.filter = LancasterStemmer()


    def preprocess(self, col=None):

        # Remove rows with missing values in column col
        self.input_df = self.input_df[pd.notnull(self.input_df[col])]
        # Speed up code using numpy vectorization
        vfunc = np.vectorize(self.text_normalization)
        # define text to be normalized
        if col == 'Title':
            self.input_df.Title = vfunc(self.input_df.Title.values)
        elif col == 'Content':
            self.input_df.Content = vfunc(self.input_df.Content.values)
        # return processed input_df
        return self.input_df

    def text_normalization(self, text):
        return text.lower()

    def test(self):
        print(self.input_df.Title)
        #print(self.input_df.Content)

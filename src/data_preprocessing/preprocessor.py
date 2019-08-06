import logging
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


    def test(self):
        print(self.input_df)
        print(self.transformation)
        print(self.filter)

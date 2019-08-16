import logging
import numpy as np
import pandas as pd
import re
import string
import spacy

import nltk
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

class Preprocessor:

    def __init__(self, input_df=None, transformation=None):
        # pass info from arguments
        self.input_df = input_df
        self.transformation = transformation

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

        # convert text to lowercase
        text = text.lower()
        # remove all special characters, punctuation and spaces from string
        text = re.sub(r'\W+',' ', text)
        # remove stop words
        removed = remove_stopwords(text)
        text = "".join(removed)
        # define transformation
        if self.transformation == "STEM":
            stemmer = LancasterStemmer()
            stem_sentence = []
            token_words = word_tokenize(text)
            stem_sentence = [stemmer.stem(word) for word in token_words]
            text = " ".join(stem_sentence)
        elif self.transformation == "LEM":
            lemmatizer = WordNetLemmatizer()
            lem_sentence = []
            token_words = word_tokenize(text)
            lem_sentence = [lemmatizer.lemmatize(word) for word in token_words]
            text = " ".join(lem_sentence)
        # return normalized text
        return text

    def join_spec_rows_of_spec_column_value(self, label, cols, spec_col):
        acum = []
        for col in cols:
            acum.append(' '.join(self.input_df.loc[self.input_df[spec_col] == label][col].values))
        return ' '.join(acum)

    def tokenize_df(self, df):
        tokenized_df = df.apply(nltk.word_tokenize)
        return tokenized_df

    def save_to_csv(self, df, path):
        df.to_csv(path_or_buf=path, index=False, sep='\t')

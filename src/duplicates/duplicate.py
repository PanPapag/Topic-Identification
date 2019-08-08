import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

class Duplicate:

    def __init__(self, path=None, df=None, threshold=None, categories=None):
        # pass info from arguments
        self.path = path
        self.df = df
        self.threshold = threshold
        self.categories = categories


    def test(self):
        print(self.path)
        print(self.threshold)
        print(self.categories)

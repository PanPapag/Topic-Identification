import pandas as pd
import os

class App:

    def __init__(self, datasets, outputs, threshold=None, preprocess=False,
                 wordcloud=False, classification=None, features=None, kfold=False, cache=False):

        self.datasets = datasets
        self.outputs = outputs
        self.threshold = threshold
        self.preprocess = preprocess
        self.wordcloud = wordcloud
        self.classification = classification
        self.features = features
        self.kfold = kfold
        self.cache = cache

    def run(self):
        print("App running")

import pandas as pd
import time
from wordcloud import WordCloud

from data_preprocessing.preprocessor import *

class WordCloud:

    def __init__(self, input_df, categories, path):
        # pass info from arguments
        self.df = pd.read_csv(input_df, sep='\t')
        self.categories = categories
        self.path = path
        self.wordcloud_preprocessor = Preprocessor(self.df, self.categories)

    def generate_wordclouds(self):
        content_per_cat = self.wordcloud_preprocessor.text_per_category()
        for label in self.categories:
            #text = content_per_cat[label]
            print("\t Generating wordcloud for category {}..".format(label))
            start = time.time()

            end = time.time()
            print("\t Wordcloud generating for category {} completed. Time elapsed: {:.3f} seconds"
                  .format(label, end - start))
            #wordcloud.to_file(self.path + label + '_wordcloud.png')

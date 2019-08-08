import pandas as pd
import time

from wordcloud import WordCloud

from data_preprocessing.preprocessor import *


class WordCloudGen:

    def __init__(self, path):
        self.path = path

    def generate_wordcloud(self, label, text):
        wordcloud = WordCloud(max_words=1000, max_font_size=40, margin=10,
                              random_state=1, width=840, height=420).generate(text)
        wordcloud.to_file(self.path + label + '_wordcloud.png')

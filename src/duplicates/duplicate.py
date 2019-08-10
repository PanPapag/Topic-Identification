import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

class Duplicate:

    def __init__(self, path=None, df=None, threshold=None, categories=None):
        # pass info from arguments
        self.path = path
        self.df = df
        self.threshold = threshold
        self.categories = categories


    @staticmethod
    def cosine_similarity(X, Y):
        # compute cosine similarity
        dot = np.dot(X, Y)
        norma = np.linalg.norm(X)
        normb = np.linalg.norm(Y)
        cos = dot / (norma * normb)
        return cos


    def export_to_csv(self, docs_similarity):

        with open(self.path + "similar_docs.csv", "w") as f:
            sep = '\t'
            f.write('Document_ID1')
            f.write(sep)
            f.write('Document_ID2')
            f.write(sep)
            f.write('Cosine Similarity')
            f.write('\n')

            for id1, id2 in docs_similarity:
                f.write(str(id1))
                f.write(sep)
                f.write(str(id2))
                f.write(sep)
                f.write(str(docs_similarity[(id1,id2)]))
                f.write('\n')


    def detect_duplicates(self):
        # define dict: (id1,id2) --> similarity
        docs_similarity = {}
        # for each category check all pairs of articles
        for label in self.categories:
            # get corpus for current category
            corpus = self.df.loc[self.df['Category'] == label]['Content'].values
            # get articles' ids
            corpus_ids = self.df.loc[self.df['Category'] == label]['Id'].values
            print(corpus_ids[0])
            # Use TF-IDF word embedding
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(corpus).toarray()

            for idx_i, i in enumerate(X):
                # if vectorized document i has at least one non zero value
                if i.any(axis=0) :
                    for idx_j in range(idx_i+1, len(X)):
                        j = X[idx_j]
                        # if vectorized document j has at least one non zero value
                        if j.any(axis=0):
                            similarity = self.cosine_similarity(i,j)
                            if similarity >= self.threshold:
                                # get documents' ids
                                id1 = corpus_ids[idx_i]
                                id2 = corpus_ids[idx_j]
                                # store similarity of documents
                                similarities[(id1,id2)] = similarity

        # write results to csv file
        self.export_to_csv(docs_similarity)

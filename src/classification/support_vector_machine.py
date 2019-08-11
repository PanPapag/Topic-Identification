from classification.classifier import Classifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_df, test_df, feature):
		Classifier.__init__(self, path, train_df, test_df, feature)

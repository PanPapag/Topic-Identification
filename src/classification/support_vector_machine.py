from classification.classifier import Classifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class SupportVectorMachines(Classifier):

	def __init__(self, path, train_df, test_df, feature):
		Classifier.__init__(self, path, train_df, test_df, feature)

	def define_features(self):
		steps = Classifier.define_features(self)
		steps.append(('clf', LinearSVC()))
		self.pipeline = Pipeline(steps)

	def run_predict(self):
		self.define_features()
		return self.predict(self.pipeline)

	def run_kfold(self):
		self.define_features()
		return self.predict_kfold(self.pipeline)

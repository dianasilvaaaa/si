from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.data_file import read_data_file
from si.model_selection.split import stratified_train_test_split
from si.metrics.accuracy import accuracy

class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = stratified_train_test_split(self.dataset, test_size=0.3)

    def test_fit(self):
        """
        Test the fitting process of the StackingClassifier.
        """
        # Define base models
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()

        # Define final model
        knn_final = KNNClassifier()

        # Create and fit StackingClassifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Assertions
        self.assertEqual(stacking_classifier.new_dataset.shape[0], self.train_dataset.shape[0])
        self.assertEqual(len(stacking_classifier.models), stacking_classifier.new_dataset.shape[1])

    def test_predict(self):
        """
        Test the predict functionality of the StackingClassifier.
        """
        # Define base models
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()

        # Define final model
        knn_final = KNNClassifier()

        # Create and fit StackingClassifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Make predictions
        predictions = stacking_classifier.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape[0])

    def test_score(self):
        """
        Test the scoring functionality of the StackingClassifier.
        """
        # Define base models
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        decision_tree = DecisionTreeClassifier()

        # Define final model
        knn_final = KNNClassifier()

        # Create and fit StackingClassifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Calculate accuracy
        accuracy_ = stacking_classifier.score(self.test_dataset)
        expected_accuracy = accuracy(self.test_dataset, stacking_classifier.predict(self.test_dataset))

        # Compare accuracy and assert
        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        average_precision_score, f1_score,
        brier_score_loss, classification_report,
        precision_recall_curve, roc_auc_score, roc_curve)

# Classifiers to be tested by default
default_classifiers = {
    'MultinomialNB': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_jobs=8)
}


class ClassificationResult:
    """
    Class for calculating model evaluation metrics.
    """
    def __init__(self, clf, y_real, y_pred, y_proba):
        """
        Initialize with real y, predicted y, and probabilities
        * y_proba should be a single column vector.
        """
        self.clf = clf
        self.y_real = y_real
        self.y_pred = y_pred
        self.y_proba = y_proba
    
    def calculate_scores(self):
        """
        Calculate various model evaluation metrics.
        """
        # Prediction based scores
        #self.report = classification_report(self.y_test, self.y_pred)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)
        
        # Probability based scores
        self.fpr, self.tpr, _ = roc_curve(self.y_test, self.y_proba)
        self.average_precision = average_precision_score(self.y_test, self.y_proba)
        self.brier_loss = brier_score_loss(self.y_test, self.y_proba)
        self.roc_auc = roc_curve(self.y_test, self.y_proba)
        self.prec_cur, self.recall_cur, _ = precision_recall_curve(self.y_test, self.y_proba)

    def print_metrics(self):
        print("Acuuracy  : %.3f" % self.accuracy)
        print("Precision : %.3f" % self.precision)
        print("Recall    : %.3f" % self.recall)
        print("F1 Score  : %.3f" % self.f1)
        print("Avg. Prec.: %.3f" % self.average_precision)
        print("Brier Loss: %.3f" % self.brier_loss)
        print("ROC AUC   : %.3f" % self.roc_auc)


class MyOneTimeClassifier:
    """
    Custom class for running classifiers of interest
    """
    def __init__(self, X_train, y_train, X_test, y_test,
                 classifier_name='Random Forest',
                 classifier=default_classifiers['Random Forest']):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf_name = classifier_name
        self.clf = classifier
        
    def classify(self, print_scores=False):
        """
        Conduct classification and calculate metrics
        """
        print("%s: Training model ..." % self.clf_name)
        self.clf.fit(self.X_train, self.y_train)

        print("%s: Calculating probablities ... " % self.clf_name)
        y_proba = self.clf.predict_proba(self.X_test)

        print("%s: Making predictions" % self.clf_name)
        y_pred = self.clf.predict(self.X_test)

        print("%s: Calculating metrics ..." % self.clf_name)
        res = ClassificationResult(self.clf, self.y_test, y_pred, y_proba)
        res.calculate_scores()

        # Print result if print_scores == True
        if print_scores:
            res.print_metrics
        
        return res
            

class MyCVClassifier:
    def __init__(self, X, y, CVFold=4,
                 classifier_name='Random Forest',
                 classifier=default_classifiers['Random Forest']):
        self.X = X
        self.y = y
        self.CVFold = CVFold
        self.clf_name = classifier_name
        self.clf = classifier

    def classify(self):
        kf = KFold(n_splits=self.CVFold)
        results = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier = MyOneTimeClassifier(
                            X_train, y_train, X_test, y_test,
                            classifier_name=self.clf_name,
                            classifier=self.clf)
            
            res = classifier.classify(print_scores=False)
            results.append(res)
        
        self.results = results
    
    def parse_results(self):
        pass



    

### END


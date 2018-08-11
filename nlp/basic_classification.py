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
        self.accuracy = accuracy_score(self.y_real, self.y_pred)
        self.precision = precision_score(self.y_real, self.y_pred)
        self.recall = recall_score(self.y_real, self.y_pred)
        self.f1 = f1_score(self.y_real, self.y_pred)
        
        # Probability based scores
        self.fpr, self.tpr, _ = roc_curve(self.y_real, self.y_proba)
        self.average_precision = average_precision_score(self.y_real, self.y_proba)
        self.brier_loss = brier_score_loss(self.y_real, self.y_proba)
        self.roc_auc = roc_auc_score(self.y_real, self.y_proba)
        self.prec_cur, self.recall_cur, _ = precision_recall_curve(self.y_real, self.y_proba)

    def print_metrics(self):
        print("Acuuracy  : %.3f" % self.accuracy)
        print("Precision : %.3f" % self.precision)
        print("Recall    : %.3f" % self.recall)
        print("F1 Score  : %.3f" % self.f1)
        print("Avg. Prec.: %.3f" % self.average_precision)
        print("Brier Loss: %.3f" % self.brier_loss)
        print("ROC AUC   : %.3f" % self.roc_auc)

    def plot_roc(self, ax=None, label=''):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.plot(self.fpr, self.tpr, label=label)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()

    def plot_precision_recall(self, ax=None, label=''):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.plot(self.recall_cur, self.prec_cur, label=label)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()

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
        
    def classify(self, verbose=True, print_scores=False):
        """
        Conduct classification and calculate metrics
        """
        if verbose:
            print("%s: Training model ..." % self.clf_name)
        self.clf.fit(self.X_train, self.y_train)

        if verbose:
            print("%s: Calculating probablities ... " % self.clf_name)
        y_proba = self.clf.predict_proba(self.X_test)

        if verbose:
            print("%s: Making predictions" % self.clf_name)
        y_pred = self.clf.predict(self.X_test)

        if verbose:
            print("%s: Calculating metrics ..." % self.clf_name)
        res = ClassificationResult(self.clf, self.y_test, y_pred, y_proba[:, 1])
        res.calculate_scores()

        # Print result if print_scores == True
        if print_scores:
            res.print_metrics
        
        return res
            

class MyCVClassifier:
    def __init__(self, X, y, CVFold=4, verbose=True,
                 classifier_name='Random Forest',
                 classifier=default_classifiers['Random Forest']):
        self.X = X
        self.y = y
        self.CVFold = CVFold
        self.verbose = verbose
        self.clf_name = classifier_name
        self.clf = classifier
        self.results = []
        self.scores_avg = {}
        self.scores_std = {}

    def classify(self):
        kf = KFold(n_splits=self.CVFold)
        results = []
        i = 1
        for train_index, test_index in kf.split(self.X, self.y):
            if self.verbose:
                print('-'*7, 'Fold %d' % (i), '-'*7)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            classifier = MyOneTimeClassifier(
                            X_train, y_train, X_test, y_test,
                            classifier_name=self.clf_name,
                            classifier=self.clf)
            
            res = classifier.classify(verbose=self.verbose, print_scores=False)
            results.append(res)
            i += 1
        
        self.results = results
    
    def parse_results(self):
        if not self.results:
            print("ERROR: No classification result exist.")
            return
        
        accuracies = [res.accuracy for res in self.results]
        precisions = [res.precision for res in self.results]
        recalls = [res.recall for res in self.results]
        f1s = [res.f1 for res in self.results]
        avg_precisions = [res.average_precision for res in self.results]
        briers = [res.brier_loss for res in self.results]
        roc_aucs = [res.roc_auc for res in self.results]

        self.scores_avg = {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s),
            'average precision': np.mean(avg_precisions),
            'brier loss': np.mean(briers),
            'ROC AUC': np.mean(roc_aucs)
        }

        self.scores_std = {
            'accuracy': np.std(accuracies),
            'precision': np.std(precisions),
            'recall': np.std(recalls),
            'f1': np.std(f1s),
            'average precision': np.std(avg_precisions),
            'brier loss': np.std(briers),
            'ROC AUC': np.std(roc_aucs)
        }

    def plot_roc(self, ax=None):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        for i, res in enumerate(self.results):
            res.plot_roc(ax=ax, label=('Fold %d' % (i + 1)))
        
        ax.set_title('ROC AUC = %.2f +/- %.3f' % \
                        (self.scores_avg['ROC AUC'], self.scores_std['ROC AUC']),
                    fontsize=14)
        
        isoline = np.linspace(0, 1, 10)
        ax.plot(isoline, isoline, ls='--', color='0.3')
    
    def plot_precision_recall(self, ax=None):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for i, res in enumerate(self.results):
            res.plot_precision_recall(ax=ax, label=('Fold %d' % (i + 1)))
        
        ax.set_title('avg_prec = %.2f, avg_recall = %.2f' % \
                        (self.scores_avg['precision'], self.scores_avg['recall']),
                    fontsize=14)
        
        isoline_x = np.linspace(0, 1, 10)
        isoline_y = 1 - isoline_x
        ax.plot(isoline_x, isoline_y, ls='--', color='0.3')

    def run(self, print_output=False):
        self.classify()
        self.parse_results()

        if print_output:
            print('='*10, '%d-fold CV Result' % self.CVFold, '='*10)
            for k, v in self.scores_avg.items():
                print('%s : %.2f +/- %.3f' % (k, v, self.scores_std[k]))


### END


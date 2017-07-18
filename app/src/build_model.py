"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._classifier = MultinomialNB()

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        _doc_term_matrix = self._vectorizer.fit_transform(X)
        self._classifier.fit(_doc_term_matrix, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        _doc_term_matrix = self._vectorizer.transform(X)
        return self._classifier.predict_proba(_doc_term_matrix)

    def predict(self, X):
        """Make predictions on new data."""
        _doc_term_matrix = self._vectorizer.transform(X)
        return self._classifier.predict(_doc_term_matrix)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        _doc_term_matrix = self._vectorizer.transform(X)
        return self._classifier.score(_doc_term_matrix)


def get_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    df = pd.read_csv(filename)
    X = df['body'].values
    y = df['section_name'].values
    return X, y


if __name__ == '__main__':
    X, y = get_data("data/articles.csv")
    tc = TextClassifier()
    tc.fit(X, y)
    print(tc.predict(np.array(['Test document'])))
    with open('static/model.pkl', 'w') as f:
        pickle.dump(tc, f)

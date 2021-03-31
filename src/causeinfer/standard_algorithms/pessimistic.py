"""
Pessimistic Uplift Transformation
---------------------------------

The Pessimistic Uplift Transformation Approach.

Based on
    Shaar, A., Abdessalem, T., and Segard, O. (2016). “Pessimistic Uplift Modeling”. ACM SIGKDD, August 2016, San Francisco, California USA, arXiv:1603.09738v1.
    URL:https://pdfs.semanticscholar.org/a67e/401715014c7a9d6a6679df70175be01daf7c.pdf.

Contents
    PessimisticUplift Class
        fit,
        predict (not available at this time),
        predict_proba
"""

from causeinfer.standard_algorithms.base_models import TransformationModel
from causeinfer.standard_algorithms.binary_transformation import BinaryTransformation
from causeinfer.standard_algorithms.reflective import ReflectiveUplift


class PessimisticUplift(TransformationModel):
    def __init__(self, model=None):
        try:
            model.__getattribute__("fit")
            model.__getattribute__("predict")
        except AttributeError:
            raise ValueError(
                "The passed model should contain both fit and predict methods."
            )
        self.w_binary_trans = BinaryTransformation(model, regularize=False)
        self.w_reflective = ReflectiveUplift(model)

    def fit(self, X, y, w):
        """
        Trains a model given covariates, responses and assignments.

        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                Matrix of covariates

            y : numpy.ndarray : (num_units,) : int, float
                Vector of unit responses

            w : numpy.ndarray : (num_units,) : int, float
                Vector of original treatment allocations across units

        Returns
        -------
            self : causeinfer.standard_algorithms.PessimisticUplift
                A trained model
        """
        self.w_binary_trans.fit(X, y, w)
        self.w_reflective.fit(X, y, w)

        return self

    # def predict(self, X):
    #     """
    #     Predicts a causal effect given covariates.

    #     Parameters
    #     ----------
    #         X : numpy.ndarray : (num_units, num_features) : int, float
    #             New data on which to make predictions

    #     Returns
    #     -------
    #         predictions : numpy.ndarray : (num_units, 2) : float
    #     """
    #     return predictions

    def predict_proba(self, X):
        """
        Predicts the probability that a subject will be a given class given covariates.

        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                New data on which to make predictions

        Returns
        -------
            probas : numpy.ndarray : (num_units, 2) : float
                Predicted probabilities for being a favorable class and an unfavorable class
        """
        w_binary_trans = self.w_binary_trans.predict_proba(X)
        w_reflective_uplift = self.w_reflective.predict_proba(X)

        return (w_binary_trans + w_reflective_uplift) / 2

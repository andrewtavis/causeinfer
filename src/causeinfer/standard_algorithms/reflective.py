"""
Reflective Uplift Transformation
--------------------------------

The Reflective Uplift Transformation Approach.

Based on
    Shaar, A., Abdessalem, T., and Segard, O. (2016). “Pessimistic Uplift Modeling”. ACM SIGKDD, August 2016, San Francisco, California USA, arXiv:1603.09738v1.
    URL:https://pdfs.semanticscholar.org/a67e/401715014c7a9d6a6679df70175be01daf7c.pdf.

Contents
    ReflectiveUplift Class
        fit,
        predict (not available at this time),
        predict_proba,
        _reflective_transformation,
        _reflective_weights
"""

import numpy as np
from causeinfer.standard_algorithms.base_models import TransformationModel


class ReflectiveUplift(TransformationModel):
    def __init__(self, model=None):
        try:
            model.__getattribute__("fit")
            model.__getattribute__("predict")
        except AttributeError:
            raise ValueError(
                "The passed model should contain both fit and predict methods."
            )
        self.model = model

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
            self : causeinfer.standard_algorithms.ReflectiveUplift
                A trained model
        """
        y_transformed = self._reflective_transformation(y, w)

        self.model.fit(X, y_transformed)
        self._reflective_weights(y, w)

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
        p_tp = self.model.predict_proba(X)[:, 0]
        p_cn = self.model.predict_proba(X)[:, 1]
        p_tn = self.model.predict_proba(X)[:, 2]
        p_cp = self.model.predict_proba(X)[:, 3]

        pred_fav = self.p_tp_fav * p_tp + self.p_cn_unfav * p_cn
        pred_unfav = self.p_tn_unfav * p_tn + self.p_cp_fav * p_cp

        return np.array([(pred_fav[i], pred_unfav[i]) for i in range(len(X))])

    def _reflective_transformation(self, y, w):
        """
        Assigns known quaternary (TP, CP, CN, TN) classes to units.

        Parameters
        ----------
            y : numpy.ndarray : (num_units,) : int, float
                Vector of unit responses

            w : numpy.ndarray : (num_units,) : int, float
                Vector of original treatment allocations across units

        Returns
        -------
            np.array(y_transformed) : np.array
                an array of transformed unit classes
        """
        y_transformed = []
        for i in range(y.shape[0]):
            if self.is_treatment_positive(y[i], w[i]):
                y_transformed.append(0)
            elif self.is_control_positive(y[i], w[i]):
                y_transformed.append(1)
            elif self.is_control_negative(y[i], w[i]):
                y_transformed.append(2)
            elif self.is_treatment_negative(y[i], w[i]):
                y_transformed.append(3)

        return np.array(y_transformed)

    def _reflective_weights(self, y, w):
        """
        Derives weights to normalize binary transformation noise.

        Parameters
        ----------
            y : numpy.ndarray : (num_units,) : int, float
                Vector of unit responses

            w : numpy.ndarray : (num_units,) : int, float
                Vector of original treatment allocations across units

        Returns
        -------
            p_tp_fav, p_cp_fav, p_cn_unfav, p_tn_unfav : np.array
                Probabilities of being a quaternary class per binary class
        """
        t_p, c_p, t_n, c_n = 0, 0, 0, 0
        fav_count, unfav_count = 0, 0
        size = y.shape[0]
        for i in range(size):
            if y[i] != 0:
                fav_count += 1
                if w[i] != 0:
                    t_p += 1
                else:
                    c_p += 1

            else:
                unfav_count += 1
                if w[i] != 0:
                    t_n += 1
                else:
                    c_n += 1

        self.p_tp_fav = t_p / fav_count
        self.p_cp_fav = c_p / fav_count
        self.p_cn_unfav = c_n / unfav_count
        self.p_tn_unfav = t_n / unfav_count

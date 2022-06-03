"""
Interaction Term
----------------

The Interaction Term Approach (The True Lift Model, The Dummy Variable Approach).

Based on
    Kuchumov, A. pyuplift: Lightweight uplift modeling framework for Python. (2019).
    URL: https://github.com/duketemon/pyuplift.
    License: https://github.com/duketemon/pyuplift/blob/master/LICENSE.

    Lo, VSY. (2002). “The true lift model: a novel data mining approach to response
    modeling in database marketing”. In:SIGKDD Explor4 (2), 78–86.
    URL: https://dl.acm.org/citation.cfm?id=772872

    Devriendt, F. et al. (2018). A Literature Survey and Experimental Evaluation of the   State-of-the-Art in Uplift
    Modeling: A Stepping Stone Toward the Development of Prescriptive Analytics. Big Data, Vol. 6, No. 1,   March 1,
    2018, pp. 1-29. Codes found at: data-lab.be/downloads.php.

Contents
    InteractionTerm Class
        fit,
        predict,
        predict_proba
"""

import numpy as np
from causeinfer.standard_algorithms.base_models import BaseModel


class InteractionTerm(BaseModel):
    def __init__(self, model=None):
        """
        Checks the attributes of the control and treatment models before assignment.
        """
        try:
            model.__getattribute__("fit")
            model.__getattribute__("predict")
        except AttributeError:
            raise AttributeError(
                "The passed model should contain both fit and predict methods."
            )

        self.model = model

    def fit(self, X, y, w):
        """
        Trains a model given covariates, responses and assignments.

        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                Matrix of covariates.

            y : numpy.ndarray : (num_units,) : int, float
                Vector of unit responses.

            w : numpy.ndarray : (num_units,) : int, float
                Vector of original treatment allocations across units.

        Returns
        -------
            self : causeinfer.standard_algorithms.InteractionTerm
                A trained model.
        """
        # Create the interaction term.
        Xw = X * w.reshape((-1, 1))

        # Add in treatment and interaction terms.
        X_fit = np.append(X, w.reshape((-1, 1)), axis=1)
        X_fit = np.append(X_fit, Xw, axis=1)

        self.model.fit(X_fit, y)

        return self

    def predict(self, X):
        """
        Predicts a causal effect given covariates.

        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                New data on which to make predictions.

        Returns
        -------
            predictions : numpy.ndarray : (num_units, 2) : float
                Predicted causal effects for all units given a 1 and 0 interaction term.
        """
        # Treatment interaction term and prediction covariates.
        w_treatment = np.full(X.shape[0], 1)
        Xw_treatment = X * w_treatment.reshape((-1, 1))

        X_pred_treatment = np.append(X, w_treatment.reshape((-1, 1)), axis=1)
        X_pred_treatment = np.append(X_pred_treatment, Xw_treatment, axis=1)

        # Control interaction term and prediction covariates.
        w_control = np.full(X.shape[0], 0)
        Xw_control = X * w_control.reshape((-1, 1))

        X_pred_control = np.append(X, w_control.reshape((-1, 1)), axis=1)
        X_pred_control = np.append(X_pred_control, Xw_control, axis=1)

        # Separate predictions.
        pred_treatment = self.model.predict(X_pred_treatment)
        pred_control = self.model.predict(X_pred_control)

        # Select the separate predictions for each interaction type.
        return np.array([(pred_treatment[i], pred_control[i]) for i in range(len(X))])

    def predict_proba(self, X):
        """
        Predicts the probability that a subject will be a given class given covariates.

        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                New data on which to make predictions.

        Returns
        -------
            probas : numpy.ndarray : (num_units, 2) : float
                Predicted causal probabilities for all units given a 1 and 0 interaction term.
        """
        # Treatment interaction term and prediction covariates.
        w_treatment = np.full(X.shape[0], 1)
        Xw_treatment = X * w_treatment.reshape((-1, 1))

        X_pred_treatment = np.append(X, w_treatment.reshape((-1, 1)), axis=1)
        X_pred_treatment = np.append(X_pred_treatment, Xw_treatment, axis=1)

        # Control interaction term and prediction covariates.
        w_control = np.full(X.shape[0], 0)
        Xw_control = X * w_control.reshape((-1, 1))

        X_pred_control = np.append(X, w_control.reshape((-1, 1)), axis=1)
        X_pred_control = np.append(X_pred_control, Xw_control, axis=1)

        # Separate probability predictions.
        pred_treatment = self.model.predict_proba(X_pred_treatment)
        pred_control = self.model.predict_proba(X_pred_control)

        # For each interaction type, select the probability to respond
        # given the treatment class.
        return np.array(
            [(pred_treatment[i][0], pred_control[i][0]) for i in range(len(X))]
        )

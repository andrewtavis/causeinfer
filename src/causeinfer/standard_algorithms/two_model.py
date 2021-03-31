"""
Two Model
---------

The Two Model Approach (Double Model, Separate Model).

Based on
    Hansotia, B. and B. Rukstales (2002). “Incremental value modeling”.
    In: Journal of Interactive Marketing 16(3), pp. 35–46.
    URL: https://search.proquest.com/openview/1f86b52432f7d80e46101b2b4b7629c0/1?cbl=32002&  pq-origsite=gscholar

    Devriendt, F. et al. (2018). A Literature Survey and Experimental Evaluation of the   State-of-the-Art in Uplift Modeling:
    A Stepping Stone Toward the Development of Prescriptive Analytics. Big Data, Vol. 6, No. 1,   March 1, 2018, pp. 1-29. Codes found at: data-lab.be/downloads.php.

Contents
    TwoModel Class
        fit,
        predict,
        predict_proba
"""

import numpy as np
from causeinfer.standard_algorithms.base_models import BaseModel


class TwoModel(BaseModel):
    def __init__(self, control_model=None, treatment_model=None):
        """
        Checks the attributes of the control and treatment models before assignment.
        """
        try:
            control_model.__getattribute__("fit")
            control_model.__getattribute__("predict")
        except AttributeError:
            raise AttributeError(
                "The passed control model should contain both fit and predict methods."
            )

        try:
            treatment_model.__getattribute__("fit")
            treatment_model.__getattribute__("predict")
        except AttributeError:
            raise AttributeError(
                "The passed treatment model should contain both fit and predict methods."
            )

        self.control_model = control_model
        self.treatment_model = treatment_model

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
            treatment_model, control_model : causeinfer.standard_algorithms.TwoModel
                Two trained models (one for training group, one for control)
        """
        # Split data into treatment and control subsets
        X_treatment, y_treatment = [], []
        X_control, y_control = [], []

        for i, e in enumerate(w):
            if e:
                X_treatment.append(X[i])
                y_treatment.append(y[i])
            else:
                X_control.append(X[i])
                y_control.append(y[i])

        # Fit two separate models
        self.treatment_model.fit(X_treatment, y_treatment)
        self.control_model.fit(X_control, y_control)

        return self

    def predict(self, X):
        """
        Predicts a causal effect given covariates.

        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                New data on which to make predictions

        Returns
        -------
            predictions : numpy.ndarray : (num_units, 2) : float
                Predicted causal effects for all units given treatment model and control
        """
        pred_treatment = self.treatment_model.predict(X)
        pred_control = self.control_model.predict(X)

        # Select the separate predictions for each model
        return np.array([(pred_treatment[i], pred_control[i]) for i in range(len(X))])

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
                Predicted probability to respond for all units given treatment and control models
        """
        pred_treatment = self.treatment_model.predict_proba(X)
        pred_control = self.control_model.predict_proba(X)

        # For each model, select the probability to respond given the treatment class
        return np.array(
            [(pred_treatment[i][0], pred_control[i][0]) for i in range(len(X))]
        )

# =============================================================================
# The Two Model Approach (Double Model, Separate Model)
# 
# Based on
# --------
#   Hansotia, B. and B. Rukstales (2002). “Incremental value modeling”. 
#   In: Journal of Interactive Marketing 16(3), pp. 35–46.
#   URL: https://search.proquest.com/openview/1f86b52432f7d80e46101b2b4b7629c0/1?cbl=32002&pq-origsite=gscholar
# 
# Contents
# --------
#   1. TwoModel Class
#       __init__
#       fit
#       predict
# =============================================================================

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import BaseModel

class TwoModel(BaseModel):
    
    def __init__(self, control_model=LinearRegression(), treatment_model=LinearRegression()):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            control_model.__getattribute__('fit')
            control_model.__getattribute__('predict')
        except AttributeError:
            raise AttributeError('Control model should contains two methods: fit and predict.')

        try:
            treatment_model.__getattribute__('fit')
            treatment_model.__getattribute__('predict')
        except AttributeError:
            raise AttributeError('Treatment model should contains two methods: fit and predict.')

        self.control_model = control_model
        self.treatment_model = treatment_model


    def fit(self, X, y, w):
        """
        Parameters
        ----------
            X : numpy ndarray (num_units, num_features) : int, float 
                Matrix of covariates

            y : numpy array (num_units,) : int, float
                Vector of unit reponses

            w : numpy array (num_units,) : int, float
                Vector of original treatment allocations across units
        
        Returns
        -------
            Two trained models (one for training group, one for control)
        """
        # Split data into treatment and control subsets
        X_treatment, y_treatment = [], []
        X_control, y_control = [], []

        for i, el in enumerate(w):
            if el:
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
        Parameters
        ----------
            X : numpy ndarray (num_units, num_features) : int, float
                New data on which to make predictions
        
        Returns
        -------
            predictions : numpy ndarray (num_units, 2) : float
                Predicted causal effects for all units given treatment model and control
        """
        pred_treatment = self.treatment_model.predict(X)
        pred_control = self.control_model.predict(X)

        predictions = np.array([(pred_treatment[i], pred_control[i]) for i in list(range(len(X)))])
        
        return predictions
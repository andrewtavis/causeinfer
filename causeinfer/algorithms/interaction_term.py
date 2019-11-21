"""
This module contains the Interaction Term Approach (The True Lift Model, Dummy Treatment Approach)

Based on
--------
- "The True Lift Model" (Lo, 2002)
"""

# =============================================================================
# Contents:
# 1. InteractionTerm Class
#   1.1 __init__
#   1.2 fit
#   1.3 predict
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.algorithms.base_models import BaseModel

class InteractionTerm(BaseModel):
    
    def __init__(self, model=LinearRegression()):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            model.__getattribute__('fit')
            model.__getattribute__('predict')
        except AttributeError:
            raise ValueError('Model should contains two methods: fit and predict.')
        
        self.model = model

    def fit(self, X, y, w):
        """
        Parameters
        ----------
        X : numpy ndarray (num_units, num_features) : int, float 
            Dataframe of covariates

        y : numpy array (num_units,) : int, float
            Vector of unit reponses

        w : numpy array (num_units,) : int, float
            Designates the original treatment allocation across units
        ----------
        
        Returns
        -------
        - A trained model
        """
        # Create the interaction term
        Xw = X * w.reshape((-1, 1))

        # Add in treatment and interaction terms
        X_train = np.append(X, w.reshape((-1, 1)), axis=1)
        X_train = np.append(X, Xw, axis=1)
        
        self.model.fit(X_train, y)
        
        return self


    def predict(self, X_pred):
        """
      Parameters
        ----------
        X_pred : int, float
             New data on which to make a prediction
        ----------
        
        Returns
        -------
        - Predicted uplift for all units
        """
        # For control
        treatment_dummy = np.array(X_pred.shape[0] * [0])
        X_pred = np.append(X_pred, treatment_dummy.reshape((-1, 1)), axis=1)
        pred_control = self.model.predict(X_pred)
        
         # For treatment
        X_pred[:, -1] = 1
        pred_treatment = self.model.predict(X_pred)
        
        return pred_treatment - pred_control
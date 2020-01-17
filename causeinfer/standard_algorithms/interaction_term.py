# =============================================================================
# The Interaction Term Approach (The True Lift Model, The Dummy Variable Approach)
# 
# Based on
# --------
#   Lo, VSY. (2002). “The true lift model: a novel data mining approach to response 
#   modeling in database marketing”. In:SIGKDD Explor4 (2), 78–86.
#   URL: https://dl.acm.org/citation.cfm?id=772872
# 
# Contents
# --------
#   1. InteractionTerm Class
#       __init__
#       fit
#       predict
# =============================================================================

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import BaseModel

class InteractionTerm(BaseModel):
    
    def __init__(self, model=LinearRegression()):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            model.__getattribute__('fit')
            model.__getattribute__('predict')
        except AttributeError:
            raise AttributeError('Model should contains two methods: fit and predict.')
        
        self.model = model

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
            A trained model
        """
        # Create the interaction term
        Xw = X * w.reshape((-1, 1))

        # Add in treatment and interaction terms
        X_fit = np.append(X, w.reshape((-1, 1)), axis=1)
        X_fit = np.append(X_fit, Xw, axis=1)
        
        self.model.fit(X_fit, y)
        
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
                Predicted causal effects for all units given a 1 and 0 interaction term
        """        
        # Treatment interaction term and prediction covariates
        w_treatment = np.full(X.shape[0], 1)
        Xw_treatment = X * w_treatment.reshape((-1, 1))
        
        X_pred_treatment = np.append(X, w_treatment.reshape((-1, 1)), axis=1)
        X_pred_treatment = np.append(X_pred_treatment, Xw_treatment, axis=1) 
        
        # Control interaction term and prediction covariates
        w_control = np.full(X.shape[0], 0)
        Xw_control = X * w_control.reshape((-1, 1))
        
        X_pred_control = np.append(X, w_control.reshape((-1, 1)), axis=1)
        X_pred_control = np.append(X_pred_control, Xw_control, axis=1)
        
        # Separate predictions
        pred_treatment = self.model.predict(X_pred_treatment)
        pred_control = self.model.predict(X_pred_control)

        predictions = np.array([(pred_treatment[i], pred_control[i]) for i in list(range(len(X)))])

        return predictions
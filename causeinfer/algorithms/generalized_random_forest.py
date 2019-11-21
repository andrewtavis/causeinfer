"""
This module contains the Generalized Random Forest approach

Based on
--------
- "Generalized Random Forests" (Athey, Tibshirani, and Wager, 2019) 
- The accompanything R package: https://github.com/grf-labs/grf/tree/master/r-package/grf/
- grf documentation: https://grf-labs.github.io/grf/
- "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (Wager and Athey 2018)
"""

# =============================================================================
# Contents:
# 1. GRF Class
#   1.1 __init__
#   1.2 fit
#   1.3 predict
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class GRF: # GeneralizedRandomForest

    def fit(self, X, y, w):
        """
        Parameters
        ----------
        X : dataframe of covariates (type(s): int, float)
        y : vector of unit reponses (type: int, float)
        w : binary vector designating the original treatment group allocation across units (type: float)
        model_class : the class of supervised learning model to use (base: LinearRegression)
        ----------

        Returns
        -------
        - A trained model
        """
        model = 1
        
        return model

    def predict(self, model, X_pred, continuous = False):
        """
        Parameters
        ----------
        model : a model that has been fit using the Generalized Random Forest algorithm
        X_pred : new data on which to make a prediction
        ----------

        Returns
        -------
        - A NumPy array of predicted outcomes for each unit in X_pred based on treatment assignment
        """
        pred_tuples = 1
        
        return pred_tuples
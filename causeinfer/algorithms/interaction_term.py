"""
This module contains the Interaction Term Approach (The True Lift Model, Dummy Treatment Approach)

Based on
--------
- "The True Lift Model" (Lo, 2002)
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# =============================================================================
# Contents:
# 1. InteractionTerm Class
#   1.1 __init__
#   1.2 interaction_term_fit
#   1.3 interaction_term_pred
# =============================================================================

class InteractionTerm():
    
    def interaction_term_fit(self, X, y, w, module = "linear_model", model_class = "LinearRegression"):
        """
        Parameters
        ----------
        X : numpy ndarray (num_units, num_features): int, float 
            Dataframe of covariates

        y : numpy array (num_units,): int, float
            Vector of unit reponses

        w : numpy array (num_units,): int, float
            Designates the original treatment allocation across units

        model_class : 
            The class of supervised learning model to use (base: LinearRegression)
        ----------
        
        Returns
        -------
        - A trained model
        """

        # create interaction terms
        xT = X * w

        # new data now includes the interaction term
        df = pd.DataFrame(X, w, xT)

        model = sklearn.module.model_class.fit(X = df, y=y)
        
        return model


    def interaction_term_pred(self, models, X_pred, y_id = "y", w_id ="w", continuous = False):
        """
      Parameters
        ----------
        model : 
            a model that has been fit using the "Response Treatment Approach"
        
        X_pred : int, float
             new data on which to make a prediction
        ----------
        
        Returns
        -------
        - A NumPy array of predicted outcomes for each unit in X_pred based on treatment assignment
        """
    
        predictors = names(X_pred)[(names(X_pred) != y_id) & (names(X_pred) != w_id)]
        
        xt_trt = X_pred["predictors"] * 1
        colnames(xt_trt) = paste("Int", colnames(xt_trt), sep = "_")
        df_trt = pd.DataFrame(X_pred, w=1, xt_trt)  
        
        xt_ctrl = X_pred["predictors"] * 0
        colnames(xt_ctrl) = paste("Int", colnames(xt_ctrl), sep = "_")
        df_ctrl = pd.DataFrame(X_pred, w=0, xt_ctrl)
        
        pred_trt = model.predict(df_trt)
        pred_ctrl = model.predict(df_ctrl)

        pred_tuples = [(pred_trt[i], pred_ctrl(i)) for i in list(range(len(X_pred)))]

        return np.array(pred_tuples)
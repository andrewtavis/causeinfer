"""
This module contains the Response Transformation Approach (Influential Marketing)

Based on
--------
- "Influential Marketing: A New Direct Marketing Strategy Addressing the Existence of Voluntary Buyers" (Lai, 2006)
- "Mining for the truly responsive customers and prospects using true-lift modeling: 
Comparison of new and existing methods" (Kane, 2014)
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# =============================================================================
# Contents:
# 1. ResponseTransformation Class
#   1.1 __init__
#   1.2 response_transformation_fit
#   1.3 response_transformation_pred
# =============================================================================

class ResponseTransformation():

    def response_transformation_fit(self, X, y, w, module = "linear_model", model_class = "LinearRegression"):
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

        df = pd.DataFrame(X, w_y = np.nan)

        df["w_y"][y == 1 & w == 1] = "TR" # Treated responders
        df["w_y"][y == 0 & w == 1] = "TN" # Treated non-responders
        df["w_y"][y == 1 & w == 0] = "CR" # Control responders
        df["w_y"][y == 0 & w == 0] = "CN" # Control non-responders

        model = sklearn.module.model_class.fit(X = df, y=y)
        
        return model


    def response_transformation_pred(self, model, X_pred, y_id = "y", w_id ="w", generalized = True, continuous = False):
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
        prob_C = prop.table(table(X_pred[str(w_id)]))[0] # Percent of units that are control
        prob_T = prop.table(table(X_pred[str(w_id)]))[1] # Percent of units that are treatment

        X_pred["w_y"] <- np.nan
        # The following splits the units into known classes, with the goal then being the derive those charactaristics for
        # Persuadables and Do Not Disturbs - the separated positive and negative classes - from the neutral classes 
        X_pred["w_y"][X_pred["y_id"] == 1 & X_pred["w_id"] == 1] = "TR" # Treated responders (Business: Sure Things or Persuadables)
        X_pred["w_y"][X_pred["y_id"] == 0 & X_pred["w_id"] == 1] = "TN" # Treated non-responders (Business: Lost Causes or Do Not Disturbs)
        X_pred["w_y"][X_pred["y_id"] == 1 & X_pred["w_id"] == 0] = "CR" # Control responders (Business: Sure Things or Do Not Disturbs)
        X_pred["w_y"][X_pred["y_id"] == 0 & X_pred["w_id"] == 0] = "CN" # Control non-responders (Business: Lost Causes or Persuadables)

        pred <- model.predict(X_pred)

        if generalized:
            # The generalized approach as suggested in Kane 2014
            pr_y1_w1 = ((pred["TR"] / prob_T) + (pred["CN"] / prob_C))
            pr_y1_w0 = ((pred["TN"] / prob_T) + (pred["CR"] / prob_C))
        else:
            # The original approach as suggested in Lai 2006 with the inclusion of Do Not Disturb units
            pr_y1_w1 = pred["TR"] + pred["CN"]
            pr_y1_w0 = pred["TN"] + pred["CR"]
        
        pred_tuples = [(pr_y1_w1[i], pr_y1_w0(i)) for i in list(range(len(X_pred)))]

        return np.array(pred_tuples)
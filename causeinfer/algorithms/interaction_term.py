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
from causeinfer.algorithms.base_models import BaseModel

# =============================================================================
# Contents:
# 1. InteractionTerm Class
#   1.1 __init__
#   1.2 interaction_term_fit
#   1.3 interaction_term_pred
# =============================================================================

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
        # Devriendt
        # create interaction terms
        xT = X * w

        # new data now includes the interaction term
        df = pd.DataFrame(X, w, xT)

        model = sklearn.module.model_class.fit(X = df, y=y)
        
        return model

        # pyuplift
        x_train = np.append(X, t.reshape((-1, 1)), axis=1)
        self.model.fit(x_train, y)
        return self


    def interaction_term_pred(self, X_pred):
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
        # Devriendt
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

        # pyuplift
        col = np.array(X.shape[0] * [0])
        x_test = np.append(X, col.reshape((-1, 1)), axis=1)
        # All treatment values == 0
        s0 = self.model.predict(x_test)
        x_test[:, -1] = 1
        # All treatment values == 1
        s1 = self.model.predict(x_test)
        return s1 - s0
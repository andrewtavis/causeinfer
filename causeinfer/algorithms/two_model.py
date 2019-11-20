"""
This module contains the Two Model Approach (Double Model, Separate Model)

Based on: 
- "Incremental Value Modeling" (Hansotia, 2002)
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from causeinfer.algorithms.base_models import BaseModel

# =============================================================================
# Contents:
# 1. TwoModel Class
#   1.1 __init__
#   1.2 two_mode_fitl
#   1.3 two_model_pred
# =============================================================================

class TwoModel(BaseModel):
    
    def __init__(self, control_model=LinearRegression(), treatment_model=LinearRegression()):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            control_model.__getattribute__('fit')
            control_model.__getattribute__('predict')
        except AttributeError:
            raise ValueError('Control model should contains two methods: fit and predict.')

        try:
            treatment_model.__getattribute__('fit')
            treatment_model.__getattribute__('predict')
        except AttributeError:
            raise ValueError('Treatment model should contains two methods: fit and predict.')

        self.control_model = control_model
        self.treatment_model = treatment_model


    def two_mode_fitl(self, X, y, w):
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
        - Two trained models (one for training group, one for control)
        """
        control_X, control_y = [], []
        treatment_X, treatment_y = [], []

        for idx, el in enumerate(w):
            if el:
                treatment_X.append(X[idx])
                treatment_y.append(y[idx])
            else:
                control_X.append(X[idx])
                control_y.append(y[idx])
        
        self.control_model.fit(control_X, control_y)
        self.treatment_model.fit(treatment_X, treatment_y)
        
        return self


    def two_model_pred(self, X_pred):
        """
        Parameters
        ----------
        X_pred : int, float
            new data on which to make a prediction
        ----------
        
        Returns
        -------
        - A NumPy array of predicted outcomes for each unit in X_pred based on treatment assignment
        """
        pred_treat = self.treatment_model.predict(X_pred)
        pred_control = self.control_model.predict(X_pred)

        tuples = [(pred_treat[i], pred_control(i)) for i in list(range(len(X_pred)))]
        
        return np.array(tuples)
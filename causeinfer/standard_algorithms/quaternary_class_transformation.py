# =============================================================================
# The Quaternary Class Transformation Approach (Response Transformation Approach)
# 
# Based on
# --------
#   Kane, K., Lo, VSY., and Zheng, J. (2014). “Mining for the truly responsive customers 
#   and prospects using truelift modeling: Comparison of new and existing methods”. 
#   In:Journal of Marketing Analytics 2(4), 218–238.
#   URL: https://link.springer.com/article/10.1057/jma.2014.18
# 
# Contents
# --------
#   1. QuaternaryClassTransformation Class
#       __init__
#       __quaternary_transformation
#       __quaternary_regularization
#       fit
#       predict
# =============================================================================

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import TransformationModel

class QuaternaryClassTransformation(TransformationModel):

    def __init__(self, model=LogisticRegression(n_jobs=-1), regularize=False):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            model.__getattribute__('fit')
            model.__getattribute__('predict')
        except AttributeError:
            raise AttributeError('Model should contains two methods: fit and predict.')
        
        self.model = model
        self.regularize = regularize
        

    def __quaternary_transformation(self, y, w):
        """
        Assigns known quaternary (TP, CP, CN, TN) classes to units

        Returns
        -------
            np.array(y_transformed) : an array of transformed unit classes
        """
        y_transformed = []
        for i in range(y.shape[0]):
            if self.is_treatment_positive(y[i], w[i]):
                y_transformed.append(0)
            elif self.is_control_positive(y[i], w[i]):
                y_transformed.append(1)
            elif self.is_control_negative(y[i], w[i]):
                y_transformed.append(2)
            elif self.is_treatment_negative(y[i], w[i]):
                y_transformed.append(3)
        
        return np.array(y_transformed)
    
    
    def __quaternary_regularization(self, y=None, w=None):
        """
        Regularization of quaternary classes is based on their treatment assignment
        """
        control_count, treatment_count = 0, 0
        for i in w:
            if i == 0.0:
                control_count += 1
            else:
                treatment_count += 1

        self.control_count = control_count
        self.treatment_count = treatment_count
        

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
        y_transformed = self.__quaternary_transformation(y, w)
        if self.regularize:
            self.__quaternary_regularization(y, w)
        
        self.model.fit(X, y_transformed)
        
        return self


    def predict(self, X, regularize=False):
        """
        Parameters
        ----------
            X : numpy ndarray (num_units, num_features) : int, float
                New data on which to make predictions
        
        Returns
        -------
            predictions : numpy ndarray (num_units, 2) : float
                Predicted probabilities for being an Affected Positive and Affected Negative
        """
        # Predictions for all four classes
        tp_pred = self.model.predict_proba(X)[:, 0]
        cp_pred = self.model.predict_proba(X)[:, 1]
        cn_pred = self.model.predict_proba(X)[:, 2]
        tn_pred = self.model.predict_proba(X)[:, 3]
        if self.regularize:
            ap_pred_regularized = tp_pred / self.treatment_count + cn_pred / self.control_count
            an_pred_regularized = tn_pred / self.treatment_count + cp_pred / self.control_count

            predictions = np.array([(ap_pred_regularized[i], an_pred_regularized[i]) for i in list(range(len(X)))])
        
        else:
            ap_pred = tp_pred + cn_pred
            an_pred = tn_pred + cp_pred

            predictions = np.array([(ap_pred[i], an_pred[i]) for i in list(range(len(X)))])
            
        return predictions
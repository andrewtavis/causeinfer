# =============================================================================
# The Quaternary Class Transformation Approach (Response Transformation Approach)
# 
# Based on
# --------
#   Kane, K., VSY. Lo, and J. Zheng (2014). “Mining for the truly responsive customers 
#   and prospects using truelift modeling: Comparison of new and existing methods”. 
#   In:Journal of Marketing Analytics 2(4), 218–238.
#   URL: https://link.springer.com/article/10.1057/jma.2014.18
# 
# Contents
# --------
#   1. QuaternaryClassTransformation Class
#       __init__
#       __encode_quaternary_class
#       __quaternary_regularization_weights
#       fit
#       predict
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import ClassTransformationModel

class QuaternaryClassTransformation(ClassTransformationModel): # import as QRT

    def __init__(self, model=LogisticRegression(n_jobs=-1), four_class=False, regularize=False):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            model.__getattribute__('fit')
            model.__getattribute__('predict')
        except AttributeError:
            raise ValueError('Model should contains two methods: fit and predict.')
        
        self.model = model
        self.regularize = regularize
        

    def __encode_quaternary_class(self, y, w):
        """
        Assigns quaternary (TP, CP, CN, TN) classes to units
        """
        y_encoded = []
        for i in range(y.shape[0]):
            if self.is_treatment_positive(y[i], w[i]):
                y_encoded.append(0)
            elif self.is_control_positive(y[i], w[i]):
                y_encoded.append(1)
            elif self.is_control_negative(y[i], w[i]):
                y_encoded.append(2)
            elif self.is_treatment_negative(y[i], w[i]):
                y_encoded.append(3)
        
        return np.array(y_encoded)
    
    
    def __quaternary_regularization_weights(self, y=None, w=None):
        """
        Regularization of quaternary classes is based on their treatment assignment
        """
        control_count, treatment_count = 0, 0
        for el in w:
            if el == 0.0:
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
                Dataframe of covariates

            y : numpy array (num_units,) : int, float
                Vector of unit reponses

            w : numpy array (num_units,) : int, float
                Designates the original treatment allocation across units
        
        Returns
        -------
            A trained model
        """
        y_encoded = self.__encode_quaternary_class(y, w)
        if self.regularize:
            self.__quaternary_regularization_weights(y, w)
        
        self.model.fit(X, y_encoded)
        
        return self


    def predict(self, X_pred, regularize=False):
        """
        Parameters
        ----------
            X_pred : int, float
                New data on which to make a prediction
        
        Returns
        -------
            Predicted uplift for all units
        """
        pred_treatment_positive = self.model.predict_proba(X_pred)[:, 0]
        pred_control_positive = self.model.predict_proba(X_pred)[:, 1]
        pred_control_negative = self.model.predict_proba(X_pred)[:, 2]
        pred_treatment_negative = self.model.predict_proba(X_pred)[:, 3]
        if self.regularize:
            
            return (pred_treatment_positive / self.treatment_count + pred_control_negative / self.control_count) - \
                (pred_treatment_negative / self.treatment_count + pred_control_positive / self.control_count)
        else:
            
            return (pred_treatment_positive + pred_control_negative) - (pred_treatment_negative + pred_control_positive)
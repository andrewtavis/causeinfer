"""
This module contains the Response Transformation Approach (Influential Marketing)

Based on
--------
- "Influential Marketing: A New Direct Marketing Strategy Addressing the Existence of Voluntary Buyers" (Lai, 2006)
- "Mining for the truly responsive customers and prospects using true-lift modeling: 
Comparison of new and existing methods" (Kane, 2014)
"""

# =============================================================================
# Contents:
# 1. ResponseTransformation Class
#   1.1 __init__
#   1.2 __encode_binary_unknown_class
#   1.3 __regularization_weights
#   1.4 fit
#   1.5 predict
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.algorithms.base_models import TransformationModel

class ResponseTransformation(TransformationModel):

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
        
    
    def __encode_binary_unknown_class(self, y, w, four_class=False):
        """
        Derives which of the unknown Affected Positive or Affected Negative 
        classes the unit could fall into
        """
        y_encode = []
        if not four_class:
            for i in range(y.shape[0]):
                # Possible Affected Positives (TPs or CNs)
                if self.is_treatment_positive(y[i], w[i]) or self.is_control_negative(y[i], w[i]):
                    y_encode.append(1)

                # Possible Affected Negatives (TNs or CPs)
                elif self.is_treatment_negative(y[i], w[i]) or self.is_control_positive(y[i], w[i]):
                    y_encode.append(0)
        
        else: 
            for i in range(y.shape[0]):
                if self.is_treatment_positive(y[i], w[i]):
                    y_encode.append(0)
                elif self.is_control_positive(y[i], w[i]):
                    y_encode.append(1)
                elif self.is_control_negative(y[i], w[i]):
                    y_encode.append(2)
                elif self.is_treatment_negative(y[i], w[i]):
                    y_encode.append(3)
        
        return np.array(y_encode)


    def __regularization_weights(self, y=None, w=None, four_class=False):
        """
        Derives regularization weights
        """
        if not four_class:
            aff_pos_count, aff_neg_count = 0, 0
            for i in range(y.shape[0]):
                # The number of possible Affected Positives (TPs or CNs)
                if self.is_treatment_positive(y[i], w[i]) or self.is_control_negative(y[i], w[i]):
                    aff_pos_count += 1

                # The number of possible Affected Negatives (TNs or CPs)
                elif self.is_treatment_negative(y[i], w[i]) or self.is_control_positive(y[i], w[i]):
                    aff_neg_count += 1

            self.ratio_aff_pos = aff_pos_count / (aff_pos_count + aff_neg_count)
            self.ratio_aff_neg = aff_neg_count / (aff_pos_count + aff_neg_count)

        else:
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
        ----------
        
        Returns
        -------
        - A trained model
        """
        y_encoded = self.__encode_binary_unknown_class(y, w)
        if self.regularize:
            self.__regularization_weights(y, w)
        
        self.model.fit(X, y_encoded)
        
        return self


    def predict(self, X_pred, four_class=False, regularize=False):
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
        if not four_class:
            pred_aff_pos = self.model.predict_proba(X_pred)[:, 1]
            if self.regularize:
                pred_aff_neg = self.model.predict_proba(X_pred)[:, 0]
                
                return (pred_aff_pos * self.ratio_aff_pos, pred_aff_neg * self.ratio_aff_neg)
            else:
                
                return 2 * pred_aff_pos - 1
        
        else:
            pred_treatment_positive = self.model.predict_proba(X_pred)[:, 0]
            pred_control_positive = self.model.predict_proba(X_pred)[:, 1]
            pred_control_negative = self.model.predict_proba(X_pred)[:, 2]
            pred_treatment_negative = self.model.predict_proba(X_pred)[:, 3]
            if self.regularize:
                
                return (pred_treatment_positive / self.treatment_count + pred_control_negative / self.control_count) - \
                    (pred_treatment_negative / self.treatment_count + pred_control_positive / self.control_count)
            else:
                
                return (pred_treatment_positive + pred_control_negative) - (pred_treatment_negative + pred_control_positive)
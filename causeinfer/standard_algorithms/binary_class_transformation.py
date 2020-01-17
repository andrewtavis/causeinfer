# =============================================================================
# The Binary Class Transformation Approach (Influential Marketing, Response Transformation Approach)
# 
# Based on
# --------
#   Lai, L.Y.-T. (2006). “Influential marketing: A new direct marketing strategy addressing 
#   the existence of voluntary buyers”. Master of Science thesis, Simon Fraser University School 
#   of Computing Science, Burnaby, BC,Canada. URL: https://summit.sfu.ca/item/6629
#   
#   Shaar, A., Abdessalem, T., and Segard, O. (2016). “Pessimistic Uplift Modeling”. ACM SIGKDD, 
#   August 2016, San Francisco, California USA, arXiv:1603.09738v1. 
#   URL:https://pdfs.semanticscholar.org/a67e/401715014c7a9d6a6679df70175be01daf7c.pdf.
# 
# Contents
# --------
#   1. BinaryClassTransformation Class
#       __init__
#       __binary_transformation
#       __binary_regularization
#       fit
#       predict
# =============================================================================

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import TransformationModel

class BinaryClassTransformation(TransformationModel):

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
        
    
    def __binary_transformation(self, y, w):
        """
        Derives which of the unknown Affected Positive or Affected Negative 
        classes the unit could fall into based known outcomes

        Returns
        -------
            np.array(y_transformed) : an array of transformed unit classes
        """
        y_transformed = []
        for i in range(y.shape[0]):
            # Favorable, possible Affected Positive units (TPs or CNs)
            if self.is_treatment_positive(y[i], w[i]) or self.is_control_negative(y[i], w[i]):
                y_transformed.append(1)

            # Unfavorable, possible Affected Negative units (TNs or CPs)
            elif self.is_treatment_negative(y[i], w[i]) or self.is_control_positive(y[i], w[i]):
                y_transformed.append(0)
        
        return np.array(y_transformed)


    def __binary_regularization(self, y=None, w=None):
        """
        Regularization of binary classes is based on the positive and negative binary affectual classes
        """
        # Initialize counts for Affected Positives and Affected Negatives
        ap_count, an_count = 0, 0
        for i in range(y.shape[0]):
            # Affected Positives (TPs or CNs)
            if self.is_treatment_positive(y[i], w[i]) or self.is_control_negative(y[i], w[i]):
                ap_count += 1

            # Affected Negatives (TNs or CPs)
            elif self.is_treatment_negative(y[i], w[i]) or self.is_control_positive(y[i], w[i]):
                an_count += 1

        self.ap_ratio = ap_count / (ap_count + an_count)
        self.an_ratio = an_count / (ap_count + an_count)
        

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
        y_transformed = self.__binary_transformation(y, w)
        if self.regularize:
            self.__binary_regularization(y, w)
        
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
        ap_pred = self.model.predict_proba(X)[:, 1]
        an_pred = self.model.predict_proba(X)[:, 0]
        if self.regularize:
            ap_pred_regularized = ap_pred * self.ap_ratio
            an_pred_regularized = an_pred * self.an_ratio
            
            predictions = np.array([(ap_pred_regularized[i], an_pred_regularized[i]) for i in list(range(len(X)))])
        
        else:    
            predictions = np.array([(ap_pred[i], an_pred[i]) for i in list(range(len(X)))])

        return predictions
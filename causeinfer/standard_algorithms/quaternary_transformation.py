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
#   Devriendt, F. et al. (2018). A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling: 
#   A Stepping Stone Toward the Development of Prescriptive Analytics. Big Data, Vol. 6, No. 1, March 1, 2018, pp. 1-29. 
#   Codes found at: data-lab.be/downloads.php.
# 
# Contents
# --------
#   1. QuaternaryTransformation Class
#       __init__
#       __quaternary_transformation
#       __quaternary_regularization
#       fit
#       predict (Not available at this time)
#       predict_proba
# =============================================================================

import numpy as np
from causeinfer.standard_algorithms.base_models import TransformationModel

class QuaternaryTransformation(TransformationModel):

    def __init__(self, model=None, regularize=False):
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
            X : numpy.ndarray : (num_units, num_features) : int, float 
                Matrix of covariates

            y : numpy.ndarray : (num_units,) : int, float
                Vector of unit reponses

            w : numpy.ndarray : (num_units,) : int, float
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


    # def predict(self, X):
    #     """
    #     Parameters
    #     ----------
    #         X : numpy ndarray : (num_units, num_features) : int, float
    #             New data on which to make predictions
        
    #     Returns
    #     -------
    #         predictions : numpy ndarray : (num_units, 2) : float
    #             Predicted probabilities for being a Favorable Clsss and Unfavorable Class
    #     """
    #     predictions = False
        
    #     return predictions


    def predict_proba(self, X):
        """
        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                New data on which to make predictions
        
        Returns
        -------
            predictions : numpy.ndarray : (num_units, 2) : float
                Predicted probabilities for being a Favorable Clsss and Unfavorable Class
        """
        # Predictions for all four classes
        pred_tp = self.model.predict_proba(X)[:, 0]
        pred_cp = self.model.predict_proba(X)[:, 1]
        pred_cn = self.model.predict_proba(X)[:, 2]
        pred_tn = self.model.predict_proba(X)[:, 3]
        if self.regularize:
            pred_fav_regularized = pred_tp / self.treatment_count + pred_cn / self.control_count
            pred_unfav_regularized = pred_tn / self.treatment_count + pred_cp / self.control_count

            predictions = np.array([(pred_fav_regularized[i], pred_unfav_regularized[i]) for i in list(range(len(X)))])
        
        else:
            pred_fav = pred_tp + pred_cn
            pred_unfav = pred_tn + pred_cp

            predictions = np.array([(pred_fav[i], pred_unfav[i]) for i in list(range(len(X)))])
            
        return predictions
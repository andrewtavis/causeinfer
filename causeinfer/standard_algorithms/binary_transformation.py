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
#   Devriendt, F. et al. (2018). A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling: 
#   A Stepping Stone Toward the Development of Prescriptive Analytics. Big Data, Vol. 6, No. 1, March 1, 2018, pp. 1-29. 
#   Codes found at: data-lab.be/downloads.php.
# 
# Contents
# --------
#   1. BinaryTransformation Class
#       __init__
#       __binary_transformation
#       __binary_regularization
#       fit
#       predict (Not available at this time)
#       predict_proba
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import TransformationModel

class BinaryTransformation(TransformationModel):

    def __init__(self, model=RandomForestClassifier(), regularize=False):
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
            np.array(y_transformed) : numpy.ndarray : an array of transformed unit classes
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
        # Initialize counts for Favorable and Unfavorable Classes
        fav_count, unfav_count = 0, 0
        for i in range(y.shape[0]):
            # Favorable (TPs or CNs) - contains all APs
            if self.is_treatment_positive(y[i], w[i]) or self.is_control_negative(y[i], w[i]):
                fav_count += 1

            # Unfavorable (TNs or CPs) - contains all ANs
            elif self.is_treatment_negative(y[i], w[i]) or self.is_control_positive(y[i], w[i]):
                unfav_count += 1

        self.fav_ratio = fav_count / (fav_count + unfav_count)
        self.unfav_ratio = unfav_count / (fav_count + unfav_count)
        

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
        y_transformed = self.__binary_transformation(y, w)
        if self.regularize:
            self.__binary_regularization(y, w)
        
        self.model.fit(X, y_transformed)
        
        return self


    # def predict(self, X):
    #     """
    #     Parameters
    #     ----------
    #         X : numpy.ndarray : (num_units, num_features) : int, float
    #             New data on which to make predictions
        
    #     Returns
    #     -------
    #         predictions : numpy.ndarray : (num_units, 2) : float
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
        pred_fav = self.model.predict_proba(X)[:, 1]
        pred_unfav = self.model.predict_proba(X)[:, 0]
        if self.regularize:
            pred_fav_regularized = pred_fav * self.fav_ratio
            pred_unfav_regularized = pred_unfav * self.unfav_ratio
            
            predictions = np.array([(pred_fav_regularized[i], pred_unfav_regularized[i]) for i in list(range(len(X)))])
        
        else:    
            predictions = np.array([(pred_fav[i], pred_unfav[i]) for i in list(range(len(X)))])

        return predictions
# =============================================================================
# The Binary Class Transformation Approach (Influential Marketing)
# 
# Based on
# --------
#   Lai, L.Y.-T. (2006). “Influential marketing: A new direct marketing strategy addressing 
#   the existence of voluntary buyers”. Master of Science thesis, Simon Fraser University School 
#   of Computing Science, Burnaby, BC,Canada. URL: https://summit.sfu.ca/item/6629
# 
# Contents
# --------
#   1. BinaryClassTransformation Class
#       __init__
#       __encode_binary_unknown_class
#       __binary_regularization_weights
#       fit
#       predict
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from causeinfer.standard_algorithms.base_models import ClassTransformationModel

class BinaryClassTransformation(ClassTransformationModel): # import as BCT

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
        
    
    def __encode_binary_unknown_class(self, y, w):
        """
        Derives which of the unknown Affected Positive or Affected Negative 
        classes the unit could fall into
        """
        y_encoded = []
        for i in range(y.shape[0]):
            # Possible Affected Positives (TPs or CNs)
            if self.is_treatment_positive(y[i], w[i]) or self.is_control_negative(y[i], w[i]):
                y_encoded.append(1)

            # Possible Affected Negatives (TNs or CPs)
            elif self.is_treatment_negative(y[i], w[i]) or self.is_control_positive(y[i], w[i]):
                y_encoded.append(0)
        
        return np.array(y_encoded)


    def __binary_regularization_weights(self, y=None, w=None):
        """
        Regularization of binary classes is based on the positive and negative binary affectual classes
        """
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
        y_encoded = self.__encode_binary_unknown_class(y, w)
        if self.regularize:
            self.__binary_regularization_weights(y, w)
        
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
        pred_aff_pos = self.model.predict_proba(X_pred)[:, 1]
        if self.regularize:
            pred_aff_neg = self.model.predict_proba(X_pred)[:, 0]
            
            return (pred_aff_pos * self.ratio_aff_pos, pred_aff_neg * self.ratio_aff_neg)
        else:
            
            return 2 * pred_aff_pos - 1
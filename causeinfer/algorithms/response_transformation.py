"""
This module contains the Response Transformation Approach (Influential Marketing)

Based on
--------
- "Influential Marketing: A New Direct Marketing Strategy Addressing the Existence of Voluntary Buyers" (Lai, 2006)
- "Mining for the truly responsive customers and prospects using true-lift modeling: 
Comparison of new and existing methods" (Kane, 2014)
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from causeinfer.algorithms.base_models import TransformationModel

# =============================================================================
# Contents:
# 1. ResponseTransformation Class
#   1.1 __init__
#   1.2 response_transformation_fit
#   1.3 response_transformation_pred
# =============================================================================

class ResponseTransformation():

    def __init__(self, model=LogisticRegression(n_jobs=-1), use_weights=False):
        """
        Checks the attributes of the contorl and treatment models before assignment
        """
        try:
            model.__getattribute__('fit')
            model.__getattribute__('predict')
        except AttributeError:
            raise ValueError('Model should contains two methods: fit and predict.')
        
        self.model = model
        self.use_weights = use_weights

    def response_transformation_fit(self, X, y, w):
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
        df = pd.DataFrame(X, w_y = np.nan)

        df["w_y"][y == 1 & w == 1] = "TR" # Treated responders
        df["w_y"][y == 0 & w == 1] = "TN" # Treated non-responders
        df["w_y"][y == 1 & w == 0] = "CR" # Control responders
        df["w_y"][y == 0 & w == 0] = "CN" # Control non-responders

        model = sklearn.module.model_class.fit(X = df, y=y)
        
        return model

        # pyuplift
        y_encoded = self.__encode_data(y, t)
        if self.use_weights:
            self.__init_weights(y, t)
        self.model.fit(X, y_encoded)
        return self


    def response_transformation_pred(self, X_pred):
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

        # pyuplift
        p_tr_cn = self.model.predict_proba(X)[:, 1]
        if self.use_weights:
            p_tn_cr = self.model.predict_proba(X)[:, 0]
            return p_tr_cn * self.p_tr_or_cn - p_tn_cr * self.p_tn_or_cr
        else:
            return 2 * p_tr_cn - 1


    def __encode_data(self, y, t):
        y_values = []
        for i in range(y.shape[0]):
            if self.is_tr(y[i], t[i]) or self.is_cn(y[i], t[i]):
                y_values.append(1)
            elif self.is_tn(y[i], t[i]) or self.is_cr(y[i], t[i]):
                y_values.append(0)
        return np.array(y_values)


    def __init_weights(self, y, t):
        pos_count, neg_count = 0, 0
        for i in range(y.shape[0]):
            if self.is_tr(y[i], t[i]) or self.is_cn(y[i], t[i]):
                pos_count += 1
            elif self.is_tn(y[i], t[i]) or self.is_cr(y[i], t[i]):
                neg_count += 1

        self.p_tr_or_cn = pos_count / (pos_count + neg_count)
        self.p_tn_or_cr = neg_count / (pos_count + neg_count)


# --------------------------------------
# Kane codes - if regularization == True
# --------------------------------------

        p_tr = self.model.predict_proba(X)[:, 0]
        p_cn = self.model.predict_proba(X)[:, 1]
        p_tn = self.model.predict_proba(X)[:, 2]
        p_cr = self.model.predict_proba(X)[:, 3]
        if self.use_weights:
            return (p_tr / self.treatment_count + p_cn / self.control_count) - \
                   (p_tn / self.treatment_count + p_cr / self.control_count)
        else:
            return (p_tr + p_cn) - (p_tn + p_cr)

    def __encode_data(self, y, t):
        y_values = []
        for i in range(y.shape[0]):
            if self.is_tr(y[i], t[i]):
                y_values.append(0)
            elif self.is_cn(y[i], t[i]):
                y_values.append(1)
            elif self.is_tn(y[i], t[i]):
                y_values.append(2)
            elif self.is_cr(y[i], t[i]):
                y_values.append(3)
        return np.array(y_values)

    def __init_weights(self, t):
        control_count, treatment_count = 0, 0
        for el in t:
            if el == 0.0:
                control_count += 1
            else:
                treatment_count += 1
        self.control_count = control_count
        self.treatment_count = treatment_count
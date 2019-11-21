"""
This module contains the base models for the following algorithms
1. The Two Model Approach
2. The Interaction Term Approach
3. The Response Transformation Appraoch

Note
----
- These classes should not be used directly. Use derived classes instead.
"""

# =============================================================================
# Contents:
# 1. BaseModel Class
#   1.1 fit
#   1.2 predict
# 2. TransformationModel Class
#   - See below for annotation explanation
#   2.1 is_treatment_positive
#   2.2 is_control_positive
#   2.3 is_control_negative
#   2.4 is_treatment_negative
# =============================================================================

class BaseModel:
    """
    Base class for the Two Model and Interaction Term Approaches
    """
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
        - self : object
        """
        return self

    def predict(self, X_pred, w_pred):
        """
        Parameters
        ----------
        X : numpy ndarray (num_pred_units, num_pred_features) : int, float 
            New data on which to make a prediction
        w_pred : (num_pred_units, num_pred_features) : int, float 
            Treatment allocation for predicted units
        Returns
        -------
        y_pred : numpy array (num_units,) or (num_pred_units, num_pred_features) : int, float
            Vector of predicted unit reponses
        """
        pass


class TransformationModel(BaseModel):
    """
    Base class for the Response Transformation Approach

    Notes
    -----
    - The following is non-standard annotation to combine marketing and other methodologies
    - Traditional marketing annotation is found in parentheses
    -----

    The response transformation approach splits the units based on response and treatment:
    - TP : Treatment Positives (Treatment Responders)
    - CP : Control Positives (Control Responders)
    - CN : Control Negatives (Control Nonresponders)
    - TN : Treatment Negatives (Treatment Nonresponders)

    From these four known classes we want to derive the charactaristic responses of four unknown classes:
    - AP : Affected Positives (Persuadables) : within TPs and CNs
    - UP : Unaffected Positives (Sure Things) : within TPs and CPs
    - UN : Unaffected Negatives (Lost Causes) : within CNs and TNs
    - AN : Affected Negatives (Do Not Disturbs) : within CPs and TNs

    The focus then falls onto predicting APs and ANs via their known classes
    """
    def is_treatment_positive(self, y, w): # (APs or UPs)
        """
        Parameters
        ----------
        y : int, float
            The target response

        w : int, float
            The treatment value
        ----------

        Returns
        -------
        is_treatment_positive : bool
        """
        return w == 1 and y == 1


    def is_control_positive(self, y, w): # (UPs or ANs)
        """
        Parameters
        ----------
        y : int, float
            The target response

        w : int, float
            The treatment value
        ----------

        Returns
        -------
        is_control_positive : bool
        """
        return w == 0 and y == 1


    def is_control_negative(self, y, w): # (APs or UNs)
        """
        Parameters
        ----------
        y : int, float
            The target response

        w : int, float
            The treatment value
        ----------

        Returns
        -------
        is_control_negative : bool
        """
        return w == 0 and y == 0

    def is_treatment_negative(self, y, w): # (UNs or ANs)
        """
        Parameters
        ----------
        y : int, float
            The target response

        w : int, float
            The treatment value
        ----------

        Returns
        -------
        is_treatment_negative : bool
        """
        return w == 1 and y == 0
"""
Base Models
-----------

Base models for the following algorithms:

- The Two Model Approach
- The Interaction Term Approach
- The Binary Class Transformation (BCT) Approach
- The Quaternary Class Transformation (QCT) Approach
- The Reflective Uplift Approach
- The Pessimistic Uplift Approach

Note: these classes should not be used directly. Please use derived classes instead.

Contents
    BaseModel Class
        fit,
        predict

    TransformationModel Class (see annotation/methodology explanation)
        is_treatment_positive,
        is_control_positive,
        is_control_negative,
        is_treatment_negative
"""


class BaseModel:
    """
    Base class for the Two Model and Interaction Term Approaches.
    """

    def fit(self, X, y, w):
        """
        Parameters
        ----------
            X : numpy.ndarray : (num_units, num_features) : int, float
                Dataframe of covariates

            y : numpy.ndarray : (num_units,) : int, float
                Vector of unit responses

            w : numpy.ndarray : (num_units,) : int, float
                Designates the original treatment allocation across units

        Returns
        -------
            self : object
        """
        return self

    def predict(self, X, w):
        """
        Parameters
        ----------
            X : numpy.ndarray : (num_pred_units, num_pred_features) : int, float
                New data on which to make a prediction

            w : numpy.ndarray : (num_pred_units, num_pred_features) : int, float
                Treatment allocation for predicted units

        Returns
        -------
            y_pred : numpy.ndarray : (num_pred_units,) : int, float
                Vector of predicted unit responses
        """
        pass


class TransformationModel(BaseModel):
    """
    Base class for the Response Transformation Approaches.

    Notes
    -----
        The following is non-standard annotation to combine marketing and other methodologies

        Traditional marketing annotation is found in parentheses

        The response transformation approach splits the units based on response and treatment:

            TP : Treatment Positives (Treatment Responders)

            CP : Control Positives (Control Responders)

            CN : Control Negatives (Control Nonresponders)

            TN : Treatment Negatives (Treatment Nonresponders)

        From these four known classes we want to derive the characteristic responses of four unknown classes:

            AP : Affected Positives (Persuadables) : within TPs and CNs

            UP : Unaffected Positives (Sure Things) : within TPs and CPs

            UN : Unaffected Negatives (Lost Causes) : within CNs and TNs

            AN : Affected Negatives (Do Not Disturbs) : within CPs and TNs

        The focus then falls onto predicting APs and ANs via their known classes
    """

    def is_treatment_positive(self, y, w):  # (APs or UPs)
        """
        Checks if a subject did respond when treated

        Parameters
        ----------
            y : int, float
                The target response

            w : int, float
                The treatment value

        Returns
        -------
            is_treatment_positive : bool
        """
        return w == 1 and y == 1

    def is_control_positive(self, y, w):  # (UPs or ANs)
        """
        Checks if a subject did respond when not treated

        Parameters
        ----------
            y : int, float
                The target response

            w : int, float
                The treatment value

        Returns
        -------
            is_control_positive : bool
        """
        return w == 0 and y == 1

    def is_control_negative(self, y, w):  # (APs or UNs)
        """
        Checks if a subject didn't respond when not treated

        Parameters
        ----------
            y : int, float
                The target response

            w : int, float
                The treatment value

        Returns
        -------
            is_control_negative : bool
        """
        return w == 0 and y == 0

    def is_treatment_negative(self, y, w):  # (UNs or ANs)
        """
        Checks if a subject didn't respond when treated

        Parameters
        ----------
            y : int, float
                The target response

            w : int, float
                The treatment value

        Returns
        -------
            is_treatment_negative : bool
        """
        return w == 1 and y == 0

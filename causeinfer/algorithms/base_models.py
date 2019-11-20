class BaseModel:
    """Base class for uplift models.
    Note: This class should not be used directly. Use derived classes instead.
    """

    def fit(self, X, y, t):
        """Build a TwoModel approach from the training set (x, y, t).
        Parameters
        ----------
        X : numpy array of shape = [n_samples, n_features]
            The training input samples.
        y : numpy array of shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in regression).
        t : numpy array of shape = [n_samples] or [n_samples, n_outputs]
            The treatments.
        Returns
        -------
        self : object
        """
        return self

    def predict(self, X, t):
        """Predict treatment effect for x.
        Parameters
        ----------
        X : numpy array of shape = [n_samples, n_features]
            The input samples.
        t : numpy array of shape = [n_samples, n_features]
            The treatments.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted treatment effects.
        """
        pass


class TransformationModel(BaseModel):
    """Base class for a transformation uplift models.
    Note: This class should not be used directly. Use derived classes instead.
    """

    def is_tr(self, y, t):
        """Is pair (y,t) is TR?
        Treatment responders (TR) are customers who were treated and responded
        Parameters
        ----------
        y : float
            The target value.
        t : float
            The treatment value.
        Returns
        -------
        is_tr : bool
        """
        return t != 0 and y != 0

    def is_cn(self, y, t):
        """Is pair (y,t) is CN?
        Control nonresponders (CN) are the customers who did not receive a treatment and did not respond.
        Parameters
        ----------
        y : float
            The target value.
        t : float
            The treatment value.
        Returns
        -------
        is_tr : bool
        """
        return t == 0 and y == 0

    def is_tn(self, y, t):
        """Is pair (y,t) is TN?
        Treatment nonresponders (TN) are customers who received a treatment but did not respond.
        Parameters
        ----------
        y : float
            The target value.
        t : float
            The treatment value.
        Returns
        -------
        is_tr : bool
        """
        return t != 0 and y == 0

    def is_cr(self, y, t):
        """Is pair (y,t) is CR?
        Control responders (CR) are the customers who responded without having received a treatment.
        Parameters
        ----------
        y : float
            The target value.
        t : float
            The treatment value.
        Returns
        -------
        is_tr : bool
        """
        return t == 0 and y != 0
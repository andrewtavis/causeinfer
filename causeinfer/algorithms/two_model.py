class TwoModel:

    def two_mode_fitl(self, X, y, w, module = "linear_model", model_class = "LinearRegression"):
        """
        Trains a model using the "Two Model Approach" (Double Model, Separate Model)

        Based on
        --------
        - "Incremental Value Modeling" (Hansotia, 2002)

        Requirements
        ------------
        - pandas : used for grouping via the DataFrame module
        - scikit-learn : used for training via sklearn.module.model_class.fit()
        - For model options see : https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

        Parameters
        ----------
        X : dataframe of covariates (type(s): int, float)
        y : vector of unit reponses (type: int, float)
        w : binary vector designating the original treatment group allocation across units (type: float)
        model_class : the class of supervised learning model to use (base: LinearRegression)

        Returns
        -------
        - Two trained models - one for the training group, and one for control
        """
        from sklearn import linear_model
        from sklearn import ensemble
        import sklearn
        import pandas as pd 

        df = pd.DataFrame(X, w)

        model_trn = sklearn.module.model_class.fit(X = df[df[w==1]], y=y)   # fit for treatment group
        model_ctrl = sklearn.module.model_class.fit(X = df[df[w==0]], y=y)  # fit for control group

        res = list(model_treatment = model_trn,
                model_control = model_ctrl, 
                model_class = model_class)

        return(res)

    def two_model_pred(self, models, X_pred, continuous = False):
        """
        Makes predicitons using the "Two Model Approach" (Double Model, Separate Model)

        Based on
        --------
        - "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling:
        A Stepping Stone Toward the Development of Prescriptive Analytics" (Devriendt, 2018) 
        - "Incremental Value Modeling" (Hansotia, 2002)

        Requirements
        ------------
        - NumPy : for arrays
        - scikit-learn : used for predictions

        Parameters
        ----------
        models : a list of two models that have been fit on the treatment and control groups respectively
        X_pred : new data on which to make a prediction

        Returns
        -------
        - A NumPy array of predicted outcomes for each unit in X_pred based on treatment assignment
        """
        import numpy as np

        pred_trt = models[model_treatment].predict(X_pred)
        pred_ctrl = models[model_control].predict(X_pred)

        tuples = [(pred_trt[i], pred_ctrl(i)) for i in list(range(len(X_pred)))]

        return np.array(tuples)
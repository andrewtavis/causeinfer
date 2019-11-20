class GRF: # GeneralizedRandomForest

    def grf_fit(self, X, y, w):
        """
        Trains a model using the Generalized Random Forest algorithm

        Based on
        --------
        - "Generalized Random Forests" (Athey, Tibshirani, and Wager, 2019) 
        - The accompanything R package: https://github.com/grf-labs/grf/tree/master/r-package/grf/
        - grf documentation: https://grf-labs.github.io/grf/
        - "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (Wager and Athey 2018)

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
        - A trained model
        """
        model = 1
        return model

    def grf_pred(self, model, X_pred, continuous = False):
        """
        Makes predicitons using the Generalized Random Forest algorithm

        Based on
        --------
        - "Generalized Random Forests" (Athey et al, 2018) 
        - The accompanything R package: https://github.com/grf-lab∂s/grf/tree/master/r-package/grf/
        - grf documentation: https://grf-labs.github.io/grf/
        - "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (Wager and Athey 2018)

        Requirements
        ------------
        - NumPy : for arrays
        - scikit-learn : used for predictions

        Parameters
        ----------
        model : a model that has been fit using the Generalized Random Forest algorithm
        X_pred : new data on which to make a prediction

        Returns
        -------
        - A NumPy array of predicted outcomes for each unit in X_pred based on treatment assignment
        """
        pred_tuples = 1
        return np.array(pred_tuples)
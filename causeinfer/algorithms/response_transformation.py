class ResponseTransformation:

    def response_transformation(self, X, y, w, module = "linear_model", model_class = "LinearRegression"):
        """
        Trains a model using the "Response Transformation Approach" (Influential Marketing)

        Based on
        --------
        - "Influential Marketing: A New Direct Marketing Strategy Addressing the Existence of Voluntary Buyers" (Lai, 2006)
        - "Mining for the truly responsive customers and prospects using true-lift modeling: 
        Comparison of new and existing methods" (Kane, 2014)

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
        from sklearn import linear_model
        from sklearn import ensemble
        import sklearn
        import pandas as pd 

        df = pd.DataFrame(X, w_y = np.nan)

        df["w_y"][y == 1 & w == 1] = "TR" # Treated responders
        df["w_y"][y == 0 & w == 1] = "TN" # Treated non-responders
        df["w_y"][y == 1 & w == 0] = "CR" # Control responders
        df["w_y"][y == 0 & w == 0] = "CN" # Control non-responders

        model = sklearn.module.model_class.fit(X = df, y=y)
        
        return model

    def response_transformation_pred(self, model, X_pred, y_id = "y", w_id ="w", generalized = True, continuous = False):
        """
        Makes predicitons using the "Response Treatment Approach" (Influential Marketing)

        Based on
        --------
        - "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling:
        A Stepping Stone Toward the Development of Prescriptive Analytics" (Devriendt, 2018) 
        - "Influential Marketing: A New Direct Marketing Strategy Addressing the Existence of Voluntary Buyers" (Lai, 2006)
        - "Mining for the truly responsive customers and prospects using true-lift modeling: 
        Comparison of new and existing methods" (Kane, 2014)

        Requirements
        ------------
        - NumPy : for arrays
        - scikit-learn : used for predictions

        Parameters
        ----------
        model : a model that has been fit using the "Response Treatment Approach"
        X_pred : new data on which to make a prediction

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
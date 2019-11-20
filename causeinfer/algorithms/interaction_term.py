class InteractionTerm:
    
    def interaction_term_fit(self, X, y, w, module = "linear_model", model_class = "LinearRegression"):
        """
        Trains a model using the "Interaction Term Approach" (Dummy Treatment Approach)

        Based on
        --------
        - "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling:
        A Stepping Stone Toward the Development of Prescriptive Analytics" (Devriendt, 2018) 
        - "The True Lift Model" (Lo, 2002)

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

        # create interaction terms
        xT = X * w

        # new data now includes the interaction term
        df = pd.DataFrame(X, w, xT)

        model = sklearn.module.model_class.fit(X = df, y=y)
        
        return model

    def interaction_term_pred(self, model, X_pred, y_id = "y", w_id ="w", continuous = False):
        """
        Makes predicitons using the "Interaction Term Approach" (Dummy Treatment Approach)

        Based on
        --------
        - "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling:
        A Stepping Stone Toward the Development of Prescriptive Analytics" (Devriendt, 2018) 
        - "The True Lift Model" (Lo, 2002)

        Requirements
        ------------
        - NumPy : for arrays
        - scikit-learn : used for predictions

        Parameters
        ----------
        models : a model that has been fit using the "Interaction Term Approach"
        X_pred : new data on which to make a prediction

        Returns
        -------
        - A NumPy array of predicted outcomes for each unit in X_pred based on treatment assignment
        """
        import numpy as np
        import pandas as pd
    
        predictors = names(X_pred)[(names(X_pred) != y_id) & (names(X_pred) != w_id)]
        
        xt_trt = X_pred["predictors"] * 1
        colnames(xt_trt) = paste("Int", colnames(xt_trt), sep = "_")
        df_trt = pd.DataFrame(X_pred, w=1, xt_trt)  
        
        xt_ctrl = X_pred["predictors"] * 0
        colnames(xt_ctrl) = paste("Int", colnames(xt_ctrl), sep = "_")
        df_ctrl = pd.DataFrame(X_pred, w=0, xt_ctrl)
        
        pred_trt = model.predict(df_trt)
        pred_ctrl = model.predict(df_ctrl)

        pred_tuples = [(pred_trt[i], pred_ctrl(i)) for i in list(range(len(X_pred)))]

        return np.array(pred_tuples)
class CauseInfer:

    # ----------------------------------------------------------------------------------------------------------------------
    # 1. Two Model Approach
    # ----------------------------------------------------------------------------------------------------------------------


    def two_model_fit(X, y, w, module = "linear_model", model_class = "LinearRegression"):
        """
        Trains a model using the "Two Model Approach" (Double Model, Separate Model)

        Based on
        --------
        - "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling:
        A Stepping Stone Toward the Development of Prescriptive Analytics" (Devriendt, 2018) 
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

    def two_model_pred(models, X_pred, continuous = False):
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


    # ----------------------------------------------------------------------------------------------------------------------
    # 2. Interaction Term Approach - Lo
    # ----------------------------------------------------------------------------------------------------------------------


    def interaction_term_fit(X, y, w, module = "linear_model", model_class = "LinearRegression"):
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

    def interaction_term_pred(model, X_pred, y_id = "y", w_id ="w", continuous = False):
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


    # ----------------------------------------------------------------------------------------------------------------------
    # 3. Response Transformation Approach - Lai, Kane
    # ----------------------------------------------------------------------------------------------------------------------


    def response_transformation_fit(X, y, w, module = "linear_model", model_class = "LinearRegression"):
        """
        Trains a model using the "Response Treatment Approach" (Influential Marketing)

        Based on
        --------
        - "A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling:
        A Stepping Stone Toward the Development of Prescriptive Analytics" (Devriendt, 2018) 
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

    def response_transformation_pred(model, X_pred, y_id = "y", w_id ="w", generalized = True, continuous = False):
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


    # ----------------------------------------------------------------------------------------------------------------------
    # 4. Generalized Random Forest (GRF) - Athey, Wager
    # ----------------------------------------------------------------------------------------------------------------------


    def grf_pred(X, y, w):
        """
        Trains a model using the Generalized Random Forest algorithm

        Based on
        --------
        - "Generalized Random Forests" (Athey et al, 2018) 
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

    def grf_pred(model, X_pred, continuous = False):
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


    # ----------------------------------------------------------------------------------------------------------------------
    # 5. Evaluation Metrics
    # ----------------------------------------------------------------------------------------------------------------------


    def generalized_random_forest_confidence_intervals(): # import as grf_ci

        return GRF_CIs

    def qini_AUUC_scores(qini = True):

        return qini_AUUC
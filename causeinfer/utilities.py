# =============================================================================
# Utility functions for data manipulation and processing
# 
# Contents
# --------
#   0. No Class
#       train_test_split
# =============================================================================

import random

def train_test_split(X, y, w, percent_train=0.7, random_state=None):
    """
    Train-test split for unit X covariates and (y,w) outcome tuples

    Parameters
    ----------
        X : [n_samples, n_features]
            Matrix of unit covariate features

        y : [n_samples,]
            Array of unit responses

        w : [n_samples,]
            Array of unit treatments

        percent_train : 
            The percent of the covariates and outcomes to delegate to model training
        
        random_state : 
            A seed for the random number generator to allow for consistency (when in doubt, 42)

    Result
    ------
        X_train, X_test, y_train, y_test, w_train, w_test : numpy array
            Arrays of split covariates and outcomes
    """
    if not (0 < percent_train < 1):
        raise ValueError('Train share should be float between 0 and 1.')

    assert len(X) == len(y) == len(w), "Lengths of covariates and outcomes not equal."

    random.seed(random_state)
    N = len(X)
    N_train = int(percent_train * N)
    train_index = random.sample([i for i in range(N)], N_train)
    test_index = [i for i in range(N) if i not in train_index]

    X_train = X[train_index, :]
    X_test = X[test_index, :]

    y_train = y[train_index]
    y_test = y[test_index]

    w_train = w[train_index]
    w_test = w[test_index]
    
    return X_train, X_test, y_train, y_test, w_train, w_test
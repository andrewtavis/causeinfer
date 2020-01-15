# =============================================================================
# Utility functions for data manipulation and processing
# 
# Contents
# --------
#   0. No Class
#       train_test_split
#       plot_unit_distributions
# =============================================================================

import random
import pandas as pd
import seaborn as sns

def train_test_split(X, y, w, 
                    percent_train=0.7, 
                    random_state=None):
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


def plot_unit_distributions(df, variable, treatment=None, 
                            plot_x_lab=None, plot_y_lab=None, plot_title=None, 
                            bins=None, fontsize=20, axis=None):
    """
    Plots seaborn countplots of unit covariate and outcome distributions

    Parameters
    ----------
        df_plot : pandas df, [n_samples, n_features]
            The data from which the plot is made

        variable : str
            A unit covariate or outcome for which the plot is desired
            
        treatment : str, optional (default=None)
            The treatment variable for comparing across segments

        x_label : str, optional (default=None)
            Label for the x-axis of the plot

        y_label : str, optional (default=None)
            label for the y-axis of the plot

        title : str, optional (default=None)
            Title for the plot

        bins : int (default=None)
            Bins the column values such that larger distributions can be plotted
            
        fontsize : int or float (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined
    
    Result
    ------
        A seaborn plot of unit distributions across the given covariate or outcome value
    """
    import re

    def int_or_text(char):
        return int(char) if char.isdigit() else char

    def alphanumeric_sort(text):
        """
        Added so the columns are correctly ordered
        """
        return [int_or_text(char) for char in re.split(r'(\d+)', text)]
    
    def float_range(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    # Set different colors for treatment plots
    if treatment:
        color_pallette = 'Set2'
    else:
        color_pallette = 'Set1'
    
    # Bin if necessary
    if df[str(variable)].dtype != int or float:
        try:
            df[str(variable)] = df[str(variable)].astype(float)
        except:
            print("The data type for the column can't be binned. The values of the calumn will be used as is.")
            bins=False
    
    if bins:
        bin_segments = list(float_range(df[str(variable)].min(), 
                                        df[str(variable)].max(),
                                        (df[str(variable)].max()-df[str(variable)].min())/bins))
        
        # So plotting bounds are clean
        bin_segments = [int(i) for i in bin_segments[0:-2]] + [int(bin_segments[-1])+1]
        
        # Bin the variable column based on the above defined list of segments
        df['binned_variable'] = pd.cut(df[str(variable)], bin_segments)
    
        order = list(df['binned_variable'].value_counts().index)    
        order.sort()
        ax = sns.countplot(data=df,
                           x='binned_variable',
                           hue=treatment,
                           order=order,
                           ax=axis,
                           palette=color_pallette)
        
        df.drop('binned_variable', axis=1, inplace=True)
    
    else:
        order = list(df[str(variable)].value_counts().index)
        order.sort(key=alphanumeric_sort)
        ax = sns.countplot(data=df,
                           x=variable,
                           hue=treatment,
                           order=order,
                           ax=axis,
                           palette=color_pallette)
    
    ax.set_xlabel(plot_x_lab, fontsize=fontsize)
    ax.set_ylabel(plot_y_lab, fontsize=fontsize)
    ax.axes.set_title(plot_title, fontsize=fontsize*1.5)
    ax.tick_params(labelsize=fontsize/1.5)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
# =============================================================================
# Utility functions for data manipulation and processing
# 
# Contents
# --------
#   0. No Class
#       train_test_split
#       plot_unit_distributions
#       over_sample
# =============================================================================

import numpy as np
import pandas as pd
import random
import seaborn as sns

def train_test_split(X, y, w, 
                    percent_train=0.7, 
                    random_state=None,
                    maintain_proportions=False):
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

        percent_train : float
            The percent of the covariates and outcomes to delegate to model training
        
        random_state : int
            A seed for the random number generator to allow for consistency (when in doubt, 42)

        maintain_proportions : bool, optional (default=False)
            Whether to maintain the treatment group proportions within the split samples

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
                            bins=None, fontsize=20, figsize=None, axis=None):
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
            
        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined
    
    Result
    ------
        Displaus a seaborn plot of unit distributions across the given covariate or outcome value
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

    # Adaptable figure sizes
    if figsize:
        sns.set(rc={'figure.figsize':figsize})

    # Set different colors for treatment plots
    if treatment:
        color_pallette = 'Set2'
    else:
        color_pallette = 'Set1'
    
    # Bin if necessary
    if bins:
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
        try:
            order = [float(i) for i in order]
            order.sort(key=int)
        except:
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


def over_sample(X_1, y_1, w_1, sample_2_size, shuffle=True):
    """
    Over-samples to provide equallity between a given sample and another it is smaller than
    
    Parameters
    ----------
        X_1 : numpy ndarray (num_sample1_units, num_sample1_features)
            Dataframe of sample covariates

        y_1 : numpy array (num_sample1_units,)
            Vector of sample unit reponses

        w_1 : numpy array (num_sample1_units,)
            Designates the original treatment allocation across sample units
            
        sample_2_size : int
            The size of the other sample to match
            
        shuffle : bool, optional (default=True)
            Whether to shuffle the new sample after it's created
    
    Returns
    -------
        The provided covariates and outcomes, having been over-sampled to match another
            X_os : numpy ndarray (num_sample2_units, num_sample2_features)
            y_os : numpy array (num_sample2_units,)
            w_os : numpy array (num_sample2_units,)
    """
    if len(X_1) >= sample_2_size:
        raise ValueError(
            "The sample trying to be over-sampled is the same size or greater than what it should be matched with."
            "Check sample sizes, and specifically that they haven't been switched on accident."
            )
    
    if len(X_1) != len(y_1) != len(w_1):
        raise ValueError(
            "The length of the covariates, responses, and treatments don't match."
            )
    
    new_samples_needed = sample_2_size - len(X_1)
    sample_indexes = list(range(len(X_1)))
    os_indexes = np.random.choice(sample_indexes, size=new_samples_needed, replace=True)
    
    new_sample_indexes = sample_indexes + list(os_indexes)
    
    if shuffle:
        random.shuffle(new_sample_indexes)
    
    X_os = X_1[new_sample_indexes]
    y_os = y_1[new_sample_indexes]
    w_os = w_1[new_sample_indexes]
    
    print("""
    Old Covariates shape  : {}
    Old responses shape   : {}
    Old treatments shape  : {}
    New covariates shape  : {}
    New responses shape   : {}
    New treatments shape  : {}
    Matched sample length : {}
                        """.format(X_1.shape, y_1.shape, w_1.shape,
                                   X_os.shape, y_os.shape, w_os.shape,
                                   sample_2_size))
    
    return X_os, y_os, w_os
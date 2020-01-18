# =============================================================================
# Evaluation metrics for models
# 
# Based on
# --------
#   Radcliffe N.J. & Surry, P.D. (2011). Real-World Uplift Modelling with Significance-Based Uplift Trees. 
#   Technical Report TR-2011-1, Stochastic Solutions, 2011, pp. 1-33.
#   
#   Uber.Causal ML: A Python Package for Uplift Modeling and Causal Inference with ML. (2019). 
#   URL:https://github.com/uber/causalml.
#
# Note
# ----
#   For the following:
#   If the true treatment effect is provided (e.g. in synthetic data), it's calculated
#   as the cumulative gain of the true treatment effect in each population.
#   Otherwise, it's calculated as the cumulative difference between the mean outcomes
#   of the treatment and control groups in each population.
#   For the former, `treatment_effect_col` should be provided. For the latter, both
#   `outcome_col` and `treatment_col` should be provided.
#
# Contents
# --------
#   0. No Class
#       plot
#       get_cum_effect
#       get_cum_gain
#       get_qini
#       plot_cum_effect
#       plot_cum_gain
#       plot_qini
#       auuc_score
#       qini_score
# =============================================================================

import numpy as np
import pandas as pd
import random
import seaborn as sns

RANDOM_COL = 'Random'

def plot(df, kind='gain', n=100, figsize=(15,5), fontsize=20, axis=None, *args, **kwarg):
    """
    Plots one of the causal_effect/gain/qini charts of model estimates

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns
        
        kind : str, optional (detault='gain')
            The kind of plot to draw: 'causal_effect', 'gain', and 'qini' are supported
        
        n : int, optional (detault=100)
            The number of samples to be used for plotting

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined
    """
    catalog = {'causal_effect': get_cum_effect,
               'gain': get_cum_gain,
               'qini': get_qini}

    assert kind in catalog.keys(), '{} plot is not implemented. Select one of {}'.format(kind, catalog.keys())

    # Pass one of the plot types and its arguments
    df = catalog[kind](df, *args, **kwarg)
    
    if (n is not None) and (n < df.shape[0]):
        df = df.iloc[np.linspace(start=0, stop=df.index[-1], num=n, endpoint=True)]

    # Adaptable figure sizes
    if figsize:
        sns.set(rc={'figure.figsize':figsize})
    ax = sns.distplot(df,ax=axis)
    ax.set_xlabel('Population', fontsize=fontsize)
    ax.set_ylabel('', fontsize=fontsize)
    ax.axes.set_title('{}'.format(kind.title()), fontsize=fontsize*1.5)


def get_cum_effect(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                   random_seed=42):
    """
    Gets average causal effects of model estimates in cumulative population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        effects : pandas.DataFrame 
            Average causal effects of model estimates in cumulative population
    """
    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns), """Either the outcome_col and treatment_col arguments
                                                    must be provided, or the treatment_effect_col argument"""

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [x for x in df.columns if x not in [outcome_col, treatment_col,
                                                      treatment_effect_col]]

    effects = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1

        if treatment_effect_col in df.columns:
            # Calculate average treatment effects of simulated data
            l = df[treatment_effect_col].cumsum() / df.index
        
        else:
            # Calculate average treatment effects using unit outcomes
            df['cumsum_treatment'] = df[treatment_col].cumsum()
            df['cumsum_control'] = df.index.values - df['cumsum_treatment']
            df['cumsum_y_treatment'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_control'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

            l = (df['cumsum_y_treatment'] / df['cumsum_treatment']
                - df['cumsum_y_control'] / df['cumsum_control'])

        effects.append(l)

    effects = pd.concat(effects, join='inner', axis=1)
    effects.loc[0] = np.zeros((effects.shape[1], ))
    effects = effects.sort_index().interpolate()

    effects.columns = model_names
    effects[RANDOM_COL] = effects[random_cols].mean(axis=1)
    effects.drop(random_cols, axis=1, inplace=True)

    return effects


def get_cum_gain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                normalize=False, random_seed=42):
    """
    Gets cumulative gains of model estimates in population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        normalize : bool, optional (detault='False')
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        gain : pandas.DataFrame
            Cumulative gains of model estimates in population
    """
    effects = get_cum_effect(df, outcome_col, treatment_col, treatment_effect_col, random_seed)

    # Cumulative gain = cumulative causal effect x of population
    gain = effects.mul(effects.index.values, axis=0)

    if normalize:
        gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

    return gain


def get_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
             normalize=False, random_seed=42):
    """
    Gets Qini of model estimates in population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        qini : pandas.DataFrame
            Qini of model estimates in population
    """
    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns)

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [x for x in df.columns if x not in [outcome_col, treatment_col,
                                                      treatment_effect_col]]

    qini = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df['cumsum_treatment'] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # Calculate average treatment effects of simulated data
            l = df[treatment_effect_col].cumsum() / df.index * df['cumsum_treatment']
        
        else:
            # Calculate average treatment effects using unit outcomes
            df['cumsum_control'] = df.index.values - df['cumsum_treatment']
            df['cumsum_y_treatment'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_control'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

            l = (df['cumsum_y_treatment']
                - df['cumsum_y_control'] * df['cumsum_treatment'] 
                / df['cumsum_control'])

        qini.append(l)

    qini = pd.concat(qini, join='inner', axis=1)
    qini.loc[0] = np.zeros((qini.shape[1], ))
    qini = qini.sort_index().interpolate()

    qini.columns = model_names
    qini[RANDOM_COL] = qini[random_cols].mean(axis=1)
    qini.drop(random_cols, axis=1, inplace=True)

    if normalize:
        qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

    return qini


def plot_cum_effect(df, n=100, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                    random_seed=42, figsize=None, fontsize=20, axis=None):
    """
    Plots the causal effect chart of model estimates in cumulative population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : causal_effect

        n : int, optional (detault=100)
            The number of samples to be used for plotting
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of the cumulative effect
    """
    plot(df, kind='causal_effect', n=n, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, random_seed=random_seed,
         figsize=figsize, fontsize=20, axis=None)


def plot_cum_gain(df, n=100, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, figsize=None, fontsize=20, axis=None):
    """
    Plots the cumulative gain chart (or uplift curve) of model estimates
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : gain

        n : int, optional (detault=100)
            The number of samples to be used for plotting
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of the cumulative gain
    """
    plot(df, kind='gain', n=n, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed,
         figsize=figsize, fontsize=20, axis=None)


def plot_qini(df, n=100, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, figsize=None, fontsize=20, axis=None):
    """
    Plots the Qini chart (or uplift curve) of model estimates
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : qini

        n : int, optional (detault=100)
            The number of samples to be used for plotting
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of the qini curve
    """
    plot(df, kind='qini', n=n, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed,
         figsize=figsize, fontsize=20, axis=None)


def auuc_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True):
    """
    Calculates the AUUC (Area Under the Uplift Curve) score
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            Whether to normalize the y-axis to 1 or not
    
    Returns
    -------
        AUUC score : float
    """
    cum_gain = get_cum_gain(df, outcome_col, treatment_col, treatment_effect_col, normalize)

    return cum_gain.sum() / cum_gain.shape[0]


def qini_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True):
    """
    Calculates the Qini score: the area between the Qini curve of a model and random assignment
    
    Parameters
    ----------
        df : pandas.DataFrame)
            A data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            The column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            The column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            Whether to normalize the y-axis to 1 or not
    
    Returns
    -------
        Qini score : float
    """
    qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize)

    return (qini.sum(axis=0) - qini[RANDOM_COL].sum()) / qini.shape[0]
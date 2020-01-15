# =============================================================================
# Visual evaluation metrics for models
# 
# Contents
# --------
# 0. No Class
#   plot
#   get_cumlift
#   get_cumgain
#   get_qini
#   plot_gain
#   plot_lift
#   plot_qini
#   auuc_score
#   qini_score
# =============================================================================

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_COL = 'Random'

def plot(df, kind='gain', n=100, figsize=(8, 8), *args, **kwarg):
    """
    Plots one of the lift/gain/Qini charts of model estimates.

    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        kind : str, optional (detault='gain')
            the kind of plot to draw: 'lift', 'gain', and 'qini' are supported
        
        n : int, optional (detault=100)
            the number of samples to be used for plotting
    """
    catalog = {'lift': get_cumlift,
               'gain': get_cumgain,
               'qini': get_qini}

    assert kind in catalog.keys(), '{} plot is not implemented. Select one of {}'.format(kind, catalog.keys())

    df = catalog[kind](df, *args, **kwarg)

    if (n is not None) and (n < df.shape[0]):
        df = df.iloc[np.linspace(0, df.index[-1], n, endpoint=True)]

    df.plot(figsize=figsize)
    plt.xlabel('Population')
    plt.ylabel('{}'.format(kind.title()))


def get_cumlift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                random_seed=42):
    """
    Gets average uplifts of model estimates in cumulative population.
    If the true treatment effect is provided as in synthetic data, average uplifts are
    calculated as the mean of the true treatment effect in each cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each cumulative population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        random_seed : int, optional (detault=42)
            random seed for numpy.random.rand()
    
    Returns
    -------
        Average uplifts of model estimates in cumulative population : pandas.DataFrame
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

    lift = []
    for i, col in enumerate(model_names):
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1

        if treatment_effect_col in df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population
            l = df[treatment_effect_col].cumsum() / df.index
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population
            df['cumsum_treatment'] = df[treatment_col].cumsum()
            df['cumsum_control'] = df.index.values - df['cumsum_tr']
            df['cumsum_y_treatment'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_control'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

            l = (df['cumsum_y_treatment'] / df['cumsum_treatment']
                - df['cumsum_y_control'] / df['cumsum_control'])

        lift.append(l)

    lift = pd.concat(lift, join='inner', axis=1)
    lift.loc[0] = np.zeros((lift.shape[1], ))
    lift = lift.sort_index().interpolate()

    lift.columns = model_names
    lift[RANDOM_COL] = lift[random_cols].mean(axis=1)
    lift.drop(random_cols, axis=1, inplace=True)

    return lift


def get_cumgain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                normalize=False, random_seed=42):
    """
    Gets cumulative gains of model estimates in population.
    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        normalize : bool, optional (detault='False')
            whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            random seed for numpy.random.rand()
    
    Returns
    -------
        Cumulative gains of model estimates in population : pandas.DataFrame
    """
    lift = get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, random_seed)

    # Cumulative gain = cumulative lift x (# of population)
    gain = lift.mul(lift.index.values, axis=0)

    if normalize:
        gain = gain.div(gain.iloc[-1, :], axis=1)

    return gain


def get_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
             normalize=False, random_seed=42):
    """
    Gets Qini of model estimates in population.
    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            random seed for numpy.random.rand()
    
    Returns
    -------
        Qini of model estimates in population : pandas.DataFrame
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
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population
            l = df[treatment_effect_col].cumsum() / df.index * df['cumsum_treatment']
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population
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
        qini = qini.div(qini.iloc[-1, :], axis=1)

    return qini


def plot_gain(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8)):
    """
    Plots the cumulative gain chart (or uplift curve) of model estimates.
    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            random seed for numpy.random.rand()
        
        n : int, optional (detault=100)
            the number of samples to be used for plotting
    """
    plot(df, kind='gain', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed)


def plot_lift(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              random_seed=42, n=100, figsize=(8, 8)):
    """
    Plots the lift chart of model estimates in cumulative population.
    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the mean of the true treatment effect in each of cumulative population.
    Otherwise, it's calculated as the difference between the mean outcomes of the
    treatment and control groups in each of cumulative population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        random_seed : int, optional (detault=42)
            random seed for numpy.random.rand()
        
        n : int, optional (detault=100)
            the number of samples to be used for plotting
    """
    plot(df, kind='lift', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, random_seed=random_seed)


def plot_qini(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
              normalize=False, random_seed=42, n=100, figsize=(8, 8)):
    """
    Plots the Qini chart (or uplift curve) of model estimates.
    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            random seed for numpy.random.rand()
        
        n : int, optional (detault=100)
            the number of samples to be used for plotting
    """
    plot(df, kind='qini', n=n, figsize=figsize, outcome_col=outcome_col, treatment_col=treatment_col,
         treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed)


def auuc_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True):
    """
    Calculates the AUUC (Area Under the Uplift Curve) score.
    
    Parameters
    ----------
        df : pandas.DataFrame
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            whether to normalize the y-axis to 1 or not
    
    Returns
    -------
        AUUC score : float
    """
    cumgain = get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, normalize)

    return cumgain.sum() / cumgain.shape[0]


def qini_score(df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True):
    """
    Calculates the Qini score: the area between the Qini curve of a model and random assignment.
    
    Parameters
    ----------
        df : pandas.DataFrame)
            a data frame with model estimates and actual data as columns
        
        outcome_col : str, optional (detault='y')
            the column name for the actual outcome
        
        treatment_col : str, optional (detault='w')
            the column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (detault='tau')
            the column name for the true treatment effect
        
        normalize : bool, optional (detault=False)
            whether to normalize the y-axis to 1 or not
    
    Returns
    -------
        Qini score : float
    """
    qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize)

    return (qini.sum(axis=0) - qini[RANDOM_COL].sum()) / qini.shape[0]
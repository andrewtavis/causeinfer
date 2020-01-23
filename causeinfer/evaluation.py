# =============================================================================
# Evaluation metrics for models
# 
# Based on
# --------
#   Radcliffe N.J. & Surry, P.D. (2011). Real-World Uplift Modelling with Significance-Based Uplift Trees. 
#   Technical Report TR-2011-1, Stochastic Solutions, 2011, pp. 1-33.
# 
#   Kane, K.,  Lo, VSY. & Zheng, J. (2014). Mining for the truly responsive customers and prospects using 
#   true-lift modeling: Comparison of new and existing methods. Journal of Marketing Analytics, Vol. 2, 
#   No. 4, December 2014, pp 218–238.
#   
#   Uber.Causal ML: A Python Package for Uplift Modeling and Causal Inference with ML. (2019). 
#   URL:https://github.com/uber/causalml.
#
# Note
# ----
#   For evaluation functions:
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
#       plot_eval
#       get_cum_effect
#       get_cum_gain
#       get_qini
#       plot_cum_effect
#       plot_cum_gain
#       plot_qini
#       auuc_score
#       qini_score
#       get_batch_metrics
#       plot_batch_metrics
#       plot_batch_effects
#       plot_batch_gain
#       plot_batch_qini
#       plot_batch_response
#       signal_to_noise
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

RANDOM_COL = 'random'

def plot_eval(df, kind=None, n=100, percent_of_pop=False, normalize=False, 
              figsize=(15,5), fontsize=20, axis=None, legend_metrics=None, 
              *args, **kwargs):
    """
    Plots one of the effect/gain/qini charts of model estimates

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and unit outcomes as columns
        
        kind : str, optional (detault='gain')
            The kind of plot to draw: 'effect,' 'gain,' and 'qini' are supported
        
        n : int, optional (detault=100)
            The number of samples to be used for plotting

        percent_of_pop : bool, optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population

        normalize : bool, for inheritance (default=False)
            Passes this argument to interior funcitons directly

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool, optional (default=True)
            Calculate AUUC or Qini metrics to add to the plot legend for gain and qini respectively
    """
    catalog = {'effect': get_cum_effect,
               'gain': get_cum_gain,
               'qini': get_qini}

    assert kind in catalog.keys(), '{} for plot_eval is not implemented. Select one of {}'.format(kind, list(catalog.keys()))

    # Pass one of the plot types and its arguments
    df_metrics  = catalog[kind](df=df, normalize=normalize, *args, **kwargs)

    if (n is not None) and (n < df_metrics.shape[0]):
        df_metrics = df_metrics.iloc[np.linspace(start=0, stop=df_metrics.index[-1], num=n, endpoint=True)]

    # Adaptable figure features
    if figsize:
        sns.set(rc={'figure.figsize':figsize})
            
    ax = sns.lineplot(data=df_metrics, ax=axis)
    if legend_metrics:
        if kind=='gain':
            metric_label = 'auuc'
            metrics = auuc_score(df=df, normalize=normalize, *args, **kwargs)
        elif kind=='qini':
            metric_label = 'qini'
            metrics = qini_score(df=df, normalize=normalize, *args, **kwargs)
        elif kind=='effect':
            print("Display metrics are AUUC or Qini, and are thus not supported for Incremental Effect Plots.")
            print("The plot will be done without them.")
            legend_metrics = False # Turn off for next line
            
    if legend_metrics:
        metric_labels = ['{}: {:.4f}'.format(metric_label, m) for m in metrics]
        metric_labels[0] = '' # Random column
        new_labels = list(df_metrics.columns) + metric_labels
        ax.legend(title='Models', labels=new_labels, ncol=2)
    else:
        ax.legend(title='Models')
    
    plot_x_label = 'Population Targeted'
    if percent_of_pop:
        plot_x_label += ' (%)'
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=df.shape[0]))
    ax.set_xlabel(plot_x_label, fontsize=fontsize)
    
    ax.set_ylabel('Cumulative Incremental Change', fontsize=fontsize)
    
    plot_title = 'Incremental {}'.format(kind.title())
    if normalize:
        plot_title += ' (Normalized)'
    ax.axes.set_title(plot_title, fontsize=fontsize*1.5)


def get_cum_effect(df, model_pred_cols=None, outcome_col='y', treatment_col='w', 
                   treatment_effect_col='tau', normalize=False, random_seed=42):
    """
    Gets average causal effects of model estimates in cumulative population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        model_pred_cols : list
            A list of columns with model estimated treatment effects
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect

        normalize : bool, not implemented (default=False)
            For consitency with gaina and qini
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        effects : pandas.DataFrame 
            Average causal effects of model estimates in cumulative population
    """
    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns), "Either the outcome_col and treatment_col arguments must be provided, or the treatment_effect_col argument"

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        # Generate random values in (0,1] to compare against on average
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    if type(model_pred_cols) == str:
        model_pred_cols = [model_pred_cols]
    model_and_random_preds = [x for x in df.columns if x in model_pred_cols + random_cols]

    effects = []
    for col in model_and_random_preds:
        # Sort by model estimates, and get the cumulateive sum of treatment along the new sorted axis
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df['cumsum_treatment'] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # Calculate iterated average treatment effects of simulated data
            iterated_effect = df[treatment_effect_col].cumsum() / df.index
        
        else:
            # Calculate iterated average treatment effects using unit outcomes
            df['cumsum_control'] = df.index.values - df['cumsum_treatment']
            df['cumsum_y_treatment'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_control'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

            iterated_effect = (df['cumsum_y_treatment'] / df['cumsum_treatment']
                            - df['cumsum_y_control'] / df['cumsum_control'])

        effects.append(iterated_effect)

    effects = pd.concat(effects, join='inner', axis=1)
    effects.loc[0] = np.zeros((effects.shape[1], )) # start from 0
    effects = effects.sort_index().interpolate()

    effects.columns = model_and_random_preds
    effects[RANDOM_COL] = effects[random_cols].mean(axis=1)
    effects.drop(random_cols, axis=1, inplace=True)
    cols = effects.columns.tolist()
    cols.insert(0, cols.pop(cols.index(RANDOM_COL)))
    effects = effects.reindex(columns=cols)

    return effects


def get_cum_gain(df, model_pred_cols=None, outcome_col='y', treatment_col='w', 
                 treatment_effect_col='tau', normalize=False, random_seed=42):
    """
    Gets cumulative gains of model estimates in population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        model_pred_cols : list
            A list of columns with model estimated treatment effects
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        normalize : bool, optional (default=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        gains : pandas.DataFrame
            Cumulative gains of model estimates in population
    """
    effects = get_cum_effect(df=df, model_pred_cols=model_pred_cols, 
                             outcome_col = outcome_col, treatment_col=treatment_col, 
                             treatment_effect_col = treatment_effect_col, random_seed=random_seed)

    # Cumulative gain is the cumulative causal effect of the population
    gains = effects.mul(effects.index.values, axis=0)

    if normalize:
        gains = gains.div(np.abs(gains.iloc[-1, :]), axis=1)

    return gains


def get_qini(df, model_pred_cols=None, outcome_col='y', treatment_col='w', 
             treatment_effect_col='tau', normalize=False, random_seed=42):
    """
    Gets Qini of model estimates in population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        model_pred_cols : list
            A list of columns with model estimated treatment effects
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        normalize : bool, optional (default=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        qinis : pandas.DataFrame
            Qini of model estimates in population
    """
    assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
            treatment_effect_col in df.columns), "Either the outcome_col and treatment_col arguments must be provided, or the treatment_effect_col argument"

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = '__random_{}__'.format(i)
        # Generate random values in (0,1] to compare against on average
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    if type(model_pred_cols) == str:
        model_pred_cols = [model_pred_cols]
    model_and_random_preds = [x for x in df.columns if x in model_pred_cols + random_cols]

    qinis = []
    for col in model_and_random_preds:
        # Sort by model estimates, and get the cumulateive sum of treatment along the new sorted axis
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df['cumsum_treatment'] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # Calculate iterated average treatment effects of simulated data
            iterated_effect = df[treatment_effect_col].cumsum() / df.index * df['cumsum_treatment']
        
        else:
            # Calculate iterated verage treatment effects using unit outcomes
            df['cumsum_control'] = df.index.values - df['cumsum_treatment']
            df['cumsum_y_treatment'] = (df[outcome_col] * df[treatment_col]).cumsum()
            df['cumsum_y_control'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

            iterated_effect = (df['cumsum_y_treatment']
                            - df['cumsum_y_control'] 
                            * df['cumsum_treatment'] / df['cumsum_control'])

        qinis.append(iterated_effect)

    qinis = pd.concat(qinis, join='inner', axis=1)
    qinis.loc[0] = np.zeros((qinis.shape[1], )) # start from 0
    qinis = qinis.sort_index().interpolate()

    qinis.columns = model_and_random_preds
    qinis[RANDOM_COL] = qinis[random_cols].mean(axis=1)
    qinis.drop(random_cols, axis=1, inplace=True)
    cols = qinis.columns.tolist()
    cols.insert(0, cols.pop(cols.index(RANDOM_COL)))
    qinis = qinis.reindex(columns=cols)

    if normalize:
        qinis = qinis.div(np.abs(qinis.iloc[-1, :]), axis=1)

    return qinis


def plot_cum_effect(df, n=100, model_pred_cols=None, percent_of_pop=False, 
                    outcome_col='y', treatment_col='w', 
                    treatment_effect_col='tau', random_seed=42, 
                    figsize=None, fontsize=20, axis=None, legend_metrics=None):
    """
    Plots the causal effect chart of model estimates in cumulative population
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : effect

        n : int, optional (detault=100)
            The number of samples to be used for plotting

        model_pred_cols : list
            A list of columns with model estimated treatment effects

        percent_of_pop : bool, optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool, optional (default=False)
            Not supported for plot_cum_effect - the user will be notified

    Returns
    -------
        A plot of the cumulative effects of all models in df
    """
    plot_eval(df=df, kind='effect', n=n, model_pred_cols=model_pred_cols, percent_of_pop=percent_of_pop, 
              outcome_col=outcome_col, treatment_col=treatment_col,
              treatment_effect_col=treatment_effect_col, random_seed=random_seed,
              figsize=figsize, fontsize=20, axis=axis, legend_metrics=legend_metrics)


def plot_cum_gain(df, n=100, model_pred_cols=None, percent_of_pop=False,
                  outcome_col='y', treatment_col='w', 
                  treatment_effect_col='tau', normalize=False, random_seed=42, 
                  figsize=None, fontsize=20, axis=None, legend_metrics=True):
    """
    Plots the cumulative gain chart (or uplift curve) of model estimates
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : gain

        n : int, optional (detault=100)
            The number of samples to be used for plotting

        model_pred_cols : list
            A list of columns with model estimated treatment effects

        percent_of_pop : bool, optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        normalize : bool, optional (default=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool, optional (default=True)
            Calculates AUUC metrics to add to the plot legend

    Returns
    -------
        A plot of the cumulative gains of all models in df
    """
    plot_eval(df=df, kind='gain', n=n, model_pred_cols=model_pred_cols, percent_of_pop=percent_of_pop,
              outcome_col=outcome_col, treatment_col=treatment_col,
              treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed,
              figsize=figsize, fontsize=20, axis=axis, legend_metrics=legend_metrics)


def plot_qini(df, n=100, model_pred_cols=None, percent_of_pop=False,
              outcome_col='y', treatment_col='w', 
              treatment_effect_col='tau', normalize=False, random_seed=42, 
              figsize=None, fontsize=20, axis=None, legend_metrics=True):
    """
    Plots the Qini chart (or uplift curve) of model estimates
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : qini

        n : int, optional (detault=100)
            The number of samples to be used for plotting

        model_pred_cols : list
            A list of columns with model estimated treatment effects

        percent_of_pop : bool, optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        normalize : bool, optional (default=False)
            Whether to normalize the y-axis to 1 or not
        
        random_seed : int, optional (detault=42)
            Random seed for numpy.random.rand()

        figsize : tuple, optional
            Allows for quick changes of figures sizes

        fontsize : int or float, optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str, optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool, optional (default=True)
            Calculates Qini metrics to add to the plot legend

    Returns
    -------
        A plot of the qini curves of all models in df
    """
    plot_eval(df=df, kind='qini', n=n, model_pred_cols=model_pred_cols, percent_of_pop=percent_of_pop, 
              outcome_col=outcome_col, treatment_col=treatment_col,
              treatment_effect_col=treatment_effect_col, normalize=normalize, random_seed=random_seed,
              figsize=figsize, fontsize=20, axis=axis, legend_metrics=legend_metrics)


def auuc_score(df, model_pred_cols=None, 
               outcome_col='y', treatment_col='w', 
               treatment_effect_col='tau', normalize=True, random_seed=None):
    """
    Calculates the AUUC score: the Area Under the Uplift Curve
    
    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        model_pred_cols : list
            A list of columns with model estimated treatment effects
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        normalize : bool, optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, for inheritance (default=None)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        AUUC score : float
    """
    gains = get_cum_gain(df=df, model_pred_cols=model_pred_cols, 
                         outcome_col=outcome_col, treatment_col=treatment_col, 
                         treatment_effect_col=treatment_effect_col, normalize=normalize)
    
    return gains.sum() / gains.shape[0]


def qini_score(df, model_pred_cols=None, 
               outcome_col='y', treatment_col='w', 
               treatment_effect_col='tau', normalize=True, random_seed=None):
    """
    Calculates the Qini score: the area between the Qini curve of a model and random assignment
    
    Parameters
    ----------
        df : pandas.DataFrame)
            A data frame with model estimates and actual data as columns

        model_pred_cols : list
            A list of columns with model estimated treatment effects
        
        outcome_col : str, optional (default=y)
            The column name for the actual outcome
        
        treatment_col : str, optional (default=w)
            The column name for the treatment indicator (0 or 1)
        
        treatment_effect_col : str, optional (default=tau)
            The column name for the true treatment effect
        
        normalize : bool, optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, for inheritance (default=None)
            Random seed for numpy.random.rand()
    
    Returns
    -------
        Qini score : float
    """
    qinis = get_qini(df=df, model_pred_cols=model_pred_cols, 
                     outcome_col=outcome_col, treatment_col=treatment_col, 
                     treatment_effect_col=treatment_effect_col, normalize=normalize)

    return (qinis.sum(axis=0) - qinis[RANDOM_COL].sum()) / qinis.shape[0]


def get_batches(df, n=10, model_pred_cols=None,
                outcome_col='y', treatment_col='w'):
    """
    Calculates the cumulative causal effects of models given batches from ranked treatment effects

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns
    
    n : int, optional (detault=10, deciles; 5, quintiles also standard)
        The number of batches to split the units into

    model_pred_cols : list
        A list of columns with model estimated treatment effects

    outcome_col : str, optional (default=y)
        The column name for the actual outcome
        
    treatment_col : str, optional (default=w)
        The column name for the treatment indicator (0 or 1)

    Returns
    -------
        df : pandas.DataFrame
            The original dataframe with columns for model rank batches given n
    """
    assert np.isin(df[outcome_col].unique(), [0, 1]).all() and np.isin(df[treatment_col].unique(), [0, 1]).all(), "Batch metrics are currently only available for numeric-binary outcomes."
    
    model_preds = [x for x in df.columns if x in model_pred_cols]
    
    for col in model_preds:
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df_batches = np.array_split(df, n)
        # Get sublists of the length of the batch filled with the batch indexes
        sublist_of_batch_indexes = [[i+1 for j in range(len(b))]for i, b in enumerate(df_batches)]
        # Assign batches to units
        df['{}_batches'.format(col)] = [val for sublist in sublist_of_batch_indexes for val in sublist]

    return df


def plot_batch_metrics(df, kind=None, n=10, model_pred_cols=None, 
                       outcome_col='y', treatment_col='w', normalize=False,
                       figsize=(15,5), fontsize=20, axis=None,
                       *args, **kwargs):
    """
    Plots the batch chart: the cumulative batch metrics predicted by a model given ranked treatment effects

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    kind : str, optional (detault='gain')
        The kind of plot to draw: 'effect,' 'gain,' 'qini,' and 'response' are supported
    
    n : int, optional (detault=10, deciles; 20, quintiles also standard)
        The number of batches to split the units into

    model_pred_cols : list
        A list of columns with model estimated treatment effects

    outcome_col : str, optional (default=y)
        The column name for the actual outcome
        
    treatment_col : str, optional (default=w)
        The column name for the treatment indicator (0 or 1)

    figsize : tuple, optional
        Allows for quick changes of figures sizes

    fontsize : int or float, optional (default=20)
        The font size of the plots, with all labels scaled accordingly

    axis : str, optional (default=None)
        Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of batch matrics of all models in df
    """
    catalog = {'effect': get_cum_effect,
               'gain': get_cum_gain,
               'qini': get_qini,
               'response': None}

    assert kind in catalog.keys(), '{} for plot_batch_metrics is not implemented. Select one of {}'.format(kind, list(catalog.keys()))

    df_batches = get_batches(df=df, n=n, model_pred_cols=model_pred_cols, 
                             outcome_col=outcome_col, treatment_col=treatment_col)
    
    df_batch_metrics = pd.DataFrame()
    if kind in ['effect', 'gain', 'qini']:
        batch_columns = ['{}_batches'.format(col) for col in model_pred_cols]
        for i in range(n):
            i+=1
            batch_metrics = catalog[kind](df=df_batches[df_batches[batch_columns]==i], 
                                          model_pred_cols=model_pred_cols, 
                                          outcome_col=outcome_col, treatment_col=treatment_col, 
                                          normalize=normalize, *args, **kwargs)
            if kind == 'effect':
                color_palette = 'Set1'
                # Select last row, the cumsum effect for the batch
                sum_batch_metrics = batch_metrics.iloc[-1,:]
            
            elif kind == 'gain':
                color_palette = 'Set2'
                # Cumulative gain is the cumulative causal effect of the population
                batch_metrics = batch_metrics.mul(batch_metrics.index.values, axis=0)

                if normalize:
                    batch_metrics = batch_metrics.div(np.abs(batch_metrics.iloc[-1, :]), axis=1)

                sum_batch_metrics = batch_metrics.sum() / batch_metrics.shape[0]
                # Make a df and transpose to a row so it can be appended
                sum_batch_metrics = pd.DataFrame(sum_batch_metrics).T
                sum_batch_metrics.columns = batch_metrics.columns

            elif kind == 'qini':
                color_palette = 'Set3'
                sum_batch_metrics = (batch_metrics.sum(axis=0) - batch_metrics[RANDOM_COL].sum()) / batch_metrics.shape[0]
                # Make a df and transpose to a row so it can be appended
                sum_batch_metrics = pd.DataFrame(sum_batch_metrics).T
                sum_batch_metrics.columns = batch_metrics.columns
            
            # Select only the model columns and append
            sum_batch_metrics = sum_batch_metrics[model_pred_cols]
            df_batch_metrics = df_batch_metrics.append(sum_batch_metrics)

    elif kind == 'response':
        color_palette = sns.color_palette()
        for model in model_pred_cols:
            # Total in-batch known class amounts (potentially add in TN and CN units later)
            df_batch_metrics['{}_tp'.format(model)] = df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 1) & (x[outcome_col] == 1)]))
            df_batch_metrics['{}_cp'.format(model)] = df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 0) & (x[outcome_col] == 1)]))
            df_batch_metrics['{}_cn'.format(model)] = df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 0) & (x[outcome_col] == 0)]))
            df_batch_metrics['{}_tn'.format(model)] = df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 1) & (x[outcome_col] == 0)]))

            # The ratios of known unit classes to the number of treatment and control units per batch
            df_batch_metrics['{}_tp_rat'.format(model)] = df_batch_metrics['{}_tp'.format(model)] \
                                                        / df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 1)]))
            df_batch_metrics['{}_cp_rat'.format(model)] = df_batch_metrics['{}_cp'.format(model)] \
                                                        / df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 0)]))
            df_batch_metrics['{}_cn_rat'.format(model)] = df_batch_metrics['{}_cn'.format(model)] \
                                                        / df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 0)]))
            df_batch_metrics['{}_tn_rat'.format(model)] = df_batch_metrics['{}_tn'.format(model)] \
                                                        / df_batches.groupby(by='{}_batches'.format(model)).apply(lambda x: len(x[(x[treatment_col] == 1)]))
            # df_batch_metrics[model] = df_batch_metrics['{}_tp_rat'.format(model)] \
            #                         - df_batch_metrics['{}_cp_rat'.format(model)]
            
            df_batch_metrics[model] = df_batch_metrics['{}_tp_rat'.format(model)] + df_batch_metrics['{}_cn_rat'.format(model)] \
                                    - df_batch_metrics['{}_cp_rat'.format(model)] - df_batch_metrics['{}_tn_rat'.format(model)]

    df_plot = pd.DataFrame()
    batches_per_model = [[i+1 for j in range(len(model_pred_cols))]for i in range(n)]
    df_plot['batch'] = [val for sublist in batches_per_model for val in sublist]
    
    metrics_per_batch = [[val for val in df_batch_metrics[col]] for col in model_pred_cols]
    df_plot['metric'] = [val for sublist in metrics_per_batch for val in sublist]
    
    models_per_batch = [[i for i in model_pred_cols] for i in range(n)]
    df_plot['model'] = [val for sublist in models_per_batch for val in sublist]

    # Adaptable figure features
    if figsize:
        sns.set(rc={'figure.figsize':figsize})

    ax = sns.barplot(data=df_plot, x='batch', y='metric', hue='model', palette=color_palette, ax=axis)
    plot_x_label = 'Batches'
    number_to_quantile = [(3, 'Tertiles'), (4, 'Quartiles'), (5, 'Quintiles'), (6, 'Sextiles'), 
                          (7, 'Septiles'), (8, 'Octiles'), (10, 'Deciles'), (20, 'Ventiles'), (100, 'Percentiles')]
    for i in number_to_quantile:
        if n == i[0]:
            plot_x_label = i[1]
    ax.set_xlabel(plot_x_label, fontsize=fontsize)
    
    kind_to_y_label = [('effect', 'Causal Effect'), ('gain','Gain (AUUC)'), 
                       ('qini','Qini'), ('response','Response Difference')]
    for i in kind_to_y_label:
        if kind == i[0]:
            plot_y_label = i[1]
    ax.set_ylabel(plot_y_label, fontsize=fontsize)
    
    kind_to_title = [('effect', 'Batch Causal Effects'), ('gain','Batch Gain (AUUC)'), 
                       ('qini','Batch Qini'), ('response','Batch Response Differences (TP+CN-CP-TN)')]
    for i in kind_to_title:
        if kind == i[0]:
            plot_title = i[1]
    ax.axes.set_title(plot_title, fontsize=fontsize*1.5)


def plot_batch_effects(df):
    plot_batch_metrics(df, kind='effects')


def plot_batch_gain(df):
    plot_batch_metrics(df, kind='gain')


def plot_batch_qini(df):
    plot_batch_metrics(df, kind='qini')


def plot_batch_reponse(df):
    plot_batch_metrics(df, kind='response')


def signal_to_noise(y, w):
    """
    Computes the signal to noise ratio of a dataset to derive the potential for causal inference efficacy
        - The signal to noise ratio is the difference in treatment and control response to the control response
        - Values close to 0 imply that CI would have little benefit over predictive modeling

    Parameters
    ----------
        y : numpy array (num_units,) : int, float
            Vector of unit reponses

        w : numpy array (num_units,) : int, float
            Vector of original treatment allocations across units
    
    Returns
    -------
        sn_ratio : float
    """
    y_treatment = [a*b for a,b in zip(y,w)]
    y_control = [a*(1-b) for a,b in zip(y,w)]
    y_treatment_sum = np.sum(y_treatment)
    y_control_sum = np.sum(y_control)

    sn_ratio = (y_treatment_sum - y_control_sum) / y_control_sum

    return sn_ratio
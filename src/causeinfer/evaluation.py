"""
Evaluation
----------

Evaluation metrics and plotting techniques for models.

Based on
    Uber.Causal ML: A Python Package for Uplift Modeling and Causal Inference with ML. (2019).
    URL:https://github.com/uber/causalml.

    Radcliffe N.J. & Surry, P.D. (2011). Real-World Uplift Modelling with Significance-Based Uplift Trees.
    Technical Report TR-2011-1, Stochastic Solutions, 2011, pp. 1-33.

    Kane, K.,  Lo, VSY. & Zheng, J. (2014). Mining for the truly responsive customers and prospects using
    true-lift modeling: Comparison of new and existing methods. Journal of Marketing Analytics, Vol. 2,
    No. 4, December 2014, pp 218–238.

    Sołtys, M., Jaroszewicz, S. & Rzepakowski, P. (2015). Ensemble methods for uplift modeling.
    Data Mining and Knowledge Discovery, Vol. 29, No. 6, November 2015,  pp. 1531–1559.

Note
    For evaluation functions:
    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.
    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

Contents
    plot_eval,
    get_cum_effect,
    get_cum_gain,
    get_qini,
    plot_cum_effect,
    plot_cum_gain,
    plot_qini,
    auuc_score,
    qini_score,
    get_batch_metrics,
    plot_batch_metrics,
    plot_batch_effects (WIP),
    plot_batch_gains (WIP),
    plot_batch_qinis (WIP),
    plot_batch_responses,
    signal_to_noise,
    iterate_model,
    eval_table
"""

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

RANDOM_COL = "random"


def plot_eval(
    df,
    kind=None,
    n=100,
    percent_of_pop=False,
    normalize=False,
    figsize=(15, 5),
    fontsize=20,
    axis=None,
    legend_metrics=None,
    *args,
    **kwargs,
):
    """
    Plots one of the effect/gain/qini charts of model estimates.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and unit outcomes as columns

        kind : str : optional (default='gain')
            The kind of plot to draw: 'effect,' 'gain,' and 'qini' are supported

        n : int, optional (default=100)
            The number of samples to be used for plotting

        percent_of_pop : bool : optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population

        normalize : bool : for inheritance (default=False)
            Passes this argument to interior functions directly

        figsize : tuple : optional
            Allows for quick changes of figures sizes

        fontsize : int or float : optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str : optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool : optional (default=True)
            Calculate AUUC or Qini metrics to add to the plot legend for gain and qini respectively
    """
    # Add ability to have straight random targeting line.
    catalog = {"effect": get_cum_effect, "gain": get_cum_gain, "qini": get_qini}

    assert (
        kind in catalog
    ), "{} for plot_eval is not implemented. Select one of {}".format(
        kind, list(catalog.keys())
    )

    # Pass one of the plot types and its arguments
    df_metrics = catalog[kind](df=df, normalize=normalize, *args, **kwargs)

    if (n is not None) and (n < df_metrics.shape[0]):
        df_metrics = df_metrics.iloc[
            np.linspace(start=0, stop=df_metrics.index[-1], num=n, endpoint=True)
        ]

    # Adaptable figure features
    if figsize:
        sns.set(rc={"figure.figsize": figsize})

    # Shifts the color palette such that models are the same color across line and batch plots
    # Random line is the first in line plots, such that it's solid
    sns.set_palette("deep")  # Default
    color_palette = sns.color_palette()
    color_palette.insert(0, color_palette.pop())
    sns.set_palette(color_palette)

    ax = sns.lineplot(data=df_metrics, ax=axis)
    if legend_metrics:
        if kind == "gain":
            metric_label = "auuc"
            metrics = auuc_score(df=df, normalize=normalize, *args, **kwargs)
        elif kind == "qini":
            metric_label = "qini"
            metrics = qini_score(df=df, normalize=normalize, *args, **kwargs)
        elif kind == "effect":
            print(
                "Display metrics are AUUC or Qini, and are thus not supported for Incremental Effect Plots."
            )
            print("The plot will be done without them.")
            legend_metrics = False  # Turn off for next line

    if legend_metrics:
        metric_labels = ["{}: {:.4f}".format(metric_label, m) for m in metrics]
        metric_labels[0] = ""  # Random column
        new_labels = list(df_metrics.columns) + metric_labels
        ax.legend(title="Models", labels=new_labels, ncol=2)
    else:
        ax.legend(title="Models")

    plot_x_label = "Population Targeted"
    if percent_of_pop:
        plot_x_label += " (%)"
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=df.shape[0]))
    ax.set_xlabel(plot_x_label, fontsize=fontsize)

    ax.set_ylabel("Cumulative Incremental Change", fontsize=fontsize)

    plot_title = "Incremental {}".format(kind.title())
    if normalize and kind in ["gain", "qini"]:
        plot_title += " (Normalized)"

    ax.axes.set_title(plot_title, fontsize=fontsize * 1.5)


def get_cum_effect(
    df,
    models=None,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
):
    """
    Gets average causal effects of model estimates in cumulative population.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        models : list
            A list of models corresponding to estimated treatment effect columns

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : not implemented (default=False)
            For consitency with gain and qini

        random_seed : int, optional (default=None)
            Random seed for numpy.random.rand()

    Returns
    -------
        effects : pandas.DataFrame
            Average causal effects of model estimates in cumulative population
    """
    assert (
        (outcome_col in df.columns)
        and (treatment_col in df.columns)
        or treatment_effect_col in df.columns
    ), "Either the outcome_col and treatment_col arguments must be provided, or the treatment_effect_col argument"

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = "__random_{}__".format(i)
        # Generate random values in (0,1] to compare against on average
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    if isinstance(models, str):
        models = [models]
    model_and_random_preds = [x for x in df.columns if x in models + random_cols]

    effects = []
    for col in model_and_random_preds:
        # Sort by model estimates, and get the cumulative sum of treatment along the new sorted axis
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df["cumsum_treatment"] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # Calculate iterated average treatment effects of simulated data
            iterated_effect = df[treatment_effect_col].cumsum() / df.index

        else:
            # Calculate iterated average treatment effects using unit outcomes
            df["cumsum_control"] = df.index.values - df["cumsum_treatment"]
            df["cumsum_y_treatment"] = (df[outcome_col] * df[treatment_col]).cumsum()
            df["cumsum_y_control"] = (
                df[outcome_col] * (1 - df[treatment_col])
            ).cumsum()

            iterated_effect = (
                df["cumsum_y_treatment"] / df["cumsum_treatment"]
                - df["cumsum_y_control"] / df["cumsum_control"]
            )

        effects.append(iterated_effect)

    effects = pd.concat(effects, join="inner", axis=1)
    effects.loc[0] = np.zeros((effects.shape[1],))  # start from 0
    effects = effects.sort_index().interpolate()

    effects.columns = model_and_random_preds
    effects[RANDOM_COL] = effects[random_cols].mean(axis=1)
    effects.drop(random_cols, axis=1, inplace=True)
    cols = effects.columns.tolist()
    cols.insert(0, cols.pop(cols.index(RANDOM_COL)))
    effects = effects.reindex(columns=cols)

    return effects


def get_cum_gain(
    df,
    models=None,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
):
    """
    Gets cumulative gains of model estimates in population.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        models : list
            A list of models corresponding to estimated treatment effect columns

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, optional (default=None)
            Random seed for numpy.random.rand()

    Returns
    -------
        gains : pandas.DataFrame
            Cumulative gains of model estimates in population
    """
    effects = get_cum_effect(
        df=df,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        random_seed=random_seed,
    )

    # Cumulative gain is the cumulative causal effect of the population
    gains = effects.mul(effects.index.values, axis=0)

    if normalize:
        gains = gains.div(np.abs(gains.iloc[-1, :]), axis=1)

    return gains


def get_qini(
    df,
    models=None,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
):
    """
    Gets Qini of model estimates in population.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        models : list
            A list of models corresponding to estimated treatment effect columns

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, optional (default=None)
            Random seed for numpy.random.rand()

    Returns
    -------
        qinis : pandas.DataFrame
            Qini of model estimates in population
    """
    assert (
        (outcome_col in df.columns)
        and (treatment_col in df.columns)
        or treatment_effect_col in df.columns
    ), "Either the outcome_col and treatment_col arguments must be provided, or the treatment_effect_col argument"

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = "__random_{}__".format(i)
        # Generate random values in (0,1] to compare against on average
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    if isinstance(models, str):
        models = [models]
    model_and_random_preds = [x for x in df.columns if x in models + random_cols]

    qinis = []
    for col in model_and_random_preds:
        # Sort by model estimates, and get the cumulateive sum of treatment along the new sorted axis
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df["cumsum_treatment"] = df[treatment_col].cumsum()

        if treatment_effect_col in df.columns:
            # Calculate iterated average treatment effects of simulated data
            iterated_effect = (
                df[treatment_effect_col].cumsum() / df.index * df["cumsum_treatment"]
            )

        else:
            # Calculate iterated average treatment effects using unit outcomes
            df["cumsum_control"] = df.index.values - df["cumsum_treatment"]
            df["cumsum_y_treatment"] = (df[outcome_col] * df[treatment_col]).cumsum()
            df["cumsum_y_control"] = (
                df[outcome_col] * (1 - df[treatment_col])
            ).cumsum()

            iterated_effect = (
                df["cumsum_y_treatment"]
                - df["cumsum_y_control"] * df["cumsum_treatment"] / df["cumsum_control"]
            )

        qinis.append(iterated_effect)

    qinis = pd.concat(qinis, join="inner", axis=1)
    qinis.loc[0] = np.zeros((qinis.shape[1],))  # start from 0
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


def plot_cum_effect(
    df,
    n=100,
    models=None,
    percent_of_pop=False,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    random_seed=None,
    figsize=None,
    fontsize=20,
    axis=None,
    legend_metrics=None,
):
    """
    Plots the causal effect chart of model estimates in cumulative population.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : effect

        n : int, optional (default=100)
            The number of samples to be used for plotting

        models : list
            A list of models corresponding to estimated treatment effect columns

        percent_of_pop : bool : optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        random_seed : int, optional (default=None)
            Random seed for numpy.random.rand()

        figsize : tuple : optional
            Allows for quick changes of figures sizes

        fontsize : int or float : optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str : optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool : optional (default=False)
            Not supported for plot_cum_effect - the user will be notified

    Returns
    -------
        A plot of the cumulative effects of all models in df
    """
    plot_eval(
        df=df,
        kind="effect",
        n=n,
        models=models,
        percent_of_pop=percent_of_pop,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        random_seed=random_seed,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
        legend_metrics=legend_metrics,
    )


def plot_cum_gain(
    df,
    n=100,
    models=None,
    percent_of_pop=False,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
    figsize=None,
    fontsize=20,
    axis=None,
    legend_metrics=True,
):
    """
    Plots the cumulative gain chart (or uplift curve) of model estimates.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : gain

        n : int, optional (default=100)
            The number of samples to be used for plotting

        models : list
            A list of models corresponding to estimated treatment effect columns

        percent_of_pop : bool : optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, optional (default=None)
            Random seed for numpy.random.rand()

        figsize : tuple : optional
            Allows for quick changes of figures sizes

        fontsize : int or float : optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str : optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool : optional (default=True)
            Calculates AUUC metrics to add to the plot legend

    Returns
    -------
        A plot of the cumulative gains of all models in df
    """
    plot_eval(
        df=df,
        kind="gain",
        n=n,
        models=models,
        percent_of_pop=percent_of_pop,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
        random_seed=random_seed,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
        legend_metrics=legend_metrics,
    )


def plot_qini(
    df,
    n=100,
    models=None,
    percent_of_pop=False,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
    figsize=None,
    fontsize=20,
    axis=None,
    legend_metrics=True,
):
    """
    Plots the Qini chart (or uplift curve) of model estimates.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        kind : qini

        n : int, optional (default=100)
            The number of samples to be used for plotting

        models : list
            A list of models corresponding to estimated treatment effect columns

        percent_of_pop : bool : optional (default=False)
            Whether the X-axis is displayed as a percent of the whole population

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, optional (default=None)
            Random seed for numpy.random.rand()

        figsize : tuple : optional
            Allows for quick changes of figures sizes

        fontsize : int or float : optional (default=20)
            The font size of the plots, with all labels scaled accordingly

        axis : str : optional (default=None)
            Adds an axis to the plot so they can be combined

        legend_metrics : bool : optional (default=True)
            Calculates Qini metrics to add to the plot legend

    Returns
    -------
        A plot of the qini curves of all models in df
    """
    plot_eval(
        df=df,
        kind="qini",
        n=n,
        models=models,
        percent_of_pop=percent_of_pop,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
        random_seed=random_seed,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
        legend_metrics=legend_metrics,
    )


def auuc_score(
    df,
    models=None,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
):
    """
    Calculates the AUUC score (Gini): the Area Under the Uplift Curve.

    Parameters
    ----------
        df : pandas.DataFrame
            A data frame with model estimates and actual data as columns

        models : list
            A list of models corresponding to estimated treatment effect columns

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, for inheritance (default=None)
            Random seed for numpy.random.rand()

    Returns
    -------
        AUUC score : float
    """
    gains = get_cum_gain(
        df=df,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
    )

    return gains.sum() / gains.shape[0]


def qini_score(
    df,
    models=None,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=None,
):
    """
    Calculates the Qini score: the area between the Qini curve of a model and random assignment.

    Parameters
    ----------
        df : pandas.DataFrame)
            A data frame with model estimates and actual data as columns

        models : list
            A list of models corresponding to estimated treatment effect columns

        outcome_col : str : optional (default=y)
            The column name for the actual outcome

        treatment_col : str : optional (default=w)
            The column name for the treatment indicator (0 or 1)

        treatment_effect_col : str : optional (default=tau)
            The column name for the true treatment effect

        normalize : bool : optional (default=False)
            Whether to normalize the y-axis to 1 or not

        random_seed : int, for inheritance (default=None)
            Random seed for numpy.random.rand()

    Returns
    -------
        Qini score : float
    """
    qinis = get_qini(
        df=df,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
        normalize=normalize,
    )

    return (qinis.sum(axis=0) - qinis[RANDOM_COL].sum()) / qinis.shape[0]


def get_batches(df, n=10, models=None, outcome_col="y", treatment_col="w"):
    """
    Calculates the cumulative causal effects of models given batches from ranked treatment effects.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    n : int, optional (default=10, deciles; 5, quintiles also standard)
        The number of batches to split the units into

    models : list
        A list of models corresponding to estimated treatment effect columns

    outcome_col : str : optional (default=y)
        The column name for the actual outcome

    treatment_col : str : optional (default=w)
        The column name for the treatment indicator (0 or 1)

    Returns
    -------
        df : pandas.DataFrame
            The original dataframe with columns for model rank batches given n
    """
    assert (
        np.isin(df[outcome_col].unique(), [0, 1]).all()
        and np.isin(df[treatment_col].unique(), [0, 1]).all()
    ), "Batch metrics are currently only available for numeric-binary outcomes."

    model_preds = [x for x in df.columns if x in models]

    for col in model_preds:
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df_batches = np.array_split(df, n)
        # Get sublists of the length of the batch filled with the batch indexes
        sublist_of_batch_indexes = [
            [i + 1 for j in range(len(b))] for i, b in enumerate(df_batches)
        ]
        # Assign batches to units
        df["{}_batches".format(col)] = [
            val for sublist in sublist_of_batch_indexes for val in sublist
        ]

    return df


def plot_batch_metrics(
    df,
    kind=None,
    n=10,
    models=None,
    outcome_col="y",
    treatment_col="w",
    normalize=False,
    figsize=(15, 5),
    fontsize=20,
    axis=None,
    *args,
    **kwargs,
):
    """
    Plots the batch chart: the cumulative batch metrics predicted by a model given ranked treatment effects.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    kind : str : optional (default='gain')
        The kind of plot to draw: 'effect,' 'gain,' 'qini,' and 'response' are supported

    n : int, optional (default=10, deciles; 20, quintiles also standard)
        The number of batches to split the units into

    models : list
        A list of models corresponding to estimated treatment effect columns

    outcome_col : str : optional (default=y)
        The column name for the actual outcome

    treatment_col : str : optional (default=w)
        The column name for the treatment indicator (0 or 1)

    figsize : tuple : optional
        Allows for quick changes of figures sizes

    fontsize : int or float : optional (default=20)
        The font size of the plots, with all labels scaled accordingly

    axis : str : optional (default=None)
        Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of batch metrics of all models in df
    """
    catalog = {
        "effect": get_cum_effect,
        "gain": get_cum_gain,
        "qini": get_qini,
        "response": None,
    }

    assert (
        kind in catalog
    ), "{} for plot_batch_metrics is not implemented. Select one of {}".format(
        kind, list(catalog.keys())
    )

    df_batches = get_batches(
        df=df, n=n, models=models, outcome_col=outcome_col, treatment_col=treatment_col
    )

    df_batch_metrics = pd.DataFrame()
    if kind in ["effect", "gain", "qini"]:
        for i in range(n + 1)[1:]:  # From 1 through n
            batch_metrics = pd.DataFrame()
            for model in models:
                effect_metrics = catalog[kind](
                    df=df_batches[df_batches["{}_batches".format(model)] == i],
                    models=model,
                    outcome_col=outcome_col,
                    treatment_col=treatment_col,
                    normalize=normalize,
                    *args,
                    **kwargs,
                )
                if kind == "effect":
                    # Select last row, the cumsum effect for the model batch, make a df and transpose
                    df_effect_metrics = pd.DataFrame(effect_metrics.iloc[-1, :]).T
                    batch_metrics = pd.concat(
                        [batch_metrics, df_effect_metrics], axis=1
                    )

                elif kind == "gain":
                    # Cumulative gain is the cumulative causal effect of the population
                    gain_metrics = effect_metrics.mul(
                        effect_metrics.index.values, axis=0
                    )

                    if normalize:
                        gain_metrics = gain_metrics.div(
                            np.abs(gain_metrics.iloc[-1, :]), axis=1
                        )

                    gain_metrics = gain_metrics.sum() / gain_metrics.shape[0]

                    # Make a df and transpose to a row for concatenation
                    df_gain_metrics = pd.DataFrame(gain_metrics).T
                    batch_metrics = pd.concat([batch_metrics, df_gain_metrics], axis=1)

                elif kind == "qini":
                    qini_metrics = (
                        effect_metrics.sum(axis=0) - effect_metrics[RANDOM_COL].sum()
                    ) / effect_metrics.shape[0]

                    # Make a df and transpose to a row for concatenation
                    df_qini_metrics = pd.DataFrame(qini_metrics).T
                    batch_metrics = pd.concat([batch_metrics, df_qini_metrics], axis=1)

            # Select model columns and append the df with the row of full model metrics
            batch_metrics = batch_metrics[models]
            df_batch_metrics = df_batch_metrics.append(batch_metrics)

    elif kind == "response":
        for model in models:
            # Total in-batch known class amounts (potentially add in TN and CN units later)
            df_batch_metrics["{}_tp".format(model)] = df_batches.groupby(
                by="{}_batches".format(model)
            ).apply(lambda x: len(x[(x[treatment_col] == 1) & (x[outcome_col] == 1)]))
            df_batch_metrics["{}_cp".format(model)] = df_batches.groupby(
                by="{}_batches".format(model)
            ).apply(lambda x: len(x[(x[treatment_col] == 0) & (x[outcome_col] == 1)]))
            df_batch_metrics["{}_cn".format(model)] = df_batches.groupby(
                by="{}_batches".format(model)
            ).apply(lambda x: len(x[(x[treatment_col] == 0) & (x[outcome_col] == 0)]))
            df_batch_metrics["{}_tn".format(model)] = df_batches.groupby(
                by="{}_batches".format(model)
            ).apply(lambda x: len(x[(x[treatment_col] == 1) & (x[outcome_col] == 0)]))

            # The ratios of known unit classes to the number of treatment and control units per batch
            df_batch_metrics["{}_tp_rat".format(model)] = df_batch_metrics[
                "{}_tp".format(model)
            ] / df_batches.groupby(by="{}_batches".format(model)).apply(
                lambda x: len(x[(x[treatment_col] == 1)])
            )
            df_batch_metrics["{}_cp_rat".format(model)] = df_batch_metrics[
                "{}_cp".format(model)
            ] / df_batches.groupby(by="{}_batches".format(model)).apply(
                lambda x: len(x[(x[treatment_col] == 0)])
            )
            df_batch_metrics["{}_cn_rat".format(model)] = df_batch_metrics[
                "{}_cn".format(model)
            ] / df_batches.groupby(by="{}_batches".format(model)).apply(
                lambda x: len(x[(x[treatment_col] == 0)])
            )
            df_batch_metrics["{}_tn_rat".format(model)] = df_batch_metrics[
                "{}_tn".format(model)
            ] / df_batches.groupby(by="{}_batches".format(model)).apply(
                lambda x: len(x[(x[treatment_col] == 1)])
            )

            df_batch_metrics[model] = (
                df_batch_metrics["{}_tp_rat".format(model)]
                - df_batch_metrics["{}_cp_rat".format(model)]
            )

            # df_batch_metrics[model] = df_batch_metrics['{}_tp_rat'.format(model)] + df_batch_metrics['{}_cn_rat'.format(model)] \
            #                         - df_batch_metrics['{}_cp_rat'.format(model)] - df_batch_metrics['{}_tn_rat'.format(model)]

    df_plot = pd.DataFrame()
    batches_per_model = [[i + 1 for j in range(len(models))] for i in range(n)]
    df_plot["batch"] = [val for sublist in batches_per_model for val in sublist]

    metrics_per_batch = [[val for val in df_batch_metrics[col]] for col in models]
    df_plot["metric"] = [val for sublist in metrics_per_batch for val in sublist]

    models_per_batch = [[i for i in models] for i in range(n)]
    df_plot["model"] = [val for sublist in models_per_batch for val in sublist]

    # Adaptable figure features
    if figsize:
        sns.set(rc={"figure.figsize": figsize})

    # Set to default sns palette
    sns.set_palette("deep")

    ax = sns.barplot(data=df_plot, x="batch", y="metric", hue="model", ax=axis)
    plot_x_label = "Batches"
    number_to_quantile = [
        (3, "Tertiles"),
        (4, "Quartiles"),
        (5, "Quintiles"),
        (6, "Sextiles"),
        (7, "Septiles"),
        (8, "Octiles"),
        (10, "Deciles"),
        (20, "Ventiles"),
        (100, "Percentiles"),
    ]
    for i in number_to_quantile:
        if n == i[0]:
            plot_x_label = i[1]
    ax.set_xlabel(plot_x_label, fontsize=fontsize)

    kind_to_y_label = [
        ("effect", "Causal Effect"),
        ("gain", "Gain (AUUC)"),
        ("qini", "Qini"),
        ("response", "Response Difference"),
    ]
    for i in kind_to_y_label:
        if kind == i[0]:
            plot_y_label = i[1]
    ax.set_ylabel(plot_y_label, fontsize=fontsize)

    kind_to_title = [
        ("effect", "Batch Causal Effects"),
        ("gain", "Batch Gain (AUUC)"),
        ("qini", "Batch Qini"),
        ("response", "Batch T-C Response Differences"),
    ]
    for i in kind_to_title:
        if kind == i[0]:
            plot_title = i[1]
    if normalize and kind in ["gain", "qini"]:
        plot_title += " (Normalized)"

    ax.axes.set_title(plot_title, fontsize=fontsize * 1.5)


def plot_batch_effects(
    df,
    kind="effect",
    n=10,
    models=None,
    outcome_col="y",
    treatment_col="w",
    normalize=False,
    figsize=(15, 5),
    fontsize=20,
    axis=None,
):
    """
    Plots the effects batch chart: the cumulative batch effects predicted by a model given ranked treatment effects.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    kind : str : 'effect'

    n : int, optional (default=10, deciles; 20, quintiles also standard)
        The number of batches to split the units into

    models : list
        A list of models corresponding to estimated treatment effect columns

    outcome_col : str : optional (default=y)
        The column name for the actual outcome

    treatment_col : str : optional (default=w)
        The column name for the treatment indicator (0 or 1)

    figsize : tuple : optional
        Allows for quick changes of figures sizes

    fontsize : int or float : optional (default=20)
        The font size of the plots, with all labels scaled accordingly

    axis : str : optional (default=None)
        Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of batch effects of all models in df
    """
    plot_batch_metrics(
        df,
        kind="effect",
        n=n,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        normalize=normalize,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
    )


def plot_batch_gains(
    df,
    kind="gain",
    n=10,
    models=None,
    outcome_col="y",
    treatment_col="w",
    normalize=False,
    figsize=(15, 5),
    fontsize=20,
    axis=None,
):
    """
    Plots the batch gain chart: the cumulative batch gain predicted by a model given ranked treatment effects.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    kind : str : 'gain'

    n : int, optional (default=10, deciles; 20, quintiles also standard)
        The number of batches to split the units into

    models : list
        A list of models corresponding to estimated treatment effect columns

    outcome_col : str : optional (default=y)
        The column name for the actual outcome

    treatment_col : str : optional (default=w)
        The column name for the treatment indicator (0 or 1)

    figsize : tuple : optional
        Allows for quick changes of figures sizes

    fontsize : int or float : optional (default=20)
        The font size of the plots, with all labels scaled accordingly

    axis : str : optional (default=None)
        Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of batch gain of all models in df
    """
    plot_batch_metrics(
        df,
        kind="gain",
        n=n,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        normalize=normalize,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
    )


def plot_batch_qinis(
    df,
    kind="qini",
    n=10,
    models=None,
    outcome_col="y",
    treatment_col="w",
    normalize=False,
    figsize=(15, 5),
    fontsize=20,
    axis=None,
):
    """
    Plots the batch qini chart: the cumulative batch qini predicted by a model given ranked treatment effects.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    kind : str : 'qini'

    n : int, optional (default=10, deciles; 20, quintiles also standard)
        The number of batches to split the units into

    models : list
        A list of models corresponding to estimated treatment effect columns

    outcome_col : str : optional (default=y)
        The column name for the actual outcome

    treatment_col : str : optional (default=w)
        The column name for the treatment indicator (0 or 1)

    figsize : tuple : optional
        Allows for quick changes of figures sizes

    fontsize : int or float : optional (default=20)
        The font size of the plots, with all labels scaled accordingly

    axis : str : optional (default=None)
        Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of batch qini of all models in df
    """
    plot_batch_metrics(
        df,
        kind="qini",
        n=n,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        normalize=normalize,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
    )


def plot_batch_responses(
    df,
    n=10,
    models=None,
    outcome_col="y",
    treatment_col="w",
    normalize=False,
    figsize=(15, 5),
    fontsize=20,
    axis=None,
):
    """
    Plots the batch response chart: the cumulative batch responses predicted by a model given ranked treatment effects.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with model estimates and unit outcomes as columns

    kind : response

    n : int, optional (default=10, deciles; 20, quintiles also standard)
        The number of batches to split the units into

    models : list
        A list of models corresponding to estimated treatment effect columns

    outcome_col : str : optional (default=y)
        The column name for the actual outcome

    treatment_col : str : optional (default=w)
        The column name for the treatment indicator (0 or 1)

    figsize : tuple : optional
        Allows for quick changes of figures sizes

    fontsize : int or float : optional (default=20)
        The font size of the plots, with all labels scaled accordingly

    axis : str : optional (default=None)
        Adds an axis to the plot so they can be combined

    Returns
    -------
        A plot of batch responses of all models in df
    """
    plot_batch_metrics(
        df,
        kind="response",
        n=n,
        models=models,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        normalize=normalize,
        figsize=figsize,
        fontsize=fontsize,
        axis=axis,
    )


def signal_to_noise(y, w):
    """
    Computes the signal to noise ratio of a dataset to derive the potential for causal inference efficacy.

    Notes
    -----
        - The signal to noise ratio is the difference in treatment and control response to the control response

        - Values close to 0 imply that CI would have little benefit over predictive modeling

    Parameters
    ----------
        y : numpy.ndarray : (num_units,) : int, float
            Vector of unit responses

        w : numpy.ndarray : (num_units,) : int, float
            Vector of original treatment allocations across units

    Returns
    -------
        sn_ratio : float
    """
    y_treatment = [a * b for a, b in zip(y, w)]
    y_control = [a * (1 - b) for a, b in zip(y, w)]
    y_treatment_sum = np.sum(y_treatment)
    y_control_sum = np.sum(y_control)

    return (y_treatment_sum - y_control_sum) / y_control_sum


def iterate_model(
    model,
    X_train,
    y_train,
    w_train,
    X_test,
    y_test,
    w_test,
    tau_test=None,
    n=10,
    pred_type="predict",
    eval_type=None,
    normalize_eval=False,
    verbose=True,
):
    """
    Trains and makes predictions with a model multiple times to derive average predictions and their variance.

    Parameters
    ----------
        model : object
            A model over which iterations will be done

        X_train : numpy.ndarray : (num_train_units, num_features) : int, float
            Matrix of covariates

        y_train : numpy.ndarray : (num_train_units,) : int, float
            Vector of unit responses

        w_train : numpy.ndarray : (num_train_units,) : int, float
            Vector of original treatment allocations across units

        X_test : numpy.ndarray : (num_test_units, num_features) : int, float
            Matrix of covariates

        y_test : numpy.ndarray : (num_test_units,) : int, float
            Vector of unit responses

        w_test : numpy.ndarray : (num_test_units,) : int, float
            Vector of original treatment allocations across units

        tau_test : numpy.ndarray : (num_test_units,) : int, float
            Vector of the actual treatment effects given simulated data

        n : int (default=10)
            The number of train and prediction iterations to run

        pred_type : str (default=pred)
            predict or predict_proba: the type of prediction the iterations will make

        eval_type : str (default=None)
            qini or auuc: the type of evaluation to be done on the predictions

            Note: if None, model predictions will be averaged without their variance being calculated

        normalize_eval : bool : optional (default=False)
            Whether to normalize the evaluation metric

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query

    Returns
    -------
        avg_preds_probas : numpy.ndarray (num_units, 2) : float
            Averaged per unit predictions

        all_preds_probas : dict
            A dictionary of all predictions produced during iterations

        avg_eval : float
            The average of the iterated model evaluations

        eval_variance : float
            The variance of all prediction evaluations

        eval_variance : float
            The variance of all prediction evaluations

        all_evals : dict
            A dictionary of all evaluations produced during iterations
    """
    # Add train_test_split?
    if pred_type == "predict":
        try:
            model.__getattribute__("fit")
            model.__getattribute__("predict")
        except AttributeError:
            raise AttributeError(
                "Model should contains two methods for predict iteration: fit and predict."
            )

    if pred_type == "predict_proba":
        try:
            model.__getattribute__("fit")
            model.__getattribute__("predict_proba")
        except AttributeError:
            raise AttributeError(
                "Model should contains two methods for predict_proba iteration: fit and predict_proba."
            )

    catalog = {"qini": qini_score, "auuc": auuc_score, None: None}

    assert (
        eval_type in catalog.keys()
    ), "The {} evaluation type for iterate_model is not implemented. Select one of {}".format(
        eval_type, list(catalog.keys())
    )

    def _add_iter_eval(
        i,
        evaluation,
        all_preds_probas,
        all_evals,
        iter_results,
        y_test,
        w_test,
        tau_test=None,
        normalize_eval=False,
    ):
        """
        Iterates a model.
        """
        all_preds_probas[str(i)] = iter_results
        iter_effects = [i[0] - i[1] for i in iter_results]

        if tau_test:
            eval_dict = {"tau": tau_test, "model": iter_effects}
            df_eval = pd.DataFrame(eval_dict, columns=eval_dict.keys())
            iter_eval = evaluation(
                df=df_eval,
                models="model",
                treatment_effect_col="tau",
                normalize=normalize_eval,
            )

        else:
            eval_dict = {"y_test": y_test, "w_test": w_test, "model": iter_effects}
            df_eval = pd.DataFrame(eval_dict, columns=eval_dict.keys())
            iter_eval = evaluation(
                df=df_eval,
                models="model",
                outcome_col="y_test",
                treatment_col="w_test",
                normalize=normalize_eval,
            )

        iter_eval = iter_eval["model"]

        all_evals[str(i)] = iter_eval

        return all_preds_probas, all_evals

    evaluation = catalog[eval_type]

    i = 0
    all_preds_probas = {}
    all_evals = {}

    model_name = str(model).split(".")[-1].split(" ")[0]
    pbar = tqdm(
        total=n,
        desc=f"{model_name} iterations",
        unit="iter",
        disable=not verbose,
        leave=False,
    )
    if pred_type == "predict":
        while i < n:
            np.random.seed()

            model.fit(X=X_train, y=y_train, w=w_train)
            iter_results = model.predict(X=X_test)

            all_preds_probas, all_evals = _add_iter_eval(
                i=i,
                evaluation=evaluation,
                all_preds_probas=all_preds_probas,
                all_evals=all_evals,
                iter_results=iter_results,
                y_test=y_test,
                w_test=w_test,
                tau_test=tau_test,
                normalize_eval=normalize_eval,
            )

            i += 1
            pbar.update(1)

    else:  # Repeated to avoid checking pred_type over all iterations
        while i < n:
            np.random.seed()

            model.fit(X=X_train, y=y_train, w=w_train)
            iter_results = model.predict_proba(X=X_test)

            all_preds_probas, all_evals = _add_iter_eval(
                i=i,
                evaluation=evaluation,
                all_preds_probas=all_preds_probas,
                all_evals=all_evals,
                iter_results=iter_results,
                y_test=y_test,
                w_test=w_test,
                tau_test=tau_test,
                normalize_eval=normalize_eval,
            )

            i += 1
            pbar.update(1)

    list_of_preds = [val for val in all_preds_probas.values()]
    avg_preds_probas = np.mean(list_of_preds, axis=0)

    list_of_evals = [val for val in all_evals.values()]
    avg_eval = np.mean(list_of_evals, axis=0)

    # Measure of variance and sd (variances greater than one, two, etc sds above 0 could be marked to indicate high model deviation)
    eval_variance = np.var(list_of_evals)
    eval_sd = np.std(list_of_evals)

    return (
        avg_preds_probas,
        all_preds_probas,
        avg_eval,
        eval_variance,
        eval_sd,
        all_evals,
    )


def eval_table(eval_dict, variances=False, annotate_vars=False):
    """
    Displays the evaluation of models given a dictionary of their evaluations over datasets.

    Parameters
    ----------
        eval_dict : dict
            A dictionary of model evaluations over datasets

        variances : bool (default=False)
            Whether to annotate the evaluations with their variances

        annotate_vars : bool (default=False)
            Whether to annotate the evaluation variances with stars given their sds

    Returns
    -------
        eval_table : pandas.DataFrame : (num_datasets, num_models)
            A dataframe of dataset to model evaluation comparisons
    """
    assert isinstance(eval_dict, dict), "Dictionary type for evaluations not provided."

    def _annotate_variances(var, sd):
        """
        Returns stars equal to the number of standard deviations away from 0 a variance is.
        """
        if not np.isnan(var / sd):
            sds_to_0 = int(var / sd)
        else:
            sds_to_0 = 0

        return "{}{}".format(str(round(var, 4)), "*" * sds_to_0)

    datasets = list(eval_dict.keys())
    models = list(list(eval_dict.values())[0].keys())

    assert (
        len(models) > 1 or len(datasets) > 1
    ), "One dimensional inputs are not accepted for DataFrames."

    eval_table = pd.DataFrame(index=range(len(datasets)), columns=range(len(models)))
    eval_table.set_axis(models, axis=1, inplace=True)
    eval_table.set_axis(datasets, axis=0, inplace=True)

    if not variances:
        for d in list(eval_table.index):
            for m in list(eval_table.columns):
                if m in eval_dict[d].keys():
                    eval_table.loc[d, m] = round(eval_dict[d][m]["avg_eval"], 4)

    else:
        if not annotate_vars:
            for d in list(eval_table.index):
                for m in list(eval_table.columns):
                    if m in eval_dict[d].keys():
                        eval_table.loc[d, m] = "{} \u00B1 {}".format(
                            round(eval_dict[d][m]["avg_eval"], 4),
                            round(eval_dict[d][m]["eval_variance"], 4),
                        )

        else:
            for d in list(eval_table.index):
                for m in list(eval_table.columns):
                    if m in eval_dict[d].keys():
                        eval_table.loc[d, m] = "{} \u00B1 {}".format(
                            round(eval_dict[d][m]["avg_eval"], 4),
                            _annotate_variances(
                                eval_dict[d][m]["eval_variance"],
                                eval_dict[d][m]["eval_sd"],
                            ),
                        )

    return eval_table

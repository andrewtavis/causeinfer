"""
Utilities
---------

Utility functions for data manipulation and processing

Contents
    train_test_split,
    plot_unit_distributions,
    over_sample,
    multi_cross_tab
"""

import random

import numpy as np
import pandas as pd
import seaborn as sns


def train_test_split(
    X, y, w, percent_train=0.7, random_state=None, maintain_proportions=False
):
    """
    Split unit X covariates and (y,w) outcome tuples into training and testing sets

    Parameters
    ----------
        X : numpy.ndarray : (n_samples, n_features)
            Matrix of unit covariate features

        y : numpy.ndarray : (n_samples,)
            Array of unit responses

        w : numpy.ndarray : (n_samples,)
            Array of unit treatments

        percent_train : float
            The percent of the covariates and outcomes to delegate to model training

        random_state : int (default=None)
            A seed for the random number generator to allow for consistency (when in doubt, 42)

        maintain_proportions : bool : optional (default=False)
            Whether to maintain the treatment group proportions within the split samples

    Returns
    -------
        X_train, X_test, y_train, y_test, w_train, w_test : numpy.ndarray
            Arrays of split covariates and outcomes
    """
    if not (0 < percent_train < 1):
        raise ValueError("Train share should be float between 0 and 1.")

    if not len(X) == len(y) == len(w):
        raise ValueError("Lengths of covariates and outcomes not equal.")

    random.seed(random_state)

    if maintain_proportions:
        w_proportions = np.array(np.unique(w, return_counts=True)).T
        treatment_1_size = w_proportions[0][1]
        treatment_2_size = w_proportions[1][1]

        # Sort treatment indexes and then subset split them into lists of indexes for each
        sorted_indexes = np.argsort(w)
        treatment_1_indexes = sorted_indexes[: int(treatment_1_size)]
        treatment_2_indexes = sorted_indexes[int(treatment_1_size) :]

        # Number to select from each treatment sample
        N_train_t1 = int(percent_train * treatment_1_size)
        N_train_t2 = int(percent_train * treatment_2_size)

        train_index_t1 = random.sample([i for i in treatment_1_indexes], N_train_t1)
        train_index_t2 = random.sample([i for i in treatment_2_indexes], N_train_t2)

        test_index_t1 = [i for i in treatment_1_indexes if i not in train_index_t1]
        test_index_t2 = [i for i in treatment_2_indexes if i not in train_index_t2]

        # Indexes for each of the train-test samples, and shuffle them
        train_indexes = train_index_t1 + train_index_t2
        test_indexes = test_index_t1 + test_index_t2
        random.shuffle(train_indexes)
        random.shuffle(test_indexes)

    else:
        N = len(X)
        N_train = int(percent_train * N)
        train_indexes = random.sample([i for i in range(N)], N_train)
        test_indexes = [i for i in range(N) if i not in train_indexes]

    X_train = X[train_indexes, :]
    X_test = X[test_indexes, :]

    y_train = y[train_indexes]
    y_test = y[test_indexes]

    w_train = w[train_indexes]
    w_test = w[test_indexes]

    return X_train, X_test, y_train, y_test, w_train, w_test


def plot_unit_distributions(
    df, variable, treatment=None, bins=None, axis=None,
):
    """
    Plots seaborn countplots of unit covariate and outcome distributions

    Parameters
    ----------
        df_plot : pandas df, [n_samples, n_features]
            The data from which the plot is made

        variable : str
            A unit covariate or outcome for which the plot is desired

        treatment : str : optional (default=None)
            The treatment variable for comparing across segments

        bins : int (default=None)
            Bins the column values such that larger distributions can be plotted

        axis : str : optional (default=None)
            Adds an axis to the plot so they can be combined

    Returns
    -------
        ax : matplotlib.axes
            Displays a seaborn plot of unit distributions across the given covariate or outcome value
    """
    import re

    def _int_or_text(char):
        return int(char) if char.isdigit() else char

    def _alphanumeric_sort(text):
        """
        Added so the columns are correctly ordered
        """
        return [_int_or_text(char) for char in re.split(r"(\d+)", text)]

    def _float_range(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step

    # Set different colors for treatment plots
    if treatment:
        color_palette = "Set2"
    else:
        color_palette = "Set1"

    # Bin if requested and possible
    if bins:
        if df[str(variable)].dtype != int or float:
            try:
                df[str(variable)] = df[str(variable)].astype(float)
            except:
                print(
                    "The data type for the column can't be binned. The values of the column will be used as is."
                )
                bins = False

    if bins:
        bin_segments = list(
            _float_range(
                df[str(variable)].min(),
                df[str(variable)].max(),
                (df[str(variable)].max() - df[str(variable)].min()) / bins,
            )
        )

        # So plotting bounds are clean
        bin_segments = [int(i) for i in bin_segments[0:-2]] + [
            int(bin_segments[-1]) + 1
        ]

        # Bin the variable column based on the above defined list of segments
        df["binned_variable"] = pd.cut(df[str(variable)], bin_segments)

        order = list(df["binned_variable"].value_counts().index)
        order.sort()

        ax = sns.countplot(
            data=df,
            x="binned_variable",
            hue=treatment,
            order=order,
            ax=axis,
            palette=color_palette,
        )

        df.drop("binned_variable", axis=1, inplace=True)

    else:
        order = list(df[str(variable)].value_counts().index)
        try:
            order = [float(i) for i in order]
            order.sort(key=int)
        except:
            order.sort(key=_alphanumeric_sort)

        ax = sns.countplot(
            data=df,
            x=variable,
            hue=treatment,
            order=order,
            ax=axis,
            palette=color_palette,
        )

    return ax


def over_sample(X_1, y_1, w_1, sample_2_size, shuffle=True, random_state=None):
    """
    Over-samples to provide equality between a given sample and another it is smaller than

    Parameters
    ----------
        X_1 : numpy.ndarray : (num_sample1_units, num_sample1_features)
            Dataframe of sample covariates

        y_1 : numpy.ndarray : (num_sample1_units,)
            Vector of sample unit responses

        w_1 : numpy.ndarray : (num_sample1_units,)
            Designates the original treatment allocation across sample units

        sample_2_size : int
            The size of the other sample to match

        shuffle : bool : optional (default=True)
            Whether to shuffle the new sample after it's created

        random_state : int (default=None)
            A seed for the random number generator to allow for consistency (when in doubt, 42)

    Returns
    -------
        The provided covariates and outcomes, having been over-sampled to match another
            X_os : numpy.ndarray : (num_sample2_units, num_sample2_features)
            y_os : numpy.ndarray : (num_sample2_units,)
            w_os : numpy.ndarray : (num_sample2_units,)
    """
    if len(X_1) >= sample_2_size:
        raise ValueError(
            "The sample trying to be over-sampled is the same size or greater than what it should be matched with. "
            "Check sample sizes, and specifically that they haven't been switched on accident."
        )

    if len(X_1) != len(y_1) != len(w_1):
        raise ValueError(
            "The length of the covariates, responses, and treatments don't match."
        )

    random.seed(random_state)

    new_samples_needed = sample_2_size - len(X_1)
    sample_indexes = list(range(len(X_1)))
    os_indexes = np.random.choice(sample_indexes, size=new_samples_needed, replace=True)

    new_sample_indexes = sample_indexes + list(os_indexes)

    if shuffle:
        random.shuffle(new_sample_indexes)

    X_os = X_1[new_sample_indexes]
    y_os = y_1[new_sample_indexes]
    w_os = w_1[new_sample_indexes]

    print(
        """
    Old Covariates shape  : {}
    Old responses shape   : {}
    Old treatments shape  : {}
    New covariates shape  : {}
    New responses shape   : {}
    New treatments shape  : {}
    Matched sample length :  {}
                        """.format(
            X_1.shape,
            y_1.shape,
            w_1.shape,
            X_os.shape,
            y_os.shape,
            w_os.shape,
            sample_2_size,
        )
    )

    return X_os, y_os, w_os


def multi_cross_tab(df, w_col, y_cols, label_limit=3, margins=True, normalize=True):
    """
    Multi response column cross tabulations

    Parameters
    ----------
        df : pandas.DataFrame [n_samples, n_features]
            Dataframe with treatment and discrete response values

        w_col : str
            The name of the treatment column

        y_cols : list
            A list of discrete valued responses

        label_limit : int (default=3)
            The limit from the response names to use in column naming

        margins : bool : optional (default=True)
            Include cross tabulation summations across columns and rows

        normalize : bool : optional (default=True)
            Whether provide normalized or aggregate values in cross tabulation

    Returns
    -------
        cross_tab : pandas.DataFrame
            A cross tabulation of responses provided against treatment
    """
    y_to_concat = []
    for y in y_cols:
        # Cross tabulate over the given response
        cross_tab_y = pd.crosstab(
            df[w_col], df[y], margins=margins, normalize=normalize
        )
        # Rename for column distinction
        if label_limit >= 0:
            cross_tab_y.columns = [
                "{}_{}".format(str(y)[: int(label_limit)], col)
                for col in cross_tab_y.columns
            ]
        else:
            cross_tab_y.columns = [
                "{}_{}".format(str(y)[int(label_limit) :], col)
                for col in cross_tab_y.columns
            ]

        y_to_concat.append(cross_tab_y)

    cross_tab = pd.concat(y_to_concat, axis=1)

    # Remove repeat of margins column
    if margins:
        all_columns = [col for col in cross_tab.columns if "All" in col]

        all_col = cross_tab[all_columns[0]]
        cross_tab["All"] = all_col

        for col in all_columns:
            del cross_tab[col]

    return cross_tab

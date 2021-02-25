"""
Utilities Tests
---------------
"""

import pandas as pd
import matplotlib.pyplot as plt

from causeinfer import utils


def test_train_test_split(X_split_proba, y_split_proba, w_split_proba):
    X_tr, X_te, y_tr, y_te, w_tr, w_te = utils.train_test_split(
        X_split_proba,
        y_split_proba,
        w_split_proba,
        percent_train=0.7,
        random_state=42,
        maintain_proportions=False,
    )
    assert len(X_tr) > len(X_te)
    assert len(y_tr) > len(y_te)
    assert len(w_tr) > len(w_te)


def test_plot_unit_distributions(monkeypatch, hillstrom_df_full):
    monkeypatch.setattr(plt, "show", lambda: None)
    utils.plot_unit_distributions(
        df=hillstrom_df_full,
        variable="channel",
        treatment="treatment",
        bins=None,
        axis=None,
    )

    utils.plot_unit_distributions(
        df=hillstrom_df_full, variable="spend", treatment=None, bins=25, axis=None,
    )


def test_over_sample(X_control_proba, y_control_proba, w_control_proba, X_treat_proba):
    X_os = utils.over_sample(
        X_1=X_control_proba,
        y_1=y_control_proba,
        w_1=w_control_proba,
        sample_2_size=len(X_treat_proba),
        shuffle=True,
    )[0]
    assert len(X_os) == len(X_treat_proba)


def test_multi_cross_tab(hillstrom_df):
    df_test = utils.multi_cross_tab(
        df=hillstrom_df,
        w_col="treatment",
        y_cols=["visit", "conversion"],
        label_limit=6,
        margins=True,
        normalize=True,
    )
    assert type(df_test) == pd.DataFrame

    df_test = utils.multi_cross_tab(
        df=hillstrom_df,
        w_col="treatment",
        y_cols=["visit", "conversion"],
        label_limit=6,
        margins=False,
        normalize=False,
    )
    assert type(df_test) == pd.DataFrame

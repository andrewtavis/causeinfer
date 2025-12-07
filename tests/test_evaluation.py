# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluation Tests
----------------
"""

import matplotlib.pyplot as plt
import numpy as np
from causeinfer import evaluation

np.random.seed(42)

models = ["two_model", "interaction_term"]


def test_plot_cum_effect(monkeypatch, df_vis_eval_proba):
    monkeypatch.setattr(plt, "show", lambda: None)
    evaluation.plot_cum_effect(
        df=df_vis_eval_proba,
        n=20,
        models=models,
        percent_of_pop=False,
        outcome_col="y_test",
        treatment_col="w_test",
        random_seed=42,
        figsize=(10, 5),
        fontsize=20,
        axis=None,
        legend_metrics=False,
    )


def test_plot_cum_gain(monkeypatch, df_vis_eval_proba):
    monkeypatch.setattr(plt, "show", lambda: None)
    evaluation.plot_cum_gain(
        df=df_vis_eval_proba,
        n=100,
        models=models,
        percent_of_pop=True,
        outcome_col="y_test",
        treatment_col="w_test",
        normalize=True,
        random_seed=42,
        figsize=None,
        fontsize=20,
        axis=None,
        legend_metrics=True,
    )


def test_plot_qini(monkeypatch, df_vis_eval_proba):
    monkeypatch.setattr(plt, "show", lambda: None)
    evaluation.plot_qini(
        df=df_vis_eval_proba,
        n=100,
        models=models,
        percent_of_pop=True,
        outcome_col="y_test",
        treatment_col="w_test",
        normalize=True,
        random_seed=42,
        figsize=None,
        fontsize=20,
        axis=None,
        legend_metrics=True,
    )


def test_plot_batch_responses(monkeypatch, df_vis_eval_proba):
    monkeypatch.setattr(plt, "show", lambda: None)
    evaluation.plot_batch_responses(
        df=df_vis_eval_proba,
        n=10,
        models=models,
        outcome_col="y_test",
        treatment_col="w_test",
        normalize=False,
        figsize=None,
        fontsize=15,
        axis=None,
    )


def test_signal_to_noise(y_split_proba, w_split_proba):
    sn_ration = evaluation.signal_to_noise(y=y_split_proba, w=w_split_proba)
    assert isinstance(sn_ration, (float, np.float64))


def test_pred_proba_eval_table(model_evaluation_dict_proba):
    evaluation.eval_table(
        model_evaluation_dict_proba, variances=True, annotate_vars=True
    )
    evaluation.eval_table(
        model_evaluation_dict_proba, variances=False, annotate_vars=False
    )

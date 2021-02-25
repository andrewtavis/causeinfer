"""
Evaluation Tests
----------------
"""

import numpy as np
import matplotlib.pyplot as plt

from causeinfer import evaluation
from causeinfer.standard_algorithms.two_model import TwoModel
from causeinfer.standard_algorithms.interaction_term import InteractionTerm

np.random.seed(42)


def test_plot_cum_effect(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    assert True


def test_plot_cum_gain(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    assert True


def test_plot_qini(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    assert True


def test_auuc_score():
    assert True


def test_qini_score():
    assert True


def test_get_batch_metrics():
    assert True


def test_plot_batch_metrics(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    assert True


def test_plot_batch_responses(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    assert True


def test_signal_to_noise(y_split_proba, w_split_proba):
    sn_ration = evaluation.signal_to_noise(y=y_split_proba, w=w_split_proba)
    assert type(sn_ration) == float or type(sn_ration) == np.float64


def test_iterate_model_pred():
    assert True


def test_pred_proba_eval_table():
    assert True

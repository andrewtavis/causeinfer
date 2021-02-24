"""
Evaluation Tests
----------------
"""

import numpy as np
import matplotlib.pyplot as plt

from causeinfer import evaluation


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


def test_plot_batch_metrics():
    assert True


def test_plot_batch_responses():
    assert True


def test_signal_to_noise(y_split, w_split):
    sn_ration = evaluation.signal_to_noise(y=y_split, w=w_split)
    assert type(sn_ration) == float or type(sn_ration) == np.float64


def test_iterate_model():
    assert True


def test_eval_table():
    assert True

# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for standard algorithm prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from causeinfer.standard_algorithms.interaction_term import InteractionTerm
from causeinfer.standard_algorithms.two_model import TwoModel

np.random.seed(42)


def test_two_model(X_train_pred, y_train_pred, w_train_pred, X_test_pred):
    tm = TwoModel(
        treatment_model=RandomForestRegressor(random_state=42),
        control_model=RandomForestRegressor(random_state=42),
    )
    tm.fit(X=X_train_pred, y=y_train_pred, w=w_train_pred)

    tm_preds = tm.predict(X=X_test_pred)

    assert tm_preds.shape == (len(tm_preds), 2)
    assert np.isfinite(tm_preds).all()


def test_interaction_term(X_train_pred, y_train_pred, w_train_pred, X_test_pred):
    it = InteractionTerm(model=RandomForestRegressor(random_state=42))
    it.fit(X=X_train_pred, y=y_train_pred, w=w_train_pred)

    it_preds = it.predict(X=X_test_pred)

    assert it_preds.shape == (len(it_preds), 2)
    assert np.isfinite(it_preds).all()

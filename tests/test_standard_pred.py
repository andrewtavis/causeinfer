"""
Standard Algorithm Predict Tests
--------------------------------
"""

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from causeinfer.standard_algorithms.two_model import TwoModel
from causeinfer.standard_algorithms.interaction_term import InteractionTerm

np.random.seed(42)


def test_two_model(X_train_pred, y_train_pred, w_train_pred, X_test_pred):
    tm = TwoModel(
        treatment_model=RandomForestRegressor(random_state=42),
        control_model=RandomForestRegressor(random_state=42),
    )
    tm.fit(X=X_train_pred, y=y_train_pred, w=w_train_pred)

    tm_preds = tm.predict(X=X_test_pred)
    assert round(tm_preds[0].tolist()[0], 2) == 0.08
    assert round(tm_preds[1].tolist()[0], 2) == 0.16


def test_interaction_term(X_train_pred, y_train_pred, w_train_pred, X_test_pred):
    it = InteractionTerm(model=RandomForestRegressor(random_state=42))
    it.fit(X=X_train_pred, y=y_train_pred, w=w_train_pred)

    it_preds = it.predict(X=X_test_pred)
    assert round(it_preds[0].tolist()[0], 2) == 0.07
    assert round(it_preds[1].tolist()[0], 2) == 0.16

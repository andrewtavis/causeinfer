"""
Standard Algorithm Predict Proba Tests
--------------------------------------
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from causeinfer.standard_algorithms.two_model import TwoModel
from causeinfer.standard_algorithms.binary_transformation import BinaryTransformation
from causeinfer.standard_algorithms.interaction_term import InteractionTerm
from causeinfer.standard_algorithms.quaternary_transformation import (
    QuaternaryTransformation,
)

np.random.seed(42)


def test_two_model(X_train, y_train, w_train, X_test):
    tm = TwoModel(
        treatment_model=RandomForestClassifier(random_state=42),
        control_model=RandomForestClassifier(random_state=42),
    )
    tm.fit(X=X_train, y=y_train, w=w_train)

    tm_probas = tm.predict_proba(X=X_test)
    assert tm_probas[0].tolist() == [0.88, 1.0]


def test_interaction_term(X_train, y_train, w_train, X_test):
    it = InteractionTerm(model=RandomForestClassifier(random_state=42))
    it.fit(X=X_train, y=y_train, w=w_train)

    it_probas = it.predict_proba(X=X_test)
    assert it_probas[0].tolist() == [0.89, 1.0]


def test_binary_transformation(X_train, y_train, w_train, X_test):
    bt = BinaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=False
    )
    bt.fit(X=X_train, y=y_train, w=w_train)

    bt_probas = bt.predict_proba(X=X_test)
    assert bt_probas[0].tolist() == [0.94, 0.06]

    bt = BinaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=True
    )
    bt.fit(X=X_train, y=y_train, w=w_train)

    bt_probas = bt.predict_proba(X=X_test)
    assert bt_probas[0].tolist() == [0.49175751503006004, 0.02861122244488978]


def test_quaternary_transformation(X_train, y_train, w_train, X_test):
    qt = QuaternaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=False
    )
    qt.fit(X=X_train, y=y_train, w=w_train)

    qt_probas = qt.predict_proba(X=X_test)
    assert qt_probas[0].tolist() == [0.95, 0.05]

    qt = QuaternaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=True
    )
    qt.fit(X=X_train, y=y_train, w=w_train)

    qt_probas = qt.predict_proba(X=X_test)
    assert round(qt_probas[0].tolist()[0], 7) == 6.35 * 10 ** -5
    assert round(qt_probas[1].tolist()[0], 7) == 3.01 * 10 ** -5

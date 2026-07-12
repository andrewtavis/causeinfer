# SPDX-License-Identifier: BSD-3-Clause
"""
Standard Algorithm Prediction Probability Tests
-----------------------------------------------
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from causeinfer.standard_algorithms.binary_transformation import BinaryTransformation
from causeinfer.standard_algorithms.interaction_term import InteractionTerm
from causeinfer.standard_algorithms.pessimistic import PessimisticUplift
from causeinfer.standard_algorithms.quaternary_transformation import (
    QuaternaryTransformation,
)
from causeinfer.standard_algorithms.reflective import ReflectiveUplift
from causeinfer.standard_algorithms.two_model import TwoModel

np.random.seed(42)


def test_two_model(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    tm = TwoModel(
        treatment_model=RandomForestClassifier(random_state=42),
        control_model=RandomForestClassifier(random_state=42),
    )
    tm.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    tm_probas = tm.predict_proba(X=X_test_proba)
    assert tm_probas[0].tolist() == [0.96, 0.65]


def test_interaction_term(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    it = InteractionTerm(model=RandomForestClassifier(random_state=42))
    it.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    it_probas = it.predict_proba(X=X_test_proba)
    assert it_probas[0].tolist() == [0.97, 0.77]


def test_binary_transformation(
    X_train_proba, y_train_proba, w_train_proba, X_test_proba
):
    bt = BinaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=False
    )
    bt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    bt_probas = bt.predict_proba(X=X_test_proba)
    assert bt_probas[0].tolist() == [0.18, 0.82]

    bt = BinaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=True
    )
    bt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    bt_probas = bt.predict_proba(X=X_test_proba)
    assert bt_probas[0].tolist() == [0.0937635270541082, 0.39285504342017363]


def test_quaternary_transformation(
    X_train_proba, y_train_proba, w_train_proba, X_test_proba
):
    qt = QuaternaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=False
    )
    qt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    qt_probas = qt.predict_proba(X=X_test_proba)
    assert qt_probas[0].tolist() == [0.18, 0.8200000000000001]

    qt = QuaternaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=True
    )
    qt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    qt_probas = qt.predict_proba(X=X_test_proba)
    assert round(qt_probas[0].tolist()[0], 7) == 1.2e-05
    assert round(qt_probas[1].tolist()[0], 7) == 6.55e-05


def test_reflective_uplift(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    ru = ReflectiveUplift(model=RandomForestClassifier(random_state=42))
    ru.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    ru_probas = ru.predict_proba(X=X_test_proba)
    assert ru_probas[0].tolist() == [0.030719595371292818, 0.4059255449636853]


def test_pessimistic_uplift(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    pu = PessimisticUplift(model=RandomForestClassifier(random_state=42))
    pu.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    pu_probas = pu.predict_proba(X=X_test_proba)
    assert pu_probas[0].tolist() == [0.04535979768564641, 0.20296277248184266]

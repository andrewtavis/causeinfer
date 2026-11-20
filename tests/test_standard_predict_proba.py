# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for standard algorithm probability prediction.
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

    assert tm_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(tm_probas).all()
    assert ((tm_probas >= 0) & (tm_probas <= 1)).all()


def test_interaction_term(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    it = InteractionTerm(model=RandomForestClassifier(random_state=42))
    it.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    it_probas = it.predict_proba(X=X_test_proba)

    assert it_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(it_probas).all()
    assert ((it_probas >= 0) & (it_probas <= 1)).all()


def test_binary_transformation(
    X_train_proba, y_train_proba, w_train_proba, X_test_proba
):
    bt = BinaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=False
    )
    bt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    bt_probas = bt.predict_proba(X=X_test_proba)

    assert bt_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(bt_probas).all()
    assert ((bt_probas >= 0) & (bt_probas <= 1)).all()

    bt = BinaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=True
    )
    bt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    bt_probas = bt.predict_proba(X=X_test_proba)

    assert bt_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(bt_probas).all()
    assert ((bt_probas >= 0) & (bt_probas <= 1)).all()


def test_quaternary_transformation(
    X_train_proba, y_train_proba, w_train_proba, X_test_proba
):
    qt = QuaternaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=False
    )
    qt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    qt_probas = qt.predict_proba(X=X_test_proba)

    assert qt_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(qt_probas).all()
    assert ((qt_probas >= 0) & (qt_probas <= 1)).all()

    qt = QuaternaryTransformation(
        model=RandomForestClassifier(random_state=42), regularize=True
    )
    qt.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    qt_probas = qt.predict_proba(X=X_test_proba)

    assert qt_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(qt_probas).all()
    assert ((qt_probas >= 0) & (qt_probas <= 1)).all()


def test_reflective_uplift(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    ru = ReflectiveUplift(model=RandomForestClassifier(random_state=42))
    ru.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    ru_probas = ru.predict_proba(X=X_test_proba)

    assert ru_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(ru_probas).all()
    assert ((ru_probas >= 0) & (ru_probas <= 1)).all()


def test_pessimistic_uplift(X_train_proba, y_train_proba, w_train_proba, X_test_proba):
    pu = PessimisticUplift(model=RandomForestClassifier(random_state=42))
    pu.fit(X=X_train_proba, y=y_train_proba, w=w_train_proba)

    pu_probas = pu.predict_proba(X=X_test_proba)

    assert pu_probas.shape == (len(X_test_proba), 2)
    assert np.isfinite(pu_probas).all()
    assert ((pu_probas >= 0) & (pu_probas <= 1)).all()

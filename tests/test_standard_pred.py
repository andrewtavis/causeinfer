"""
Standard Algorithm Predict Tests
--------------------------------
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from causeinfer.standard_algorithms.two_model import TwoModel
from causeinfer.standard_algorithms.interaction_term import InteractionTerm

from causeinfer import utils
from causeinfer.data import cmf_micro

np.random.seed(42)

data_cmf_micro = cmf_micro.load_cmf_micro(
    file_path="./causeinfer/data/datasets/cmf_micro",
    format_covariates=True,
    normalize=True,
)

df = pd.DataFrame(
    data_cmf_micro["dataset_full"], columns=data_cmf_micro["dataset_full_names"]
)

X = data_cmf_micro["features"]
y = data_cmf_micro["response_biz_index"]  # response_biz_index or response_women_emp
w = data_cmf_micro["treatment"]

control_indexes = [i for i, e in enumerate(w) if e == 0]
treatment_indexes = [i for i, e in enumerate(w) if e == 1]

X_control = X[control_indexes]
y_control = y[control_indexes]
w_control = w[control_indexes]

X_treatment = X[treatment_indexes]
y_treatment = y[treatment_indexes]
w_treatment = w[treatment_indexes]

X_os, y_os, w_os = utils.over_sample(
    X_1=X_control,
    y_1=y_control,
    w_1=w_control,
    sample_2_size=len(X_treatment),
    shuffle=True,
    random_state=42,
)

X_split = np.append(X_os, X_treatment, axis=0)
y_split = np.append(y_os, y_treatment, axis=0)
w_split = np.append(w_os, w_treatment, axis=0)

X_train, X_test, y_train, y_test, w_train, w_test = utils.train_test_split(
    X_split,
    y_split,
    w_split,
    percent_train=0.7,
    random_state=42,
    maintain_proportions=True,
)


def test_two_model():
    tm = TwoModel(
        treatment_model=RandomForestRegressor(random_state=42),
        control_model=RandomForestRegressor(random_state=42),
    )
    tm.fit(X=X_train, y=y_train, w=w_train)

    tm_preds = tm.predict(X=X_test)
    assert tm_preds[0].tolist() == [0.08064385365694761, 0.08077331738546491]


def test_interaction_term():
    it = InteractionTerm(model=RandomForestRegressor(random_state=42))
    it.fit(X=X_train, y=y_train, w=w_train)

    it_preds = it.predict(X=X_test)
    assert it_preds[0].tolist() == [0.0678707442432642, 0.0690310326963663]

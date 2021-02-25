"""
Fixtures
--------
"""

import os
import random

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import causeinfer

from causeinfer import utils
from causeinfer import evaluation

from causeinfer.data import cmf_micro
from causeinfer.data import download_utils
from causeinfer.data import hillstrom
from causeinfer.data import mayo_pbc

from causeinfer.standard_algorithms import base_models
from causeinfer.standard_algorithms import binary_transformation
from causeinfer.standard_algorithms import interaction_term
from causeinfer.standard_algorithms import quaternary_transformation
from causeinfer.standard_algorithms import two_model

np.random.seed(42)

hillstrom.download_hillstrom()
hillstrom_data = hillstrom.load_hillstrom(
    file_path="./datasets/hillstrom.csv", format_covariates=True, normalize=True
)

hill_df = pd.DataFrame(
    hillstrom_data["dataset_full"], columns=hillstrom_data["dataset_full_names"]
)

X = hillstrom_data["features"]
y = hillstrom_data["response_visit"]
w = hillstrom_data["treatment"]

control_indexes = [i for i, e in enumerate(w) if e == 0]
womens_indexes = [i for i, e in enumerate(w) if e == 2]

X_c_proba = X[control_indexes]
y_c_proba = y[control_indexes]
w_c_proba = w[control_indexes]

X_t_proba = X[womens_indexes]
y_t_proba = y[womens_indexes]
w_t_proba = w[womens_indexes]

if 2 in w_t_proba:
    w_t_proba = [1 for i in w_t_proba if i == 2]

X_os_proba, y_os_proba, w_os_proba = utils.over_sample(
    X_1=X_c_proba,
    y_1=y_c_proba,
    w_1=w_c_proba,
    sample_2_size=len(X_t_proba),
    shuffle=False,
)

X_sp_proba = np.append(X_os_proba, X_t_proba, axis=0)
y_sp_proba = np.append(y_os_proba, y_t_proba, axis=0)
w_sp_proba = np.append(w_os_proba, w_t_proba, axis=0)

(
    X_tr_proba,
    X_te_proba,
    y_tr_proba,
    y_te_proba,
    w_tr_proba,
    w_te_proba,
) = utils.train_test_split(
    X_sp_proba,
    y_sp_proba,
    w_sp_proba,
    percent_train=0.7,
    random_state=42,
    maintain_proportions=True,
)


@pytest.fixture(params=[hill_df])
def hillstrom_df(request):
    return request.param


data_raw = hillstrom.load_hillstrom(
    file_path="./datasets/hillstrom.csv", format_covariates=False, normalize=False
)

hill_df_full = pd.DataFrame(
    data_raw["dataset_full"], columns=data_raw["dataset_full_names"]
)


@pytest.fixture(params=[hill_df_full])
def hillstrom_df_full(request):
    return request.param


# Control and treatment
@pytest.fixture(params=[X_c_proba])
def X_control_proba(request):
    return request.param


@pytest.fixture(params=[y_c_proba])
def y_control_proba(request):
    return request.param


@pytest.fixture(params=[w_c_proba])
def w_control_proba(request):
    return request.param


@pytest.fixture(params=[X_t_proba])
def X_treat_proba(request):
    return request.param


@pytest.fixture(params=[y_t_proba])
def y_treat_proba(request):
    return request.param


@pytest.fixture(params=[w_t_proba])
def w_treat_proba(request):
    return request.param


@pytest.fixture(params=[X_sp_proba])
def X_split_proba(request):
    return request.param


@pytest.fixture(params=[y_sp_proba])
def y_split_proba(request):
    return request.param


@pytest.fixture(params=[w_sp_proba])
def w_split_proba(request):
    return request.param


# Train and test
@pytest.fixture(params=[X_tr_proba])
def X_train_proba(request):
    return request.param


@pytest.fixture(params=[y_tr_proba])
def y_train_proba(request):
    return request.param


@pytest.fixture(params=[w_tr_proba])
def w_train_proba(request):
    return request.param


@pytest.fixture(params=[X_te_proba])
def X_test_proba(request):
    return request.param


@pytest.fixture(params=[y_te_proba])
def y_test_proba(request):
    return request.param


@pytest.fixture(params=[w_te_proba])
def w_test_proba(request):
    return request.param


data_cmf_micro = cmf_micro.load_cmf_micro(
    file_path="./causeinfer/data/datasets/cmf_micro",
    format_covariates=True,
    normalize=True,
)

cmf_m_df = pd.DataFrame(
    data_cmf_micro["dataset_full"], columns=data_cmf_micro["dataset_full_names"]
)

X = data_cmf_micro["features"]
y = data_cmf_micro["response_biz_index"]  # response_biz_index or response_women_emp
w = data_cmf_micro["treatment"]

control_indexes = [i for i, e in enumerate(w) if e == 0]
treatment_indexes = [i for i, e in enumerate(w) if e == 1]

X_c_pred = X[control_indexes]
y_c_pred = y[control_indexes]
w_c_pred = w[control_indexes]

X_t_pred = X[treatment_indexes]
y_t_pred = y[treatment_indexes]
w_t_pred = w[treatment_indexes]

X_os_pred, y_os_pred, w_os_pred = utils.over_sample(
    X_1=X_c_pred,
    y_1=y_c_pred,
    w_1=w_c_pred,
    sample_2_size=len(X_t_pred),
    shuffle=True,
    random_state=42,
)

X_sp_pred = np.append(X_os_pred, X_t_pred, axis=0)
y_sp_pred = np.append(y_os_pred, y_t_pred, axis=0)
w_sp_pred = np.append(w_os_pred, w_t_pred, axis=0)

(
    X_tr_pred,
    X_te_pred,
    y_tr_pred,
    y_te_pred,
    w_tr_pred,
    w_te_pred,
) = utils.train_test_split(
    X_sp_pred,
    y_sp_pred,
    w_sp_pred,
    percent_train=0.7,
    random_state=42,
    maintain_proportions=True,
)


@pytest.fixture(params=[cmf_m_df])
def cmf_micro_df(request):
    return request.param


# Control and treatment
@pytest.fixture(params=[X_c_pred])
def X_control_pred(request):
    return request.param


@pytest.fixture(params=[y_c_pred])
def y_control_pred(request):
    return request.param


@pytest.fixture(params=[w_c_pred])
def w_control_pred(request):
    return request.param


@pytest.fixture(params=[X_t_pred])
def X_treat_pred(request):
    return request.param


@pytest.fixture(params=[y_t_pred])
def y_treat_pred(request):
    return request.param


@pytest.fixture(params=[w_t_pred])
def w_treat_pred(request):
    return request.param


@pytest.fixture(params=[X_sp_pred])
def X_split_pred(request):
    return request.param


@pytest.fixture(params=[y_sp_pred])
def y_split_pred(request):
    return request.param


@pytest.fixture(params=[w_sp_pred])
def w_split_pred(request):
    return request.param


# Train and test
@pytest.fixture(params=[X_tr_pred])
def X_train_pred(request):
    return request.param


@pytest.fixture(params=[y_tr_pred])
def y_train_pred(request):
    return request.param


@pytest.fixture(params=[w_tr_pred])
def w_train_pred(request):
    return request.param


@pytest.fixture(params=[X_te_pred])
def X_test_pred(request):
    return request.param


@pytest.fixture(params=[y_te_pred])
def y_test_pred(request):
    return request.param


@pytest.fixture(params=[w_te_pred])
def w_test_pred(request):
    return request.param


os.system("rm -rf ./datasets")


# Iterated proba models
tm = two_model.TwoModel(
    treatment_model=RandomForestClassifier(random_state=42),
    control_model=RandomForestClassifier(random_state=42),
)
it = interaction_term.InteractionTerm(model=RandomForestClassifier(random_state=42))

model_eval_dict_proba = {}
model_eval_dict_proba["Hillstrom"] = {}
model_eval_dict_proba

for model in [tm, it]:
    (
        avg_preds,
        all_preds,
        avg_eval,
        eval_variance,
        eval_sd,
        all_evals,
    ) = evaluation.iterate_model(
        model=model,
        X_train=X_tr_proba,
        y_train=y_tr_proba,
        w_train=w_tr_proba,
        X_test=X_te_proba,
        y_test=y_te_proba,
        w_test=w_te_proba,
        tau_test=None,
        n=5,
        pred_type="predict_proba",
        eval_type="qini",
        normalize_eval=False,
        verbose=True,
    )
    model_eval_dict_proba["Hillstrom"].update(
        {
            str(model)
            .split(".")[-1]
            .split(" ")[0]: {
                "avg_preds": avg_preds,
                "all_preds": all_preds,
                "avg_eval": avg_eval,
                "eval_variance": eval_variance,
                "eval_sd": eval_sd,
                "all_evals": all_evals,
            }
        }
    )

# Treatment and control probability subtraction
tm_effects_proba = [
    pred[0] - pred[1]
    for pred in model_eval_dict_proba["Hillstrom"]["TwoModel"]["avg_preds"]
]

# Treatment interaction and control interaction probability subtraction
it_effects_proba = [
    pred[0] - pred[1]
    for pred in model_eval_dict_proba["Hillstrom"]["InteractionTerm"]["avg_preds"]
]

visual_eval_dict_proba = {
    "y_test": y_te_proba,
    "w_test": w_te_proba,
    "two_model": tm_effects_proba,
    "interaction_term": it_effects_proba,
}

df_vis_proba = pd.DataFrame(
    visual_eval_dict_proba, columns=visual_eval_dict_proba.keys()
)


@pytest.fixture(params=[df_vis_proba])
def df_vis_eval_proba(request):
    return request.param


@pytest.fixture(params=[model_eval_dict_proba])
def model_evaluation_dict_proba(request):
    return request.param


# Iterated pred models
tm = two_model.TwoModel(
    treatment_model=RandomForestRegressor(random_state=42),
    control_model=RandomForestRegressor(random_state=42),
)
it = interaction_term.InteractionTerm(model=RandomForestRegressor(random_state=42))

model_eval_dict_pred = {}
model_eval_dict_pred["CMF Microfinance"] = {}
model_eval_dict_pred

for model in [tm, it]:
    (
        avg_preds,
        all_preds,
        avg_eval,
        eval_variance,
        eval_sd,
        all_evals,
    ) = evaluation.iterate_model(
        model=model,
        X_train=X_tr_pred,
        y_train=y_tr_pred,
        w_train=w_tr_pred,
        X_test=X_te_pred,
        y_test=y_te_pred,
        w_test=w_te_pred,
        tau_test=None,
        n=5,
        pred_type="predict",
        eval_type="auuc",
        normalize_eval=False,
        verbose=True,
    )
    model_eval_dict_pred["CMF Microfinance"].update(
        {
            str(model)
            .split(".")[-1]
            .split(" ")[0]: {
                "avg_preds": avg_preds,
                "all_preds": all_preds,
                "avg_eval": avg_eval,
                "eval_variance": eval_variance,
                "eval_sd": eval_sd,
                "all_evals": all_evals,
            }
        }
    )


# Treatment and control probability subtraction
tm_effects_pred = [
    pred[0] - pred[1]
    for pred in model_eval_dict_pred["CMF Microfinance"]["TwoModel"]["avg_preds"]
]

# Treatment interaction and control interaction probability subtraction
it_effects_pred = [
    pred[0] - pred[1]
    for pred in model_eval_dict_pred["CMF Microfinance"]["InteractionTerm"]["avg_preds"]
]

visual_eval_dict_pred = {
    "y_test": y_te_pred,
    "w_test": w_te_pred,
    "two_model": tm_effects_pred,
    "interaction_term": it_effects_pred,
}

df_vis_pred = pd.DataFrame(visual_eval_dict_pred, columns=visual_eval_dict_pred.keys())


@pytest.fixture(params=[df_vis_pred])
def df_vis_eval_pred(request):
    return request.param

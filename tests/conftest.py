"""
Fixtures
--------
"""

import os
import random

import numpy as np
import pandas as pd
import pytest

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

df = pd.DataFrame(
    hillstrom_data["dataset_full"], columns=hillstrom_data["dataset_full_names"]
)

X = hillstrom_data["features"]
y = hillstrom_data["response_visit"]
w = hillstrom_data["treatment"]

control_indexes = [i for i, e in enumerate(w) if e == 0]
womens_indexes = [i for i, e in enumerate(w) if e == 2]

X_c = X[control_indexes]
y_c = y[control_indexes]
w_c = w[control_indexes]

X_t = X[womens_indexes]
y_t = y[womens_indexes]
w_t = w[womens_indexes]

if 2 in w_t:
    w_t = [1 for i in w_t if i == 2]
w_t[:5]

X_os, y_os, w_os = utils.over_sample(
    X_1=X_c, y_1=y_c, w_1=w_c, sample_2_size=len(X_t), shuffle=False,
)

X_sp = np.append(X_os, X_t, axis=0)
y_sp = np.append(y_os, y_t, axis=0)
w_sp = np.append(w_os, w_t, axis=0)

X_tr, X_te, y_tr, y_te, w_tr, w_te = utils.train_test_split(
    X_sp, y_sp, w_sp, percent_train=0.7, random_state=42, maintain_proportions=True,
)


@pytest.fixture(params=[df])
def hillstrom_df(request):
    return request.param


data_raw = hillstrom.load_hillstrom(
    file_path="./datasets/hillstrom.csv", format_covariates=False, normalize=False
)

df_full = pd.DataFrame(data_raw["dataset_full"], columns=data_raw["dataset_full_names"])


@pytest.fixture(params=[df_full])
def hillstrom_df_full(request):
    return request.param


# Control and treatment
@pytest.fixture(params=[X_c])
def X_control(request):
    return request.param


@pytest.fixture(params=[y_c])
def y_control(request):
    return request.param


@pytest.fixture(params=[w_c])
def w_control(request):
    return request.param


@pytest.fixture(params=[X_t])
def X_treat(request):
    return request.param


@pytest.fixture(params=[y_t])
def y_treat(request):
    return request.param


@pytest.fixture(params=[w_t])
def w_treat(request):
    return request.param


@pytest.fixture(params=[X_sp])
def X_split(request):
    return request.param


@pytest.fixture(params=[y_sp])
def y_split(request):
    return request.param


@pytest.fixture(params=[w_sp])
def w_split(request):
    return request.param


# Train and test
@pytest.fixture(params=[X_tr])
def X_train(request):
    return request.param


@pytest.fixture(params=[y_tr])
def y_train(request):
    return request.param


@pytest.fixture(params=[w_tr])
def w_train(request):
    return request.param


@pytest.fixture(params=[X_te])
def X_test(request):
    return request.param


@pytest.fixture(params=[y_te])
def y_test(request):
    return request.param


@pytest.fixture(params=[w_te])
def w_test(request):
    return request.param


os.system("rm -rf ./datasets")

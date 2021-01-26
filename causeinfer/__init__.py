from causeinfer.data import cmf_micro

data_cmf_micro = cmf_micro.load_cmf_micro(
    user_file_path="datasets/cmf_micro", format_covariates=True, normalize=True
)

df = pd.DataFrame(
    data_cmf_micro["dataset_full"], columns=data_cmf_micro["dataset_full_names"]
)

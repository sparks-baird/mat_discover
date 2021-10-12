# make sure to pip3 install automatminer if using conda (pip might have issues)
from automatminer import MatPipe
from automatminer.presets import get_preset_config
from automatminer.automl.adaptors import TPOTAdaptor

# https://matsci.org/t/error-found-array-with-0-feature-s/4848/10?u=sgbaird
config_dict_1 = {
    "sklearn.ensemble.RandomForestRegressor": {
        "n_estimators": [20, 100, 200, 500, 1000],
        "max_features": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        "min_samples_split": range(2, 21, 3),
        "min_samples_leaf": range(1, 21, 3),
        "bootstrap": [True, False],
    },
    "sklearn.ensemble.GradientBoostingRegressor": {
        "n_estimators": [20, 100, 200, 500, 1000],
        "loss": ["ls", "lad", "huber", "quantile"],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
        "max_depth": range(1, 11, 2),
        "min_samples_split": range(2, 21, 3),
        "min_samples_leaf": range(1, 21, 3),
        "subsample": list(np.arange(0.0, 1.05, 0.05)),
        "max_features": list(np.arange(0.0, 1.05, 0.05)),
        "alpha": [0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
    },
    "sklearn.ensemble.ExtraTreesRegressor": {
        "n_estimators": [20, 100, 200, 500, 1000],
        "max_features": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        "min_samples_split": range(2, 21, 3),
        "min_samples_leaf": range(1, 21, 3),
        "bootstrap": [True, False],
    },
    "sklearn.tree.DecisionTreeRegressor": {
        "max_depth": range(1, 11, 2),
        "min_samples_split": range(2, 21, 3),
        "min_samples_leaf": range(1, 21, 3),
    },
    "sklearn.neighbors.KNeighborsRegressor": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "sklearn.linear_model.Lasso": {
        "alpha": [1e-2, 1e-1, 1e0, 1e1, 1e2]
    },  # J alpha values taken from Takigawa-2019
    "sklearn.linear_model.LassoLarsCV": {"normalize": [True, False]},
    "sklearn.linear_model.RidgeCV": {},
    "sklearn.linear_model.ElasticNetCV": {
        "l1_ratio": list(np.arange(0.0, 1.05, 0.05)),
        "tol": [1e-05, 0.0001, 0.001, 0.01, 0.1],
    },
    "sklearn.preprocessing.MaxAbsScaler": {},
    "sklearn.preprocessing.RobustScaler": {},
    "sklearn.preprocessing.StandardScaler": {},
    "sklearn.preprocessing.MinMaxScaler": {},
    "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
    "sklearn.preprocessing.PolynomialFeatures": {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False],
    },
    "sklearn.kernel_approximation.RBFSampler": {
        "gamma": list(np.arange(0.0, 1.05, 0.05))
    },
    "sklearn.kernel_approximation.Nystroem": {
        "kernel": [
            "rbf",
            "cosine",
            "chi2",
            "laplacian",
            "polynomial",
            "poly",
            "linear",
            "additive_chi2",
            "sigmoid",
        ],
        "gamma": list(np.arange(0.0, 1.05, 0.05)),
        "n_components": range(1, 11),
    },
    "tpot.builtins.ZeroCount": {},
    "tpot.builtins.OneHotEncoder": {
        "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
        "sparse": [False],
        "threshold": [10],
    },
    "sklearn.preprocessing.Binarizer": {"threshold": list(np.arange(0.0, 1.05, 0.05))},
    "sklearn.cluster.FeatureAgglomeration": {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
    },
    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {"sklearn.feature_selection.f_regression": None},
    },
    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11),
    },
    "sklearn.decomposition.FastICA": {"tol": list(np.arange(0.0, 1.05, 0.05))},
    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
}

config = get_preset_config("production")

config["learner"] = TPOTAdaptor(
    max_time_mins=60,
    max_eval_time_mins=2,
    cv=5,
    verbosity=3,
    memory="auto",
    template="Selector-Transformer-Regressor",
    scoring="neg_mean_absolute_error",
    config_dict=config_dict_1,
)
pipe = MatPipe(**config)


# pipe = MatPipe.from_preset("express")
run_pipeline = True
if run_pipeline:
    pipe.fit(elast_df, "K_VRH")
    predicted_df = pipe.predict(all_df, ignore=["task_id"])

    pipe.save("mp-k-vrh.p")
    predicted_df.to_pickle("predicted_df.pkl")
else:
    pipe.load("mp-k-vrh.p")
    predicted_df.to_pickle("predicted_df.pkl")

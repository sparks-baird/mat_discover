"""Train Discover() using different Scalers and relative weights and create outputs.

# TODO: replace with a more sophisticated ML "experiment" tracker.

"""
# %% imports
from os.path import join
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
from crabnet.data.materials_data import elasticity
from mat_discover.mat_discover_ import Discover, cdf_sorting_error

# %% setup
# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy = False
# set gcv to False for a quicker run --> group-cross validation can take a while
gcv = False

Scalers = [MinMaxScaler, RobustScaler, PowerTransformer]
weight_pairs = [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1)]

disc = Discover(dummy_run=dummy, device="cuda")
train_df, val_df = disc.data(elasticity, "train.csv", dummy=dummy)

# %% Loop through parameter combinations
for Scaler in Scalers:
    for weight_pair in weight_pairs:
        print("=" * 20 + "PARAMETERS" + "=" * 20)
        print(
            disc.Scaler.__name__,
            ", pred_weight: ",
            disc.pred_weight,
            ", proxy_weight: ",
            disc.proxy_weight,
        )
        print("=" * 50)
        disc = Discover(
            figure_dir=join(
                "examples",
                "hyperparameter_combinations",
                "figures_"
                + Scaler.__name__
                + "_weights_"
                + str(weight_pair[0])
                + "-"
                + str(weight_pair[1]),
            ),  # e.g. "figures_RobustScaler_weights_0-1"
            table_dir=join(
                "examples",
                "hyperparameter_combinations",
                "tables_"
                + Scaler.__name__
                + "_weights_"
                + str(weight_pair[0])
                + "-"
                + str(weight_pair[1]),
            ),  # e.g. "tables_RobustScaler_weights_0-1"
            dummy_run=dummy,
            device="cuda",
            Scaler=Scaler,
            pred_weight=weight_pair[0],
            proxy_weight=weight_pair[1],
        )
        disc.fit(train_df)
        score = disc.predict(val_df, umap_random_state=42)
        print(disc.dens_score_df.head(100))
        print(disc.peak_score_df.head(100))
        print(disc.comb_out_df)
        disc.plot()

cdf_sorting_error

# %% Code Graveyard
# nested list comp: https://www.geeksforgeeks.org/nested-list-comprehensions-in-python/
# discs = [
#     [
#         Discover(
#             figure_dir=join(
#                 "examples",
#                 "hyperparameter_combinations",
#                 "figures_"
#                 + Scaler.__name__
#                 + "weights_"
#                 + str(weight_pair[0])
#                 + "-"
#                 + str(weight_pair[1]),
#             ),  # e.g. "figures_RobustScaler_weights_0-1"
#             table_dir=join(
#                 "examples",
#                 "hyperparameter_combinations",
#                 "tables_"
#                 + Scaler.__name__
#                 + "weights_"
#                 + str(weight_pair[0])
#                 + "-"
#                 + str(weight_pair[1]),
#             ),  # e.g. "tables_RobustScaler_weights_0-1"
#             dummy_run=dummy,
#             device="cuda",
#             Scaler=Scaler,
#             pred_weight=weight_pair[0],
#             proxy_weight=weight_pair[1],
#         )
#         for weight_pair in weight_pairs
#     ]
#     for Scaler in Scalers
# ]

"""
Use CrabNet outside of DiSCoVeR to do optimization; compare with random search.

# TODO: incorporate CrabNet uncertainty into search
"""
from crabnet.train_crabnet import get_model

# %% imports
from tqdm import tqdm
import dill as pickle
from copy import deepcopy
import numpy as np

from mat_discover.adaptive_design import ad_experiments_metrics
from mat_discover.utils.extraordinary import extraordinary_split

from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.adaptive_design import Adapt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %% setup
train_df, val_df = data(elasticity, "train.csv", dummy=False, random_state=42)
train_df, val_df, extraordinary_thresh = extraordinary_split(
    train_df, val_df, random_state=42
)
np.random.seed(42)

# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy_run = False
if dummy_run:
    val_df = val_df.iloc[:100]

# name_mapper = {"target": "Bulk Modulus (GPa)"}
# extraordinary_histogram(train_df, val_df, labels=name_mapper)

n_iter = 900  # of objective function evaluations (e.g. wet-lab synthesis)
n_repeats = 5

rand_experiments = []
for i in range(n_repeats):
    print(f"[RANDOM-EXPERIMENT: {i}]")
    rand_train_df = deepcopy(train_df)
    rand_val_df = deepcopy(val_df)
    adapt = Adapt(
        rand_train_df,
        rand_val_df,
        timed=False,
        dummy_run=dummy_run,
        device="cpu",
        dist_device="cpu",
    )
    rand_experiments.append(
        adapt.closed_loop_adaptive_design(
            n_experiments=n_iter, random_search=True, print_experiment=False
        )
    )


performance_experiments = []
for i in range(n_repeats):
    print(f"[PERFORMANCE-EXPERIMENT: {i}]")
    perf_train_df = deepcopy(train_df)
    perf_val_df = deepcopy(val_df)
    next_experiments = []
    for j in tqdm(range(n_iter)):
        crabnet_model = get_model(
            train_df=perf_train_df, verbose=False, learningcurve=False
        )
        val_true, val_pred, _, val_sigma = crabnet_model.predict(perf_val_df)
        perf_val_df["pred"] = val_pred
        perf_val_df["sigma"] = val_sigma
        idx = perf_val_df.pred.idxmax()
        # idx = np.where(val_pred == max(val_pred))[0][0]
        move_row = perf_val_df.loc[idx]
        perf_train_df.append(move_row)
        perf_val_df = perf_val_df.drop(index=idx)
        next_experiments.append(move_row.to_dict())
    experiment = ad_experiments_metrics(
        next_experiments, train_df, extraordinary_thresh
    )
    performance_experiments.append(experiment)

experiments = [
    rand_experiments,
    performance_experiments,
]

y_names = ["cummax", "target", "cumthresh", "n_unique_atoms", "n_unique_templates"]

rows = len(y_names)
cols = len(experiments)

x = list(range(n_iter))
y = np.zeros((rows, cols, n_repeats, n_iter))
formula = rows * [cols * [n_repeats * [None]]]
for (col, experiment) in enumerate(experiments):
    for (row, y_name) in enumerate(y_names):
        # loop through DataFrame rows: https://stackoverflow.com/a/11617194/13697228
        for (page, sub_experiment) in enumerate(experiment):
            y[row, col, page] = sub_experiment[y_name].values.tolist()
            formula[row][col][page] = sub_experiment["formula"].values.tolist()

labels = {
    "_index": "adaptive design iteration",
    "target": "Bulk Modulus (GPa)",
    "cummax": "Cumulative Max (GPa)",
    "cumthresh": "Cumulative Extraordinary (#)",
    "n_unique_atoms": "Unique Atoms (#)",
    "n_unique_templates": "Unique Chemical Templates (#)",
}
y_names = [labels[name] for name in y_names]

# def extraordinary_subplots():
fig = make_subplots(
    rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02
)

x_pars = ["Random", "Performance"]
col_nums = [str(i) for i in range((rows - 1) * cols + 1, rows * cols + 1)]
row_nums = [""] + [str(i) for i in list(range(cols + 1, rows * cols, cols))]

colors = ["red", "black", "green", "blue"]
for row in range(rows):
    for col in range(cols):
        color = colors[col]
        for page in range(n_repeats):
            fig.append_trace(
                go.Scatter(
                    x=x,
                    y=y[row, col, page],
                    line=dict(color=color),
                    text=formula[row][col][page],
                    hovertemplate="Formula: %{text} <br>Iteration: %{x} <br>Target (GPa): %{y})",
                ),
                row=row + 1,
                col=col + 1,
            )
for col_num, x_par in zip(col_nums, x_pars):
    fig["layout"][f"xaxis{col_num}"]["title"] = f"{x_par} AD Iteration (#)"

for row_num, y_name in zip(row_nums, y_names):
    fig["layout"][f"yaxis{row_num}"]["title"] = y_name

fig.update_traces(showlegend=False)
fig.update_layout(height=300 * rows, width=300 * cols)
fig.show()

with open("crabnet_performance.pkl", "wb") as f:
    pickle.dump(experiments, f)

1 + 1

# %% Code Graveyard
# performance_experiments = [
#     ad_experiments_metrics(exp, train_df, extraordinary_thresh)
#     for exp in performance_experiments
# ]

"""Compare DiSCoVeR to random search."""
# %% imports
import numpy as np

from mat_discover.utils.extraordinary import (
    extraordinary_split,
    extraordinary_histogram,
)
from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.adaptive_design import Adapt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %% setup
train_df, val_df = data(elasticity, "train.csv", dummy=False)
np.random.seed(42)
train_df, val_df, extraordinary_thresh = extraordinary_split(
    train_df, val_df, random_state=42
)

# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy_run = True
if dummy_run:
    val_df = val_df.iloc[:100]

# name_mapper = {"target": "Bulk Modulus (GPa)"}
# extraordinary_histogram(train_df, val_df, labels=name_mapper)

n_iter = 10  # of objective function evaluations (e.g. wet-lab synthesis)
n_repeats = 1

rand_experiments = [
    Adapt(
        train_df, val_df, dummy_run=dummy_run, device="cuda"
    ).closed_loop_adaptive_design(n_experiments=n_iter, random_search=True)
    for _ in range(n_repeats)
]

print("[Novelty-Experiments]")
novelty_experiments = [
    Adapt(
        train_df, val_df, dummy_run=dummy_run, device="cuda", pred_weight=0
    ).closed_loop_adaptive_design(n_experiments=n_iter)
    for _ in range(n_repeats)
]

print("[Equal-Experiments]")
equal_experiments = [
    Adapt(
        train_df, val_df, dummy_run=dummy_run, device="cuda"
    ).closed_loop_adaptive_design(n_experiments=n_iter)
    for _ in range(n_repeats)
]

print("[Performance-Experiments]")
performance_experiments = [
    Adapt(
        train_df, val_df, dummy_run=dummy_run, device="cuda", proxy_weight=0
    ).closed_loop_adaptive_design(n_experiments=n_iter)
    for _ in range(n_repeats)
]

experiments = [
    rand_experiments,
    novelty_experiments,
    equal_experiments,
    performance_experiments,
]

y_names = ["cummax", "target", "cumthresh", "n_unique_atoms", "n_unique_templates"]

rows = len(y_names)
cols = len(experiments)

# x = [[list(range(n_iter))] * cols] * rows
x = list(range(n_iter))
y = np.zeros((rows, cols, n_repeats, n_iter))
for (col, experiment) in enumerate(experiments):
    for (row, y_name) in enumerate(y_names):
        for (page, sub_experiment) in enumerate(experiment):
            y[row, col, page] = sub_experiment[y_name].values.tolist()

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

x_pars = ["Random", "Novelty", "50/50", "Performance"]
col_nums = [str(i) for i in range((rows - 1) * cols + 1, rows * cols + 1)]
row_nums = [""] + [str(i) for i in list(range(cols + 1, rows * cols, cols))]

colors = ["red", "black", "green", "blue"]
for row in range(rows):
    for col in range(cols):
        color = colors[col]
        for page in range(n_repeats):
            fig.append_trace(
                go.Scatter(x=x, y=y[row, col, page], line=dict(color=color)),
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


# TODO: elemental prevalence variety and distribution (periodic table?)
# TODO: chemical template variety and distribution (need a package)
# TODO: 3 different parameter weightings
# REVIEW: should I also include RobustScaler vs. MinMaxScaler?

1 + 1

# %% Code Graveyard
# fig.update_yaxes(dtick=np.log10(2))

# fig.update_layout(
#     yaxis = dict(
#         tickmode = 'array',
#         tickvals = [1, 3, 5, 7, 9, 11],
#         ticktext = ['One', 'Three', 'Five', 'Seven', 'Nine', 'Eleven']
#     )
# )

# rand_experiments = []
# for i in range(900):
#     rand_train_df, rand_val_df, next_experiment = rand_disc.suggest_next_experiment(
#         rand_train_df,
#         rand_val_df,
#         fit=False,
#         predict=False,
#         return_df=True,
#         random_search=True,
#         print_experiment=False,
#     )
#     rand_experiments.append(next_experiment)

# rand_df = pd.DataFrame(rand_experiments)
# rand_df["cummax"] = rand_df.target.cummax()

# rand_df["cumthresh"] = (rand_df.target >= extraordinary_lower).cumsum()

# rand_experiments = pd.concat([rand_experiments] * 3)
# ids = [0] * n_exp + [1] * n_exp + [2] * n_exp
# rand_experiments["ids"] = ids

# fig = px.line(
#     rand_experiments.rename(columns=labels),
#     x=rand_experiments.index,
#     y=y[0:2],
#     labels=labels,
#     facet_col="ids",
# )
# fig.show()

# y_pars = [
#     "Bulk Modulus (GPa)",
#     "Cumulative Max (GPa)",
#     "Cumulative Threshold (#)",
#     # "Unique Elements (#)",
# ]

# x_combs = dict(zip(x_pars, x_pars))
# for row, experiment in enumerate(experiments):
#     for col, y_name in enumerate(y):
#         experiments[col].rename(columns=labels).y_name

# x = [[list(range(n_iter))] * cols] * rows
# y = np.random.rand(rows, cols, n_iter)

# experiments = [rand_experiments] * 4

# rand_adapt = deepcopy(equal_adapt)

# novelty_adapt = deepcopy(equal_adapt)
# novelty_adapt.pred_weight = 0.0

# perf_adapt = deepcopy(equal_adapt)
# perf_adapt.proxy_weight = 0.0

from copy import deepcopy
import pandas as pd
import plotly.express as px

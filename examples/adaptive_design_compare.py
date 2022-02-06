"""Compare DiSCoVeR to random search."""
# %% imports
import dill as pickle
from os.path import join
import numpy as np

from mat_discover.utils.extraordinary import (
    extraordinary_split,
    extraordinary_histogram,
)

from crabnet.data.materials_data import elasticity
from mat_discover.utils.data import data
from mat_discover.adaptive_design import Adapt

from mat_discover.utils.plotting import matplotlibify
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %% setup
train_df, val_df = data(elasticity, "train.csv", dummy=False, random_state=42)
train_df, val_df, extraordinary_thresh = extraordinary_split(
    train_df, val_df, train_size=100, extraordinary_percentile=0.98, random_state=42
)
np.random.seed(42)
# REVIEW: why do I think this RNG affects anything downstream? (CrabNet, yes, but I'm
# having trouble thinking of where else an RNG would have an effect, other than
# rand_experiments, which makes me think - why do multiple repeats for the real ones?)

# set dummy to True for a quicker run --> small dataset, MDS instead of UMAP
dummy_run = False
if dummy_run:
    val_df = val_df.iloc[:100]
    n_iter = 3
    n_repeats = 1
else:
    n_iter = 900  # of objective function evaluations (e.g. wet-lab synthesis)
    n_repeats = 1

name_mapper = {"target": "Bulk Modulus (GPa)"}
extraordinary_histogram(train_df, val_df, labels=name_mapper)

rand_experiments = []

for i in range(n_repeats + 4):
    print(f"[RANDOM-EXPERIMENT: {i}]")
    adapt = Adapt(
        train_df,
        val_df,
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

novelty_experiments = []
for i in range(n_repeats):
    print(f"[NOVELTY-EXPERIMENT: {i}]")
    adapt = Adapt(
        train_df,
        val_df,
        timed=False,
        dummy_run=dummy_run,
        device="cuda",
        dist_device="cuda",
        pred_weight=0,
    )
    novelty_experiments.append(
        adapt.closed_loop_adaptive_design(n_experiments=n_iter, print_experiment=False)
    )

equal_experiments = []
for i in range(n_repeats):
    print(f"[EQUAL-EXPERIMENT: {i}]")
    adapt = Adapt(
        train_df,
        val_df,
        timed=False,
        dummy_run=dummy_run,
        device="cuda",
        dist_device="cuda",
    )
    equal_experiments.append(
        adapt.closed_loop_adaptive_design(n_experiments=n_iter, print_experiment=False)
    )

performance_experiments = []
for i in range(n_repeats):
    print(f"[PERFORMANCE-EXPERIMENT: {i}]")
    adapt = Adapt(
        train_df,
        val_df,
        timed=False,
        dummy_run=dummy_run,
        device="cuda",
        dist_device="cuda",
        proxy_weight=0,
    )
    performance_experiments.append(
        adapt.closed_loop_adaptive_design(n_experiments=n_iter, print_experiment=False)
    )

experiments = [
    rand_experiments,
    novelty_experiments,
    equal_experiments,
    performance_experiments,
    # performance_experiments_check,
]

y_names = ["cummax", "target", "cumthresh", "n_unique_atoms", "n_unique_templates"]

rows = len(y_names)
cols = len(experiments)

x = list(range(n_iter))
y = np.zeros((rows, cols, n_repeats + 4, n_iter))
formula = rows * [cols * [(n_repeats + 4) * [None]]]
for (col, experiment) in enumerate(experiments):
    for (row, y_name) in enumerate(y_names):
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

x_pars = ["Random", "Novelty", "50/50", "Performance"]
# x_pars = ["Random", "Performance", "Performance Check"]
col_nums = [str(i) for i in range((rows - 1) * cols + 1, rows * cols + 1)]
row_nums = [""] + [str(i) for i in list(range(cols + 1, rows * cols, cols))]

colors = ["red", "black", "green", "blue"]
for row in range(rows):
    for col in range(cols):
        color = colors[col]
        for page in range(n_repeats + 4):
            # if col == 0 or page == 0:
            if page == 0:
                fig.append_trace(
                    go.Scatter(
                        x=x,
                        y=y[row, col, page],
                        line=dict(color=color),
                        text=formula[row][col][page],
                        hovertemplate="Formula: %{text} <br>Iteration: %{x} <br>y: %{y}",
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

fig.write_html(join("figures", "ad-compare.html"))


fig2, scale = matplotlibify(
    fig, size=28, width_inches=3.5 * cols, height_inches=3.5 * rows
)
fig2.write_image(join("figures", "ad-compare.png"))

with open("rand_novelty_equal_performance.pkl", "wb") as f:
    pickle.dump(experiments, f)

# TODO: val RMSE vs. iteration
# TODO: elemental prevalence distribution (periodic table?)
# TODO: chemical template distribution (need a package)
# TODO: 3 different parameter weightings
# TODO: interactive, selectable interface to change what is displayed
# REVIEW: should I also include RobustScaler vs. MinMaxScaler?

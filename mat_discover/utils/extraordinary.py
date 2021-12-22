from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from mat_discover.utils.plotting import matplotlibify


def extraordinary_split(
    train_df,
    val_df=None,
    train_size=100,
    extraordinary_percentile=0.98,
    random_state=None,
):
    # set aside high-performing candidates
    if val_df is not None:
        train_val_df = train_df.append(val_df)
    else:
        train_val_df = train_df

    train_val_df = train_val_df.sort_values("target", axis=0, ascending=False)
    extraordinary_thresh = np.percentile(
        train_val_df.target.values, 100 * extraordinary_percentile
    )
    extraordinary_df = train_val_df[train_val_df.target >= extraordinary_thresh]
    train_val_df = train_val_df[train_val_df.target < extraordinary_thresh]

    train_df, val_df = train_test_split(
        train_val_df, train_size=train_size, random_state=random_state
    )
    val_df = val_df.append(extraordinary_df)

    return train_df, val_df, extraordinary_thresh


def extraordinary_histogram(
    train_df,
    val_df,
    log_y=False,
    labels=None,
    fpath=join("figures", "extraordinary_histogram"),
):

    # why should I make a copy of a data frame in pandas https://stackoverflow.com/a/27680109/13697228
    train_df = train_df.copy()
    train_df["split"] = "training"
    val_df = val_df.copy()
    val_df["split"] = "validation"

    train_val_df = pd.merge(train_df, val_df, how="outer")

    fig = px.histogram(
        train_val_df,
        x="target",
        color="split",
        labels=labels,
        barmode="overlay",
        marginal="rug",
        hover_data=train_df.columns,
        log_y=log_y,
    )

    fig.show()

    if fpath is not None:
        fig.write_html(fpath + ".html")

    fig = px.histogram(
        train_val_df,
        x="target",
        color="split",
        labels=labels,
        barmode="overlay",
        marginal="violin",
        hover_data=train_df.columns,
        log_y=log_y,
    )

    fig.update_layout(
        legend_orientation="h",
        legend_y=1.025,
        legend_yanchor="bottom",
    )

    fig, scale = matplotlibify(fig, size=20)

    if fpath is not None:
        fig.write_image(fpath + ".png", scale=scale)


# %% Code Graveyard
# n_extraordinary = int(
#     np.floor((1 - extraordinary_percentile) * train_val_df.shape[0])
# )
# extraordinary_df = train_val_df.head(n_extraordinary)
# extraordinary_lower = extraordinary_df.tail(1).target.values[0]
# train_val_df = train_val_df.iloc[n_extraordinary:]

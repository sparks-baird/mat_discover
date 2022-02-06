"""Various plotting functions for cluster properties and UMAP visualization."""
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

# import seaborn as sns

# TODO: change to square plots
def umap_cluster_scatter(std_emb, labels, figure_dir="figures"):
    """Plot UMAP embeddings colored by cluster IDs.

    Parameters
    ----------
    std_emb : 2d array
        UMAP embedded coordinates.
    labels : 1d array
        Cluster IDs associated with the UMAP coordinates.

    Returns
    -------
    fig : Figure
        Handle to Matplotlib Figure.
    """
    # TODO: update plotting commands to have optional arguments (e.g. std_emb and labels)
    cmap = plt.cm.nipy_spectral
    mx = np.max(labels)
    # cmap = sns.color_palette("Spectral", mx + 1, as_cmap=True)
    class_ids = labels != -1
    fig = plt.Figure()
    ax = plt.scatter(
        std_emb[:, 0], std_emb[:, 1], c=labels, s=0.1, cmap=cmap, label=labels
    )
    unclass_ids = np.invert(class_ids)
    unclass_frac = np.sum(unclass_ids) / len(labels)
    plt.axis("off")

    if unclass_frac != 0.0:
        ax2 = plt.scatter(
            std_emb[unclass_ids, 0],
            std_emb[unclass_ids, 1],
            c=labels[unclass_ids],
            s=0.1,
            cmap=plt.cm.nipy_spectral,
            label=labels[unclass_ids],
        )
        # How to put the legend out of the plot: https://stackoverflow.com/a/4701285/13697228
        plt.legend(
            [ax2],
            ["Unclassified: " + "{:.1%}".format(unclass_frac)],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
        )
    plt.tight_layout()
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join(figure_dir, "umap-cluster-scatter"))
    plt.show()
    return fig

    # TODO: update label ints so they don't overlap so much (skip some based on length of labels)
    lbl_ints = np.arange(np.amax(labels) + 1)
    if unclass_frac != 1.0:
        plt.colorbar(ax, boundaries=lbl_ints - 0.5, label="Cluster ID").set_ticks(
            lbl_ints
        )
    plt.show()


def cluster_count_hist(labels, figure_dir="figures"):
    """Plot histogram of cluster counts, colored by cluster IDs.

    Parameters
    ----------
    labels : 1d array
        Cluster IDs.

    Returns
    -------
    fig : Figure
        Handle to Matplotlib Figure.
    """
    col_scl = MinMaxScaler()
    unique_labels = np.unique(labels)
    col_trans = col_scl.fit(unique_labels.reshape(-1, 1))
    scl_vals = col_trans.transform(unique_labels.reshape(-1, 1))
    color = plt.cm.nipy_spectral(scl_vals)
    # mx = np.max(labels)
    # cmap = sns.color_palette("Spectral", mx + 1, as_cmap=True)
    # color = cmap(scl_vals)

    fig = plt.Figure()
    plt.bar(*np.unique(labels, return_counts=True), color=color)
    plt.xlabel("cluster ID")
    plt.ylabel("number of compounds")
    plt.tight_layout()
    plt.savefig(join(figure_dir, "cluster-count-hist"))
    plt.show()
    return fig


def target_scatter(std_emb, target, figure_dir="figures"):
    """Plot UMAP embedding locations colored by target values.

    Parameters
    ----------
    std_emb : 2d array
        UMAP embedding coordinates.
    target : 1d array
        Target properties corresponding to `std_emb`.

    Returns
    -------
    fig : Figure
        Handle to Matplotlib Figure.
    """
    # TODO: change to log colorscale or a higher-contrast
    fig = plt.Figure()
    plt.scatter(
        std_emb[:, 0],
        std_emb[:, 1],
        c=target,
        s=0.1,
        cmap="Spectral_r",
        norm=mpl.colors.LogNorm(),
    )
    plt.axis("off")
    plt.colorbar(label="Bulk Modulus (GPa)", orientation="horizontal")
    plt.tight_layout()
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join(figure_dir, "target-scatter"))
    plt.show()
    return fig


def dens_scatter(x, y, pdf_sum, figure_dir="figures"):
    """Plot DensMAP densities at the `x` and `y` embedding coordinates.

    Parameters
    ----------
    x : 1d array
        x-coordinates
    y : 1d array
        y-coordinates
    pdf_sum : 1d array
        probabilities evaluated at each of the `x` and `y` coordinate pairs.

    Returns
    -------
    fig : Figure
        Handle to Matplotlib Figure.

    See Also
    --------
    mat_discover_.mvn_prob_sum : used to obtain `x`, `y`, and `pdf_sum`
    """
    # TODO: add callouts to specific locations (high-scoring compounds)
    fig = plt.Figure()
    plt.scatter(x, y, c=pdf_sum)
    plt.axis("off")
    plt.tight_layout()
    plt.colorbar(label="Density", orientation="horizontal")
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join(figure_dir, "dens-scatter"))
    plt.show()
    return fig


def dens_targ_scatter(std_emb, target, x, y, pdf_sum, figure_dir="figures"):
    """Plot overlay of density scatter and target scatter plots.

    Parameters
    ----------
    std_emb : 2d array
        UMAP embedding coordinates.
    target : 1d array
        Target properties corresponding to `std_emb`.
    x : 1d array
        x-coordinates
    y : 1d array
        y-coordinates
    pdf_sum : 1d array
        probabilities evaluated at each of the `x` and `y` coordinate pairs.

    Returns
    -------
    fig : Figure
        Handle to Matplotlib Figure.

    See Also
    --------
    dens_scatter : density scatter plot
    targ_scatter : target scatter plot
    """
    fig = plt.Figure()
    plt.scatter(x, y, c=pdf_sum)
    plt.scatter(
        std_emb[:, 0],
        std_emb[:, 1],
        c=target,
        s=2,
        cmap="Spectral",
        edgecolors="none",
        alpha=0.15,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join(figure_dir, "dens-targ-scatter"))
    plt.show()
    return fig


def group_cv_parity(ytrue, ypred, labels, figure_dir="figures"):
    """Leave-one-cluster-out cross-validation parity plot colored by `labels`.

    Parameters
    ----------
    ytrue : 1d array
        True target values.
    ypred : 1d array
        Predicted target values.
    labels : 1d array
        Cluster IDs.

    Returns
    -------
    fig : Figure
        Handle to Matplotlib Figure.
    """
    labels = np.array(labels)
    col_scl = MinMaxScaler()
    col_trans = col_scl.fit(labels.reshape(-1, 1))
    scl_vals = col_trans.transform(labels.reshape(-1, 1))
    color = plt.cm.nipy_spectral(scl_vals)

    mx = np.nanmax([ytrue, ypred])

    fig = plt.scatter(ytrue, ypred, c=color)
    plt.plot([0, 0], [mx, mx], "--", linewidth=1)

    plt.xlabel(r"$E_\mathregular{avg,true}$ (GPa)")
    plt.ylabel(r"$E_\mathregular{avg,pred}$ (GPa)")
    plt.tight_layout()
    plt.savefig(join(figure_dir, "group-cv-parity"))
    plt.show()
    return fig


def matplotlibify(fig, size=24, width_inches=3.5, height_inches=3.5, dpi=142):
    # make it look more like matplotlib
    # modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
    font_dict = dict(family="Arial", size=size, color="black")

    # app = QApplication(sys.argv)
    # screen = app.screens()[0]
    # dpi = screen.physicalDotsPerInch()
    # app.quit()

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )
    fig.update(layout_coloraxis_showscale=False)

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    return fig, scale

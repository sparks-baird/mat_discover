from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler

# import seaborn as sns

# TODO: change to square plots
def umap_cluster_scatter(std_emb, labels):
    # TODO: update plotting commands to have optional arguments (e.g. std_emb and labels)
    cmap = plt.cm.nipy_spectral
    mx = np.max(labels)
    # cmap = sns.color_palette("Spectral", mx + 1, as_cmap=True)
    class_ids = labels != -1
    fig = plt.Figure()
    ax = plt.scatter(
        std_emb[:, 0],
        std_emb[:, 1],
        c=labels,
        s=2,
        cmap=cmap,
        label=labels,
    )
    unclass_ids = np.invert(class_ids)
    unclass_frac = np.sum(unclass_ids) / len(labels)
    plt.axis("off")

    if unclass_frac != 0.0:
        ax2 = plt.scatter(
            std_emb[unclass_ids, 0],
            std_emb[unclass_ids, 1],
            c=labels[unclass_ids],
            s=2,
            cmap=plt.cm.nipy_spectral,
            label=labels[unclass_ids],
        )
        plt.legend([ax2], ["Unclassified: " + "{:.1%}".format(unclass_frac)])
    # plt.tight_layout()
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join("figures", "umap-cluster-scatter"))
    plt.show()
    return fig

    # TODO: update label ints so they don't overlap so much (skip some based on length of labels)
    lbl_ints = np.arange(np.amax(labels) + 1)
    if unclass_frac != 1.0:
        plt.colorbar(ax, boundaries=lbl_ints - 0.5, label="Cluster ID").set_ticks(
            lbl_ints
        )
    plt.show()


def cluster_count_hist(labels):
    col_scl = MinMaxScaler()
    unique_labels = np.unique(labels)
    col_trans = col_scl.fit(unique_labels.reshape(-1, 1))
    scl_vals = col_trans.transform(unique_labels.reshape(-1, 1))
    color = plt.cm.nipy_spectral(scl_vals)
    # mx = np.max(labels)
    # cmap = sns.color_palette("Spectral", mx + 1, as_cmap=True)
    # color = cmap(scl_vals)

    fig = plt.bar(*np.unique(labels, return_counts=True), color=color)
    plt.xlabel("cluster ID")
    plt.ylabel("number of compounds")
    # plt.tight_layout()
    plt.savefig(join("figures", "cluster-count-hist"))
    plt.show()
    return fig


def target_scatter(std_emb, target):
    # TODO: change to log colorscale or a higher-contrast
    fig = plt.scatter(
        std_emb[:, 0],
        std_emb[:, 1],
        c=target,
        s=2,
        cmap="Spectral_r",
        norm=mpl.colors.LogNorm(),
    )
    plt.axis("off")
    plt.colorbar(label="Bulk Modulus (GPa)", orientation="horizontal")
    # plt.tight_layout()
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join("figures", "target-scatter"))
    plt.show()
    return fig


def dens_scatter(x, y, pdf_sum):
    # TODO: add callouts to specific locations (high-scoring compounds)
    fig = plt.scatter(x, y, c=pdf_sum)
    plt.axis("off")
    # plt.tight_layout()
    plt.colorbar(label="Density", orientation="horizontal")
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join("figures", "dens-scatter"))
    plt.show()
    return fig


def dens_targ_scatter(std_emb, target, x, y, pdf_sum):
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
    # plt.tight_layout()
    plt.gca().set_aspect("equal", "box")
    plt.savefig(join("figures", "dens-targ-scatter"))
    plt.show()
    return fig


def group_cv_parity(ytrue, ypred, labels):
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
    # plt.tight_layout()
    plt.savefig(join("figures", "group-cv-parity"))
    plt.show()
    return fig

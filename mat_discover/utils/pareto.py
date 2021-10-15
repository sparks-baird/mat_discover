"""Helper functions for finding and plotting a pareto front."""
from os.path import join

# import sys
# from PyQt5.QtWidgets import QApplication

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points.

    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient

    Fairly fast for many datapoints, less fast for many costs, somewhat readable

    Modified from: https://stackoverflow.com/a/40239615/13697228
    """
    mx = np.max(costs)
    costs = np.nan_to_num(costs, nan=mx)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def get_pareto_ind(proxy, target, reverse_x=True):
    # use reverse_x if using "peak"
    if reverse_x:
        inpt = [proxy, -target]
    else:
        inpt = [-proxy, -target]
    pareto_ind = np.nonzero(is_pareto_efficient_simple(np.array(inpt).T))
    return pareto_ind


def pareto_plot(
    df,
    x="neigh_avg_targ (GPa)",
    y="target (GPa)",
    color="Peak height (GPa)",
    hover_data=["formula"],
    fpath=join("figures", "pareto-front"),
    reverse_x=True,
    parity_type="max-of-both",
    pareto_front=True,
    color_continuous_scale=None,
    color_discrete_map=None,
    xrange=None,
):
    """Generate and save pareto plot for two variables.

    Parameters
    ----------
    df : DataFrame
        Contains relevant variables for pareto plot.
    x : str, optional
        Name of df column to use for x-axis, by default "proxy"
    y : str, optional
        Name of df column to use for y-axis, by default "target"
    color : str, optional
        Name of df column to use for colors, by default "Peak height (GPa)"
    hover_data : list of str, optional
        Name(s) of df columns to display on hover, by default ["formulas"]
    fpath : str, optional
        Filepath to which to save HTML and PNG. Specify as None if no saving
        is desired, by default "pareto-plot"
    reverse_x : bool, optional
        Whether to reverse the x-axis (i.e. for maximize y and minimize x front)
    parity_type : str, optional
        What kind of parity line to plot: "max-of-both", "max-of-each", or "none"
    """
    mx = np.max(df[color])
    if color_continuous_scale is None and color_discrete_map is None:
        if isinstance(df[color].iloc[0], (int, np.integer)):
            # if mx < 24:
            #     df.loc[:, color] = df[color].astype(str)

            # color_discrete_map = px.colors.qualitative.Dark24
            # color_discrete_map = sns.color_palette("Spectral", mx + 1, as_cmap=True)
            # scatter_color_kwargs = {"color_continuous_scale": color_discrete_map}

            def mpl_to_plotly(cmap, pl_entries=11, rdigits=2):
                # cmap - colormap
                # pl_entries - int = number of Plotly colorscale entries
                # rdigits - int -=number of digits for rounding scale values
                scale = np.linspace(0, 1, pl_entries)
                colors = (cmap(scale)[:, :3] * 255).astype(np.uint8)
                pl_colorscale = [
                    [round(s, rdigits), f"rgb{tuple(color)}"]
                    for s, color in zip(scale, colors)
                ]
                return pl_colorscale

            nipy_spectral = mpl_to_plotly(
                plt.cm.nipy_spectral, pl_entries=mx + 1, rdigits=3
            )

            scatter_color_kwargs = {
                "color_continuous_scale": nipy_spectral  # px.colors.sequential.Blackbody_r
            }
    elif color_continuous_scale is not None:
        scatter_color_kwargs = {"color_continuous_scale": color_continuous_scale}
    elif color_discrete_map is not None:
        scatter_color_kwargs = {"color_discrete_sequence": color_discrete_map}

    # trace order counts 0, 1, 2, ... instead of 0, 1, 10, 11
    df["color_num"] = df[color].astype(int)
    df = df.sort_values("color_num")

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        **scatter_color_kwargs,
    )

    # unpack
    proxy = df[x]
    target = df[y]

    if pareto_front:
        pareto_ind = get_pareto_ind(proxy, target, reverse_x=reverse_x)

        # pf_hover_data = df.loc[:, hover_data].iloc[pareto_ind]
        # fig.add_scatter(x=proxy[pareto_ind], y=target[pareto_ind])
        # Add scatter trace with medium sized markers
        sorter = np.flip(np.argsort(target.iloc[pareto_ind]))
        fig.add_scatter(
            mode="lines",
            line={"color": "black", "width": 1, "dash": "dash"},
            x=proxy.iloc[pareto_ind].iloc[sorter],
            y=target.iloc[pareto_ind].iloc[sorter],
            marker_symbol="circle-open",
            marker_size=10,
            hoverinfo="skip",
            name="pareto front",
        )
    else:
        pareto_ind = None

    # parity line
    if parity_type == "max-of-both":
        mx = np.nanmax([proxy, target])
        mx2 = mx
    elif parity_type == "max-of-each":
        mx, mx2 = np.nanmax(proxy), np.nanmax(target)

    if parity_type is not None:
        fig.add_trace(go.Line(x=[0, mx], y=[0, mx2], name="parity"))

    # legend and reversal
    fig.update_layout(legend_orientation="h", legend_y=1.1, legend_yanchor="bottom")

    if reverse_x:
        fig.update_layout(xaxis=dict(autorange="reversed"))

    fig.show()

    if fpath is not None:
        fig.write_html(fpath + ".html")

    # make it look more like matplotlib
    # modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
    font_dict = dict(family="Arial", size=24, color="black")

    # app = QApplication(sys.argv)
    # screen = app.screens()[0]
    # dpi = screen.physicalDotsPerInch()
    # app.quit()
    dpi = 142

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=3.5 * dpi,
        height=3.5 * dpi,
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

    if xrange is not None:
        fig.update_xaxes(range=xrange)

    # saving
    if fpath is not None:
        width_inches = 3.5
        width_default_px = fig.layout.width
        targ_dpi = 300
        scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

        fig.write_image(fpath + ".png", scale=scale)

    return fig, pareto_ind

"""Helper functions for finding and plotting a pareto front."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
    fpath="pareto-front",
    reverse_x=True,
    parity_type="max-of-both",
    pareto_front=True,
    color_continuous_scale=None,
    color_discrete_map=None,
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
    if color_continuous_scale is None and color_discrete_map is None:
        if isinstance(df[color][0], (int, np.integer)):
            if np.max(df[color]) < 20:
                df.loc[:, color] = df[color].astype(str)
            scatter_color_kwargs = {
                "color_discrete_sequence": px.colors.qualitative.Dark24
            }
        else:
            scatter_color_kwargs = {
                "color_continuous_scale": px.colors.sequential.Blackbody_r
            }
    elif color_continuous_scale is not None:
        scatter_color_kwargs = {"color_continuous_scale": color_continuous_scale}
    elif color_discrete_map is not None:
        scatter_color_kwargs = {"color_discrete_sequence": color_discrete_map}

    # TODO: update trace order
    df = df.sort_values(color)
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
        fig.add_scatter(
            mode="markers",
            x=proxy.iloc[pareto_ind],
            y=target.iloc[pareto_ind],
            marker_symbol="circle-open",
            marker_size=10,
            hoverinfo="skip",
            name="Pareto Front",
        )

    # parity line
    if parity_type == "max-of-both":
        mx = np.nanmax([proxy, target])
        mx2 = mx
    elif parity_type == "max-of-each":
        mx, mx2 = np.nanmax(proxy), np.nanmax(target)

    if parity_type is not None:
        fig.add_trace(go.Line(x=[0, mx], y=[0, mx2], name="parity"))

    # legend and reversal
    fig.update_layout(legend_orientation="h", legend_y=1.1)
    if reverse_x:
        fig.update_layout(xaxis=dict(autorange="reversed"))
    fig.show()

    # saving
    if fpath is not None:
        fig.write_image(fpath + ".png")
        fig.write_html(fpath + ".html")

    return fig, pareto_ind

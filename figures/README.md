# Figures

> :warning: IF YOU ARE ON THE [THE GITHUB WEBSITE](https://github.com/sparks-baird/mat_discover/tree/main/figures), PLEASE INSTEAD NAVIGATE TO [THE DOCUMENTATION WEBSITE](https://mat-discover.readthedocs.org/en/latest/figures.html) :warning:

Various figures, both interactive and non-interactive, related to the DiSCoVeR algorithm as applied to compounds and clusters. For more details, see [the ChemRXiv paper](https://dx.doi.org/10.33774/chemrxiv-2021-5l2f8).

## Interactive Figures

Compound-wise and cluster-wise interactive figures produced via [Plotly](https://plotly.com/python/).

### Instructions

1. Navigate to https://mat-discover.readthedocs.org/en/latest/figures.html
3. Hover your mouse over datapoints to see more information (e.g. composition, bulk modulus, cluster ID)

Alternatively, you can download the `html` file at [https://github.com/sparks-baird/mat_discover/figures](https://github.com/sparks-baird/mat_discover/figures) and open it in your browser of choice.

### Compound-Wise

#### Pareto Front Training Contribution to Validation Log Density Proxy

[<img src="pf-train-contrib-proxy.png" alt="Pareto Front Training Contribution to Validation Log Density Proxy" width=350>](https://mat-discover.readthedocs.io/en/latest/figures.html#pareto-front-training-contribution-to-validation-log-density-proxy)
<hr>

#### Pareto Front k-Nearest Neighbor Average Proxy  

[<img src="pf-peak-proxy.png" alt="Pareto Front k-Nearest Neighbor Average Proxy" width=350>](https://mat-discover.readthedocs.io/en/latest/figures.html#pareto-front-k-nearest-neighbor-average-proxy)
<hr>

#### Pareto Front Density Proxy (Train and Validation Points)

[<img src="pf-dens-proxy.png" alt="Pareto Front Density Proxy (Train and Validation Points)" width=350>](https://mat-discover.readthedocs.io/en/latest/figures.html#pareto-front-density-proxy-train-and-validation-points)
<hr>

### Cluster-Wise

#### Pareto Front Validation Fraction

[<img src="pf-frac-proxy.png" alt="Pareto Front Validation Fraction" width=350>](https://mat-discover.readthedocs.io/en/latest/figures.html#pareto-front-validation-fraction)
<hr>

#### Leave-one-cluster-out Cross-validation k-Nearest Neighbor Average Parity Plot

[<img src="gcv-pareto.png" alt="Leave-one-cluster-out Cross-validation k-Nearest Neighbor Average Parity Plot" width=350>](https://mat-discover.readthedocs.io/en/latest/figures.html#leave-one-cluster-out-cross-validation-k-nearest-neighbor-average-parity-plot)
<hr>

## Non-interactive Figures

The following are non-interactive figures focused on cluster properties and density/target characteristics within DensMAP embedding space.

### Cluster Properties

Cluster Count Histogram  
<img src="cluster-count-hist.png" alt="Cluster Count Histogram" width=350>
<hr>

DensMAP Embeddings Colored by Cluster  
[<img src="umap-cluster-scatter.png" alt="DensMAP Embeddings Colored by Cluster" width=350>](https://mat-discover.readthedocs.io/en/latest/figures.html#densmap-embeddings-colored-by-cluster)
<hr>

### Density and Target Characteristics

Density Scatter Plot of 2D DensMAP Embeddings  
<img src="dens-scatter.png" alt="Density Scatter Plot of 2D DensMAP Embeddings" width=350>
<hr>

Target Scatter Plot of 2D DensMAP Embeddings  
<img src="target-scatter.png" alt="Density Scatter Plot of 2D DensMAP Embeddings" width=350>
<hr>

Density Scatter Plot with Bulk Modulus Overlay in 2D DensMAP Embedding Space  
<img src="dens-targ-scatter.png" alt="Density Scatter Plot with Bulk Modulus Overlay in 2D DensMAP Embedding Space" width=350>
<hr>

### Adaptive Design Comparison

Take an initial training set of 100 chemical formulas and associated Materials Project bulk moduli followed by 900 adaptive design iterations (x-axis) using random search, novelty-only (performance weighted at 0), a 50/50 weighting split, and performance-only (novelty weighted at 0). These are the columns. The rows are the total number of observed "extraordinary" compounds (top 2%), the total number of additional unique atoms, and total number of additional unique chemical formulae templates. These are the rows. In other words:

- How many "extraordinary" compounds have been observed so far?
- How many unique atoms have been explored so far? (not counting atoms already in the starting 100 formulas)
- How many unique chemical templates (e.g. A2B3, ABC, ABC2) have been explored so far? (not counting templates already in the starting 100 formulas)

Random search is compared with DiSCoVeR performance/proxy weights. The 50/50 weighting split offers a good trade-off between performance and novelty.

[<img src="ad-compare.png" alt="Pareto Front Training Contribution to Validation Log Density Proxy" width=800>](https://mat-discover.readthedocs.io/en/latest/figures.html#adaptive-design-comparison)


<!-- 
# Code Graveyard
[Pareto Front Training Contribution to Validation Log Density Proxy](pf-train-contrib-proxy.html)  
![Pareto Front Training Contribution to Validation Log Density Proxy](pf-train-contrib-proxy.png)](pf-train-contrib-proxy.html)

[Pareto Front k-Nearest Neighbor Average Proxy](pf-peak-proxy.html)
[![Pareto Front k-Nearest Neighbor Average Proxy](pf-peak-proxy.png)](pf-peak-proxy.html)

[Pareto Front Density Proxy (Train and Validation Points)](pf-dens-proxy.html)  
[![Pareto Front Density Proxy (Train and Validation Points)](pf-dens-proxy.png)](pf-dens-proxy.html)

[Pareto Front Validation Fraction](pf-frac-proxy.html)
[![Pareto Front Validation Fraction](pf-frac-proxy.png)](pf-frac-proxy.html)

[Leave-one-cluster-out Cross-validation k-Nearest Neighbor Average Parity Plot](gcv-pareto.html)  
[![Leave-one-cluster-out Cross-validation k-Nearest Neighbor Average Parity Plot](gcv-pareto.png)](gcv-pareto.html)
 -->

<!--  <a href=*.html>
   <img alt="" src=*.png width=350>
</a> -->

<!-- <img alt="" src=*.png width=350> -->

<!-- <body>
   <a href=pf-train-contrib-proxy.html>
      <img alt="Pareto Front Training Contribution to Validation Log Density Proxy" src=pf-train-contrib-proxy.png width=350>
   </a>
</body> -->

<!-- If you are on the GitHub page (i.e. [https://github.com/sparks-baird/mat_discover/figures](https://github.com/sparks-baird/mat_discover/figures)) and would like to view the interactive figures without downloading the HTML files, please first navigate to [https://sparks-baird.github.io/mat_discover/figures/](https://sparks-baird.github.io/mat_discover/figures/) and then click the image you want to view. -->

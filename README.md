# Fixed-bandwidth vs. adaptive-bandwidth KDE for flow duration curves

Jump to [Reproducing the results](#reproducing-the-results).

## Background

The kernel bandwidth is a hyperparameter that governs smoothing of the marginal density of finite observations. The optimality of a fixed ("plug-in") kernel bandwidth estimator has been documented on the basis of a Gaussian parametric assumption on the "roughness" of the pdf $[\int f''(x)]^2 dx$.

We propose setting the bandwidth as a function of (an assumed piecewise linear) measurement uncertainty, using the heteroscedastic precision floor of daily streamflow records as prior knowledge. We demonstrate the approach on the [Caravan](https://github.com/kratzert/Caravan) large-sample dataset, comparing flow duration curves estimated with the fixed (FB) and adaptive (AB) methods across geographically distinct regions.

Both methods assume a Gaussian kernel:

$$f(x) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{h \sqrt{2 \pi}} \exp \left( \frac{-(x - x_i)^2}{2h^2} \right)$$

The Silverman (1998) rule-of-thumb bandwidth is:

$$\hat{h} = 1.06 \min(\hat{\sigma},\, \text{IQR}) \, N^{-1/5}$$

where $\hat{\sigma}$ is the sample standard deviation, IQR is the interquartile range, and $N$ is the sample size.

The adaptive bandwidth uses a piecewise linear measurement error model to set a per-observation bandwidth floor. The error model is shown below:

![An approximate error model for daily streamflows](images/error_model.png)

Both the error function and the minimum flow threshold are site-specific assumptions. Probabilistic rating curves, when available, could replace the empirical model used here.

## Methodology

1. Compute the global unit area runoff (UAR) range $[\text{UAR}_\text{min},\, \text{UAR}_\text{max}]$ across all stations in the region.
2. Extract unique flow values per station to define the sample support.
3. Exclude stations with a single unique value or fewer than one complete hydrological year.
4. Apply the piecewise error model to each unique value $q$, returning relative error $\varepsilon(q)$.
5. Compute the bandwidth floor in log space: $b_{\text{floor}} = \log(1 + \varepsilon)$.
6. Compute log-space Voronoi half-widths $\Delta_i$ between adjacent unique values.
7. Set the per-observation bandwidth: $h_i = \max(\Delta_i,\, b_{\text{floor},i})$.
8. Assign each observation its bandwidth by index lookup in the sorted unique-value array.

The result is a vector $\mathbf{h}$, one bandwidth per observation, used as the standard deviation of the Gaussian kernel centred on $\log(u_i)$.

9. Select bin count $M$ such that quantization error is near the 5% nominal measurement error. For the evaluated sample, 256 bins (8-bit) over the global UAR range yields approximately 4% error.
10. Log-transform all observations: $z_i = \log(u_i)$.
11. Evaluate the adaptive KDE on the log-space grid:

$$\hat{f}(z) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{h_i \sqrt{2\pi}} \exp\!\left(\frac{-(z - z_i)^2}{2 h_i^2}\right)$$

12. Convert the PDF to a PMF: $p_j = \hat{f}(z_j) \cdot w_j$, then normalize so $\sum_j p_j = 1$.

The Silverman estimator uses [KDEpy](https://kdepy.readthedocs.io/en/latest/), which evaluates the kernel via FFT convolution in $O(n \log n)$.

Divergence between the two estimators is measured with the Kolmogorov-Smirnov (KS) statistic: the maximum absolute difference between the two CDFs over their shared support.

## Data

Download the Caravan dataset from [huggingface.co/datasets/kratzert/Caravan](https://huggingface.co/datasets/kratzert/Caravan). Set `Config.caravan_dir` in `scripts/config.py` to the local path of the extracted `Caravan-csv` folder before running any scripts. The folder must contain an `attributes/` subdirectory and a `timeseries/csv/` subdirectory, both organized by region.

## Reproducing the results

1. Clone the repository.
2. Create a virtual environment and install dependencies (`uv` recommended):

```bash
uv venv
uv pip install -r requirements.txt
```

3. Set `Config.caravan_dir` in `scripts/config.py` to the local path of the Caravan dataset.

4. Run preprocessing to compute reference distributions:

```bash
python scripts/preprocess.py [region|index|all]
```

Outputs written to `data/baseline_distributions/{region}/{N}_bits/`:
- `pmf_obs.csv`: observed empirical PMF
- `pmf_kde_adaptive.csv`: adaptive-bandwidth KDE PMF
- `pmf_kde_silverman.csv`: Silverman KDE PMF
- `pmf_lnmle.csv`: log-normal MLE PMF

Outputs written to `cache/{region}/`:
- `complete_year_stats.parquet`: complete Oct-Sep hydrological years per station
- `station_meta.parquet`: station metadata filtered by record length and UAR bounds
- `weibull_quantiles.parquet`: empirical ECDF quantiles for dip-test input

5. Run the analysis to compute per-station divergence scores:

```bash
python scripts/run_analysis.py [region|index|all]
```

Outputs written to `cache/`:
- `{region}_kde_comparison.parquet`: per-station KS, $W_1$, ED, ISD, and KL scores at 6, 8, and 10 bits
- `{region}_worst10_pmfs.parquet`: PMF data for the 10 worst-case stations per metric at 8 bits
- `{region}_median_pmfs.parquet`: PMF data for 8 median stations per metric at 8 bits
- `{region}_bin_concentration.parquet`: top-bin mass fraction statistics per station

6. Build the interactive HTML report:

```bash
python scripts/build_report.py [region|index|all]
```

Output: `scripts/max_kde_diffs.html`

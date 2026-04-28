# Fixed "plug-in" vs. adaptive bandwidth estimation for Kernel Density Estimation

## Supporting code and information

The kernel bandwidth is a hyperparameter governing smoothing of the PDF from finite observations.  Several rules of thumb have been proposed on the basis of expected minimum mean integrated square error, however these make a parametric assumption in approximating the curvature of the pdf $f(x)$.

As an alternative, owing to the heteroscedastic uncertainty of daily streamflow, we propose setting the kernel bandwidth as a function of measurement uncertainty.  This approach incorporates measurement uncertainty as prior knowledge in setting the kernel bandwidth, representing an assumed precision floor on measurements instead of minimizing integrated square error on a set of measurements whose reported precision is generally not justified by the uncertainty in the data.

We demonstrate the method using a large sample of streamflow records from North America (HYSETS) by computing flow duration curves from empirical records using a rule of thumb method and compare it to the proposed adaptive bandwidth kernel method.  Both methods assume a Gaussian kernel, with the general form of the kernel density estimator given by:

$$f(x) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{h \sqrt{2 \pi}} \exp \left( \frac{-(x - x_i)^2}{2h^2} \right)$$

The rule of thumb bandwidth $h$ approximation comes from Silverman (1998):
$$\hat f(x) = 1.06 \sigma N^{1/5}$$

The two estimates are the sample standard deviation, $\sigma$, and a rescaled IQR (which agrees with $\sigma$ asymptotically for normal distributions).

Where:
* sigma is the sample standard deviation
* N is the sample size

In estimating daily average streamflow records from a stage discharge (rating curve), the Water Survey of Canada data collection standards state a simple rate of 5% error "unless otherwise specified" by categorical, qualitative data quality information.  Uncertainty is however greater at the extremes, in particular where flows are extrapolated from calibration points in the rating curve.  As a result, for the purpose of presenting an example we apply an empirical, piecewise linear approximation of error as well as a minimum flow value below which we define flows to be unverifiable or unresolvable. The error model is shown in the figure below:

[An approximate error model for daily streamflows](akde_bandwidth_error_model.png)

Both the piecewise linear error function and the minimum flow threshold are assumptions that vary from site to site.  Rating curve points and full replication information for daily flow estimates are generally unavailable for stations in North America, but in the future probabilistic rating curves would allow for probabilistic rating curve development which could be used to set the kernel bandwidth adaptively to avoid encoding spurious precision in the flow duration curve.

## Methodology

1. Obtain the sample-wide range of flow and unit area runoff for a large sample of streamflow observations.  $[\text{UAR}_\text{global}^\text{min}, \text{UAR}_\text{global}^\text{max}]$
2.  Filter out stations with only a single unique value, and filter out stations with less than one year of complete records.
3. Convert unit area runoff to volumetric flow: $q = u \cdot A / 1000$
4. Extract the unique values of $q$ to define the support of the sample.
5. Apply the piecewise measurement error model to each unique $q$ value, returning a relative error fraction $\varepsilon(q)$.
6. Compute the measurement error bandwidth floor in log space for each unique value: $b_{\text{floor}} = \log(1 + \varepsilon)$. This is the minimum bandwidth needed to spread a kernel across the plausible measurement uncertainty range.
7. If fewer than two unique values exist, perturb the data with uniform noise scaled to $\pm \varepsilon$ of the unique value, then recompute unique values.
8. Compute log spacing between adjacent unique values using a Voronoi decomposition in log space:
   - Find midpoints between consecutive log-transformed values.
   - Extend the boundary midpoints outward by mirroring (so every point has a finite cell).
   - The half-width of each cell is $\Delta_i = (\text{right midpoint} - \text{left midpoint}) / 2$.
9. Set the per-sample bandwidth as the maximum of the Voronoi half-width and the error floor: $h_i = \max(\Delta_i,\, b_{\text{floor},i})$
10. Assign each observation its bandwidth by looking up its index in the sorted unique-value array.

**Result:** a vector of per-sample bandwidths $h_i$, one per observation, used as the standard deviation of a Gaussian kernel centred on $\log(u_i)$ during density evaluation.  To compute the probability mass function:

11. Determine the number of bins that provides approximately the 5% nominal error rate for daily streamflow estimates.  For the sample evaluated, 256 bins over $[\text{UAR}_\text{global}^\text{min}, \text{UAR}_\text{global}^\text{max}]$ yields roughly 5%, which conveniently is close to 8 bits, the universal standard for encoding in most computer architectures.
12. Log-transform all observations: $\ell_i = \log(u_i)$
13. Evaluate the kernel sum on the log-space grid of $M$ points to get the PDF:

$$\hat{f}(\ell) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{h_i \sqrt{2\pi}} \exp\!\left(\frac{-(\ell - \ell_i)^2}{2 h_i^2}\right)$$

In code, this is done by:

```python
def kde_kernel(log_data, bw_array, log_grid):
    H = bw_values[:, None]  # (N, 1)
    U = (log_grid[None, :] - log_data) / H  # (N, M)
    K = np.exp(-0.5 * U**2) / (H * np.sqrt(2 * np.pi))
    return K.sum(axis=0) / log_data.shape[0]
```

13. Normalize $\hat{f}$ so it integrates to 1 over the log-space grid (if density is a desired output).
14. Multiply by the log-space bin widths $w_j = \Delta \log(u_j)$ to convert the PDF to a PMF: $p_j = \hat{f}(\ell_j) \cdot w_j$
15. Normalize the PMF so it sums to 1.

**Result:** $p_j$ is the probability mass assigned to each log-space bin, and $\hat{f}$ is the corresponding density in log space.

The rule of thumb approximation is computed using the [KDEpy python package](https://kdepy.readthedocs.io/en/latest/) which uses a fast fourier transform to compute the density.  Data points are binned onto a regular grid, the kernel is evaluated once on that grid, and the two are convolved via FFT which uses the convolution theorem to run in $O(n \log n)$ rather than $O(N⋅M)$.

With these two estimators evaluated on the same data over a large sample of records, we then want to understand how different the estimates can be.  For this, we compute the Kolmogorov-Smirnov (KS) statistic between each pair of estimators.  The KS statistic measures the maximum distance between the two CDFs at any point in their support.  

The main data folder is found in: [Caravan-csv](../../../../../Documents/common_data/Caravan-csv), where there is an `attributes` folder with catchment attributes for each station, grouped by their geographic region.  The `timeseries/csv` folder has the same region set, and these folders contain csv timeseries files for each station.

## Results



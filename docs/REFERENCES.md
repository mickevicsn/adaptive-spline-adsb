# Methodology References

These references shaped the reconstruction methodology.  They explain why the project uses paired position/velocity observations, adaptive smoothing, robust ADS-B preprocessing, local trajectory modelling, and Kalman/RTS baselines.

## V-Spline

Z. Cao, D. B. Dunson, and M. P. Wand, **"V-Spline: An Adaptive Smoothing Spline for Trajectory Reconstruction"**, Sensors 2021, 21(9), 3215.

- URL: https://www.mdpi.com/1424-8220/21/9/3215
- Why it matters: introduces the V-Spline idea of combining position residuals, velocity residuals, and an acceleration penalty, including an adaptive penalty for irregular sampling and noisy velocity.

## ADS-B preprocessing and data quality

X. Olive, J. Krummen, B. Figuet, and R. Alligier, **"Filtering Techniques for ADS-B Trajectory Preprocessing"**, Journal of Open Aviation Science.

- URL: https://journals.open.tudelft.nl/joas/article/view/7882
- Why it matters: motivates careful filtering because crowdsourced ADS-B and Mode S data can be noisy, uncertain, quantized, and incomplete.

J. Sun, J. Ellerbroek, and J. M. Hoekstra, **"Reconstructing Aircraft Turn Manoeuvres for Trajectory Analyses Using ADS-B Data"**, SESAR Innovation Days 2019.

- URL: https://www.sesarju.eu/sites/default/files/documents/sid/2019/papers/SIDs_2019_paper_57.pdf
- Why it matters: shows why ADS-B is valuable for trajectory studies but limited for microscopic aircraft-dynamics analysis; supports careful derivative and turn-metric interpretation.

## ADS-B message structure and asynchronous stream issues

J. Sun, **The 1090 Megahertz Riddle — ADS-B Basics**.

- URL: https://mode-s.org/1090mhz/content/ads-b/1-basics.html
- Why it matters: explains ADS-B message types and separate airborne position and velocity broadcasts.

FAA Advisory Circular AC 20-165, **Airworthiness Approval of Automatic Dependent Surveillance - Broadcast Out Systems**.

- URL: https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC%2020-165.pdf
- Why it matters: discusses ADS-B system timing and latency considerations, which supports treating raw ADS-B as asynchronous measurement evidence.

## B-splines

SciPy documentation, **`scipy.interpolate.BSpline`**.

- URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
- Why it matters: documents the B-spline representation and basis behavior used conceptually by the B-spline backend.

## Kalman filtering and RTS smoothing

Kalman filtering and Rauch--Tung--Striebel smoothing references.

- Overview URL: https://en.wikipedia.org/wiki/Kalman_filter
- Original RTS reference: H. E. Rauch, F. Tung, and C. Striebel, "Maximum likelihood estimates of linear dynamic systems," AIAA Journal, 1965.
- Why it matters: Kalman/RTS provides a standard state-space smoothing baseline for sequential noisy measurements.

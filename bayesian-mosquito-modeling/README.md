# Bayesian modeling of tiger mosquito reports

A hierarchical Poisson model of *Aedes albopictus* occurrence reports in metropolitan France, with lognormal annual rates and a custom Hamiltonian Monte Carlo sampler. The workflow covers GBIF data extraction, aggregation, posterior estimation, and predictive checks.

**Read:** [report (French)](raport/rapport.pdf) · [model and HMC implementation](scripts/modele.py) · [predictive results](data/prediction_resume.csv)

## Data and findings

The saved dataset contains **2,200 GBIF records**, aggregated into **84 region-year counts across 13 years**. The sampler uses analytical gradients, leapfrog integration, and a Metropolis acceptance step. Twelve configurations were explored; the retained run contains 20,000 iterations, with 4,000 discarded as burn-in and approximately 82.6% acceptance.

Posterior predictive 95% intervals covered **46/84 observations (54.8%)**, while covering all 13 annual totals. The annual model captures aggregate behavior better than individual counts. It has no separate regional effects; occurrence reports also reflect observation effort, not just mosquito abundance. These checks use the fitting data, not held-out forecasts. R-hat and effective sample size were not computed.

## Run

The original scripts, PDF report, data, and figures are included. From this directory:

```bash
python -m pip install -r requirements.txt
python scripts/02_Transformation.py
python scripts/03_Description.py
python scripts/05_estimation.py
python scripts/06_diagnostique.py
python scripts/07_prediction.py
```

These commands regenerate saved tables and figures. The HMC step includes parameter tuning and may take time. To inspect the existing run, start with the report and `figures/`.

Optional fresh extraction requires R packages `rgbif` and `dplyr`, then `Rscript scripts/01_Extraction.R`. GBIF changes over time, so a new extraction can differ from the included snapshot and overwrites it.

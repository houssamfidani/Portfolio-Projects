from pathlib import Path
import os
from math import lgamma

project_dir = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(project_dir / ".matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


input_csv = project_dir / "data" / "echantillons_hmc.csv"
input_resume_csv = project_dir / "data" / "table_resumee.csv"
output_obs_csv = project_dir / "data" / "prediction_observations.csv"
output_year_csv = project_dir / "data" / "prediction_annees.csv"
output_summary_csv = project_dir / "data" / "prediction_resume.csv"
figures_dir = project_dir / "figures"

(project_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)


def logsumexp(x, axis=0):
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(y, axis=axis)


def log_poisson_pmf(y, lam):
    return y * np.log(lam) - lam - lgamma(y + 1)


df = pd.read_csv(input_csv)
resume = pd.read_csv(input_resume_csv)

resume["year"] = pd.to_numeric(resume["year"], errors="coerce").astype(int)
resume["n_occurrences"] = pd.to_numeric(resume["n_occurrences"], errors="coerce")

N = len(df)
burn_in = N // 5
post = df.iloc[burn_in:N].copy()

lambda_cols = [col for col in post.columns if col.startswith("log_lambda_")]
years = [int(col.split("_")[-1]) for col in lambda_cols]
year_to_col = {year: col for year, col in zip(years, lambda_cols)}

obs_years = resume["year"].to_numpy(dtype=int)
y_obs = resume["n_occurrences"].to_numpy(dtype=float)
n_obs = len(resume)

lambda_mat = np.column_stack(
    [np.exp(post[year_to_col[year]].to_numpy(dtype=float)) for year in obs_years]
)
y_rep = np.random.poisson(lambda_mat)

pred_mean = np.mean(y_rep, axis=0)
pred_median = np.median(y_rep, axis=0)
pred_q025 = np.quantile(y_rep, 0.025, axis=0)
pred_q975 = np.quantile(y_rep, 0.975, axis=0)
inside = ((y_obs >= pred_q025) & (y_obs <= pred_q975)).astype(int)
residus = y_obs - pred_mean

log_pmf = np.column_stack(
    [log_poisson_pmf(int(y_obs[j]), lambda_mat[:, j]) for j in range(n_obs)]
)
log_pred_density = logsumexp(log_pmf, axis=0) - np.log(log_pmf.shape[0])

pred_obs = resume.copy()
pred_obs["pred_mean"] = pred_mean
pred_obs["pred_median"] = pred_median
pred_obs["pred_q025"] = pred_q025
pred_obs["pred_q975"] = pred_q975
pred_obs["inside_95"] = inside
pred_obs["residu"] = residus
pred_obs["log_pred_density"] = log_pred_density
pred_obs.to_csv(output_obs_csv, index=False)

group_index = {year: np.where(obs_years == year)[0] for year in sorted(set(obs_years))}
year_rows = []

for year in sorted(group_index):
    ind = group_index[year]
    y_year = y_rep[:, ind].sum(axis=1)
    obs_total = float(np.sum(y_obs[ind]))
    year_rows.append(
        {
            "year": year,
            "n_sites": len(ind),
            "obs_total": obs_total,
            "pred_total_mean": np.mean(y_year),
            "pred_total_median": np.median(y_year),
            "pred_total_q025": np.quantile(y_year, 0.025),
            "pred_total_q975": np.quantile(y_year, 0.975),
            "inside_total_95": int(
                (obs_total >= np.quantile(y_year, 0.025))
                and (obs_total <= np.quantile(y_year, 0.975))
            ),
        }
    )

pred_year = pd.DataFrame(year_rows)
pred_year.to_csv(output_year_csv, index=False)

rmse = np.sqrt(np.mean((y_obs - pred_mean) ** 2))
mae = np.mean(np.abs(y_obs - pred_mean))
coverage = np.mean(inside)
mean_lpd = np.mean(log_pred_density)
coverage_year = np.mean(pred_year["inside_total_95"].to_numpy(dtype=float))

summary = pd.DataFrame(
    [
        {
            "N": N,
            "burn_in": burn_in,
            "n_obs": n_obs,
            "rmse": rmse,
            "mae": mae,
            "coverage_95_obs": coverage,
            "mean_log_pred_density": mean_lpd,
            "coverage_95_year": coverage_year,
        }
    ]
)
summary.to_csv(output_summary_csv, index=False)


fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(
    pred_year["year"],
    pred_year["pred_total_mean"],
    yerr=[
        pred_year["pred_total_mean"] - pred_year["pred_total_q025"],
        pred_year["pred_total_q975"] - pred_year["pred_total_mean"],
    ],
    fmt="o",
    label="Prediction posterieure",
)
ax.plot(pred_year["year"], pred_year["obs_total"], color="red", label="Observe")
ax.set_xlabel("Annee")
ax.set_ylabel("Total annuel")
ax.set_title("Prediction des totaux annuels")
ax.legend()
plt.tight_layout()
plt.savefig(figures_dir / "hmc_prediction_annees.png", dpi=300)
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residus, bins=20, density=True)
ax.axvline(np.mean(residus), color="red")
ax.set_xlabel("Observe - prediction")
ax.set_ylabel("Densite")
ax.set_title("Residus predictifs")
plt.tight_layout()
plt.savefig(figures_dir / "hmc_prediction_residus.png", dpi=300)
plt.close()


print("Taille totale de la chaine :", N)
print("Burn-in :", burn_in)
print("Nombre d'observations :", n_obs)
print("\nResume predictif :")
print(summary.to_string(index=False))
print("\nExtrait prediction par observation :")
print(
    pred_obs[
        [
            "year",
            "stateProvince",
            "n_occurrences",
            "pred_mean",
            "pred_q025",
            "pred_q975",
            "inside_95",
        ]
    ].head(12).to_string(index=False)
)
print("\nResume prediction par annee :")
print(pred_year.to_string(index=False))
print("\nFichiers crees :")
print("- prediction_observations.csv")
print("- prediction_annees.csv")
print("- prediction_resume.csv")
print("- hmc_prediction_annees.png")
print("- hmc_prediction_residus.png")

from pathlib import Path
import os

project_dir = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(project_dir / ".matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


input_csv = project_dir / "data" / "echantillons_hmc.csv"
input_grid_csv = project_dir / "data" / "grille_hmc.csv"
input_resume_csv = project_dir / "data" / "table_resumee.csv"
output_csv = project_dir / "data" / "diagnostic_hmc.csv"
output_global_csv = project_dir / "data" / "resume_posterior_global.csv"
output_year_csv = project_dir / "data" / "resume_posterior_annees.csv"
figures_dir = project_dir / "figures"

(project_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)


def acf(x, lag_max):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    den = np.sum(x ** 2)

    if den == 0:
        return np.zeros(lag_max + 1)

    vals = np.zeros(lag_max + 1)
    vals[0] = 1.0

    for lag in range(1, lag_max + 1):
        vals[lag] = np.sum(x[:-lag] * x[lag:]) / den

    return vals


def resume_variable(x):
    x = np.asarray(x, dtype=float)
    return {
        "moyenne": np.mean(x),
        "mediane": np.median(x),
        "q025": np.quantile(x, 0.025),
        "q975": np.quantile(x, 0.975),
        "ecart_type": np.std(x),
    }


df = pd.read_csv(input_csv)
df_resume = pd.read_csv(input_resume_csv)
df_resume["year"] = pd.to_numeric(df_resume["year"], errors="coerce").astype(int)
df_resume["n_occurrences"] = pd.to_numeric(df_resume["n_occurrences"], errors="coerce")
N = len(df)
burn_in = N // 5

grid = pd.read_csv(input_grid_csv)
eps = grid.loc[0, "eps"]
L = int(grid.loc[0, "L"])

cols = ["log_lambda_2010", "log_lambda_2020", "log_lambda_2025", "mu", "tau2"]
cols = [col for col in cols if col in df.columns]

post = df.iloc[burn_in:N].copy()
tx_accept = np.mean(df["accept"].to_numpy()[1:N])

resume = []

for col in cols:
    x = post[col].to_numpy(dtype=float)
    acfx = acf(x, 10)
    resume.append(
        {
            "parametre": col,
            "moyenne_post_burnin": np.mean(x),
            "ecart_type_post_burnin": np.std(x),
            "acf_1": acfx[1],
            "acf_5": acfx[5],
            "acf_10": acfx[10],
        }
    )

resume = pd.DataFrame(resume)
resume["burn_in"] = burn_in
resume["N"] = N
resume["eps"] = eps
resume["L"] = L
resume["tx_accept"] = tx_accept
resume.to_csv(output_csv, index=False)


global_rows = []

for col in ["mu", "log_tau2", "tau2"]:
    if col in post.columns:
        vals = post[col].to_numpy(dtype=float)
        out = resume_variable(vals)
        out["parametre"] = col
        global_rows.append(out)

tau = np.sqrt(post["tau2"].to_numpy(dtype=float))
out = resume_variable(tau)
out["parametre"] = "tau"
global_rows.append(out)

global_rows = pd.DataFrame(global_rows)
global_rows["burn_in"] = burn_in
global_rows["N"] = N
global_rows.to_csv(output_global_csv, index=False)


lambda_cols = [col for col in df.columns if col.startswith("log_lambda_")]
years = [int(col.split("_")[-1]) for col in lambda_cols]

obs = (
    df_resume.groupby("year")["n_occurrences"]
    .agg(["sum", "size"])
    .reset_index()
    .rename(columns={"sum": "occurrences_totales", "size": "n_sites"})
)

resume_year = []

for year, col in zip(years, lambda_cols):
    log_vals = post[col].to_numpy(dtype=float)
    lambda_vals = np.exp(log_vals)
    out_log = resume_variable(log_vals)
    out_lambda = resume_variable(lambda_vals)
    row = {
        "year": year,
        "log_lambda_moyenne": out_log["moyenne"],
        "log_lambda_mediane": out_log["mediane"],
        "log_lambda_q025": out_log["q025"],
        "log_lambda_q975": out_log["q975"],
        "lambda_moyenne": out_lambda["moyenne"],
        "lambda_mediane": out_lambda["mediane"],
        "lambda_q025": out_lambda["q025"],
        "lambda_q975": out_lambda["q975"],
        "variance_intra_moyenne": out_lambda["moyenne"],
        "variance_intra_mediane": out_lambda["mediane"],
    }
    resume_year.append(row)

resume_year = pd.DataFrame(resume_year)
resume_year = resume_year.merge(obs, on="year", how="left")
resume_year["moyenne_observee_par_site"] = (
    resume_year["occurrences_totales"] / resume_year["n_sites"]
)
resume_year.to_csv(output_year_csv, index=False)


fig, ax = plt.subplots(nrows=len(cols), ncols=2, figsize=(12, 2.8 * len(cols)))

if len(cols) == 1:
    ax = np.array([ax])

for i, col in enumerate(cols):
    x = df[col].to_numpy(dtype=float)
    ax[i, 0].plot(x)
    ax[i, 0].axvline(burn_in, color="darkgrey")
    ax[i, 0].set_title(col)
    ax[i, 0].set_xlabel("Iteration")
    ax[i, 0].set_ylabel("Valeur")

    ax[i, 1].hist(post[col].to_numpy(dtype=float), bins=30, density=True)
    ax[i, 1].set_title(col + " post burn-in")
    ax[i, 1].set_xlabel("Valeur")
    ax[i, 1].set_ylabel("Densite")

plt.tight_layout()
plt.savefig(figures_dir / "hmc_chaines.png", dpi=300)
plt.close()


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11, 8))

ax[0].errorbar(
    resume_year["year"],
    resume_year["log_lambda_moyenne"],
    yerr=[
        resume_year["log_lambda_moyenne"] - resume_year["log_lambda_q025"],
        resume_year["log_lambda_q975"] - resume_year["log_lambda_moyenne"],
    ],
    fmt="o",
)
ax[0].set_title("log lambda par annee")
ax[0].set_xlabel("Annee")
ax[0].set_ylabel("log lambda")

ax[1].errorbar(
    resume_year["year"],
    resume_year["lambda_moyenne"],
    yerr=[
        resume_year["lambda_moyenne"] - resume_year["lambda_q025"],
        resume_year["lambda_q975"] - resume_year["lambda_moyenne"],
    ],
    fmt="o",
    label="Posterior mean et IC",
)
ax[1].plot(
    resume_year["year"],
    resume_year["moyenne_observee_par_site"],
    color="red",
    label="Moyenne observee par site",
)
ax[1].set_title("lambda par annee")
ax[1].set_xlabel("Annee")
ax[1].set_ylabel("lambda")
ax[1].legend()

plt.tight_layout()
plt.savefig(figures_dir / "hmc_variabilite_annees.png", dpi=300)
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))
vals = post["tau2"].to_numpy(dtype=float)
ax.hist(vals, bins=30, density=True)
ax.axvline(np.mean(vals), color="red")
ax.axvline(np.quantile(vals, 0.025), color="darkgrey")
ax.axvline(np.quantile(vals, 0.975), color="darkgrey")
ax.set_title("Posterior de tau2")
ax.set_xlabel("tau2")
ax.set_ylabel("Densite")
plt.tight_layout()
plt.savefig(figures_dir / "hmc_tau2.png", dpi=300)
plt.close()


print("Taille totale de la chaine :", N)
print("Burn-in :", burn_in)
print("Pas d'integration :", eps)
print("Nombre de pas leapfrog :", L)
print("Taux d'acceptation :", tx_accept)
print("\nResume numerique :")
print(resume.to_string(index=False))
print("\nResume global :")
print(global_rows.to_string(index=False))
print("\nResume par annee :")
print(
    resume_year[
        [
            "year",
            "lambda_moyenne",
            "lambda_q025",
            "lambda_q975",
            "moyenne_observee_par_site",
        ]
    ].to_string(index=False)
)
print("\nFichiers crees :")
print("- diagnostic_hmc.csv")
print("- resume_posterior_global.csv")
print("- resume_posterior_annees.csv")
print("- hmc_chaines.png")
print("- hmc_variabilite_annees.png")
print("- hmc_tau2.png")

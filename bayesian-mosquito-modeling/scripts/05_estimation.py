from pathlib import Path

import numpy as np
import pandas as pd

from modele import algorithme_hmc

project_dir = Path(__file__).resolve().parent.parent
input_csv = project_dir / "data" / "table_resumee.csv"
output_csv = project_dir / "data" / "echantillons_hmc.csv"
output_grid_csv = project_dir / "data" / "grille_hmc.csv"

df = pd.read_csv(input_csv)
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df["n_occurrences"] = pd.to_numeric(df["n_occurrences"], errors="coerce")

grouped = df.groupby("year")["n_occurrences"]
annees = grouped.sum().index.to_numpy()
sommes = grouped.sum().to_numpy(dtype=float)
tailles = grouped.size().to_numpy(dtype=float)

I = len(annees)
d = I + 2


N = 20000
N_grid = 3000
eps_vals = [0.03, 0.05, 0.07, 0.10]
L_vals = [5, 10, 15]
masse = np.ones(d)

zinit = np.log(sommes / tailles)
muinit = np.mean(zinit)
log_tau2init = np.log(np.var(zinit) + 0.1)
xinit = np.concatenate((zinit, np.array([muinit, log_tau2init])))

grille = []

for eps in eps_vals:
    for L in L_vals:
        chaine_grid, accept_grid = algorithme_hmc(N_grid, xinit, eps, L, sommes, tailles, masse)
        tx_accept = np.mean(accept_grid[1:N_grid])
        grille.append(
            {
                "eps": eps,
                "L": L,
                "tx_accept": tx_accept,
                "dernier_mu": chaine_grid[-1, I],
                "dernier_tau2": np.exp(chaine_grid[-1, I + 1]),
            }
        )

grille = pd.DataFrame(grille)
grille["ecart_cible"] = np.abs(grille["tx_accept"] - 0.80)
grille = grille.sort_values(["ecart_cible", "eps", "L"]).reset_index(drop=True)
grille.to_csv(output_grid_csv, index=False)

eps = grille.loc[0, "eps"]
L = int(grille.loc[0, "L"])

chaine, accept = algorithme_hmc(N, xinit, eps, L, sommes, tailles, masse)


echantillons = pd.DataFrame(
    chaine[:, :I],
    columns=[f"log_lambda_{annee}" for annee in annees],
)

echantillons["mu"] = chaine[:, I]
echantillons["log_tau2"] = chaine[:, I + 1]
echantillons["tau2"] = np.exp(chaine[:, I + 1])
echantillons["accept"] = accept

echantillons.to_csv(output_csv, index=False)

tx_accept = np.mean(accept[1:N])

print("Nombre d'annees :", I)
print("Dimension du probleme :", d)
print("Grille testee :")
print(grille[["eps", "L", "tx_accept"]].to_string(index=False))
print("Pas d'integration :", eps)
print("Nombre de pas leapfrog :", L)
print("Taux d'acceptation :", tx_accept)
print("Fichier grille cree :", output_grid_csv)
print("Fichier cree :", output_csv)

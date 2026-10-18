from pathlib import Path
import os

project_dir = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(project_dir / ".matplotlib")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

input_csv = project_dir / "data" / "aedes_albopictus_france_metropolitaine.csv"
figures_dir = project_dir / "figures"

(project_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv, low_memory=False)

df = df[df["decimalLatitude"].notna() & df["decimalLongitude"].notna()]
df["year"] = pd.to_numeric(df["year"], errors="coerce")

print("Nombre total d'observations :", len(df))
print("Nombre d'années distinctes :", df["year"].nunique())
print("Années présentes :", sorted(df["year"].dropna().unique()))
print("Latitude min / max :", df["decimalLatitude"].min(), df["decimalLatitude"].max())
print("Longitude min / max :", df["decimalLongitude"].min(), df["decimalLongitude"].max())
print("\nTop 10 des régions :")
print(df["stateProvince"].value_counts(dropna=False).head(10))

occ_par_an = (
    df.dropna(subset=["year"])
    .groupby("year")
    .size()
    .reset_index(name="n_occurrences")
)

plt.figure(figsize=(10, 5))
plt.bar(occ_par_an["year"].astype(int).astype(str), occ_par_an["n_occurrences"])
plt.xticks(rotation=45)
plt.xlabel("Année")
plt.ylabel("Occurrences")
plt.title("Occurrences par année")
plt.tight_layout()
plt.savefig(figures_dir / "occurrences_par_annee.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 8))
plt.scatter(df["decimalLongitude"], df["decimalLatitude"], s=10, alpha=0.6)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Carte des occurrences")
plt.tight_layout()
plt.savefig(figures_dir / "carte_occurrences.png", dpi=300)
plt.close()

print("\nFichiers créés :")
print("- occurrences_par_annee.png")
print("- carte_occurrences.png")

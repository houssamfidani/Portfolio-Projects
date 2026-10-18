from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
input_csv = project_dir / "data" / "aedes_albopictus_france_metropolitaine.csv"
output_csv = project_dir / "data" / "table_resumee.csv"

df = pd.read_csv(input_csv, low_memory=False)

df = df[df["stateProvince"].notna() & (df["stateProvince"].astype(str).str.strip() != "")]

resume = (
    df.groupby(["year", "stateProvince"])
    .size()
    .reset_index(name="n_occurrences")
)

print("Nombre d'observations résumées :", len(resume))
print(resume.head())

resume.to_csv(output_csv, index=False)

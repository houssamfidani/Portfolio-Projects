library(rgbif)
library(dplyr)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- dirname(normalizePath(script_path))
output_csv <- file.path(script_dir, "..", "data", "aedes_albopictus_france_metropolitaine.csv")

species_name <- "Aedes albopictus"

geom <- "POLYGON((-5.5 41.0, 9.8 41.0, 9.8 51.5, -5.5 51.5, -5.5 41.0))"

taxon_key <- name_backbone(name = species_name)$usageKey
cat("taxon_key =", taxon_key, "\n")

limit <- 300
start <- 0
parts <- list()

repeat {
  res <- occ_search(
    taxonKey = taxon_key,
    country = "FR",
    geometry = geom,
    occurrenceStatus = "PRESENT",
    limit = limit,
    start = start
  )

  dat <- res$data

  if (is.null(dat) || nrow(dat) == 0) {
    break
  }

  parts[[length(parts) + 1]] <- dat
  start <- start + nrow(dat)

  cat("Lignes récupérées :", start, "/", res$meta$count, "\n")

  if (start >= res$meta$count) {
    break
  }
}

df <- bind_rows(parts)

write.csv(df, output_csv, row.names = FALSE)

cat("Total final :", nrow(df), "\n")

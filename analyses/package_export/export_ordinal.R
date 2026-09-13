# Export the ordinal regression as a small RDS the R package can ship: the twelve feature
# names, the z-scaling used at fit time, and the proportional-odds coefficients and
# cut-points. Fit is the train+val one from feature_classifiers.qmd section 6.1-6.2.
# Checked against the predictions that analysis exported for both held-out splits.

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
source(file.path(here, "features.R"))
root <- normalizePath(file.path(here, "..", ".."))
uv <- "/Users/jowillia/Documents/GitHub/uv-agroinfiltration-dataset"

all_features <- c(
  "ent_gap", "green_gap", "cda_green", "sep", "cda_green_sd", "cda_bright",
  "cda_bright_sd", "leaf_green_sd", "mean_green", "log_gr", "leaf_bright", "entropy_raw"
)

feat_all <- read.csv(file.path(uv, "data", "features", "cda_image_features.csv"))
split <- read.csv(file.path(root, "data", "datasets", "medoid_64", "medoid_64.csv"))
split$image_id <- sub("\\.tif$", "", split$filename)
d <- merge(split, feat_all, by = "image_id")
stopifnot(nrow(d) == nrow(split), all(d$label == d$score))

tv <- d$split %in% c("train", "val")
Xtv <- as.matrix(d[tv, all_features])
ytv <- d$score[tv]

mu <- colMeans(Xtv)
s <- apply(Xtv, 2, sd)
s[s == 0] <- 1
dtr <- data.frame(scale(Xtv, mu, s), y = factor(ytv, levels = 0:6, ordered = TRUE))
fit <- suppressWarnings(MASS::polr(y ~ ., data = dtr, Hess = TRUE))

ordinal <- list(
  features = all_features, mu = mu, s = s,
  beta = coef(fit), zeta = fit$zeta
)
saveRDS(ordinal, file.path(here, "ordinal_regression.rds"))

want <- read.csv(file.path(root, "analyses", "03_feature_classifiers", "model_cache", "nondl_heldout_predictions.csv"))
got <- d[match(want$image_id, d$image_id), ]
stopifnot(all(got$score == want$true))
p <- ordinal_probs(ordinal, as.matrix(got[, all_features]))
cls <- max.col(p, ties.method = "first") - 1L
cat(sprintf(
  "%d crops | class mismatches %d | max prob diff %.2e\n",
  nrow(want), sum(cls != want$ordinal_regression),
  max(abs(p - as.matrix(want[, paste0("p", 0:6)])))
))
for (sp in unique(want$split)) {
  i <- want$split == sp
  cat(sprintf(
    "  %-8s exact %.4f near %.4f\n", sp, mean(cls[i] == want$true[i]),
    mean(abs(cls[i] - want$true[i]) <= 1)
  ))
}

# Two PCA objects for AutoCDAScorer's existing diagnostic_pca() plots, in the same format
# as the base_cnn_pca.rds it already ships: five sklearn components over the 12,288 raw
# pixels of a 64x64 crop, fitted on the crops each model was trained on.
#   medoid_64      train+val, serving the ordinal regression
#   medoid_64_aug  the geometric-augmentation training subset, serving the CNNs
# Images are RGB on a 0-1 scale, matching what the package's loaders return. The training
# arrays are BGR/256, so the channels are reversed here and the scale left alone.

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
root <- normalizePath(file.path(here, "..", ".."))
reticulate::use_python("~/miniforge3/envs/agroinfiltration_scoring/bin/python", required = TRUE)

py <- reticulate::py_run_string(sprintf('
import sys, numpy as np
sys.path.insert(0, "%s/src")
import base_dataset

d = base_dataset.load_dataset("%s/data/datasets/medoid_64/medoid_64.npy")
medoid = np.concatenate([np.asarray(d["train_images"]), np.asarray(d["val_images"])])

z = np.load("%s/data/datasets/medoid_64_aug/medoid_64_aug.npz")
y, is_aug = z["labels"].astype(int), z["is_aug"].astype(bool)
per_class = int(np.bincount(y[~is_aug], minlength=7).max())
rng = np.random.default_rng(7)          # run_search.py subset(), --subset_seed default
keep = []
for c in range(7):
    real, aug = np.where((y == c) & ~is_aug)[0], np.where((y == c) & is_aug)[0]
    keep.append(rng.permutation(real)[:per_class] if len(real) >= per_class
                else np.concatenate([real, rng.permutation(aug)[:per_class - len(real)]]))
geom = z["images"][np.sort(np.concatenate(keep))].astype("float64") / 256.0
', root, root, root))

fit_pca <- function(x, n = 5L) {
  sk <- reticulate::import("sklearn.decomposition")
  x <- x[, , , 3:1] # channels reversed from BGR into RGB
  flat <- matrix(aperm(x, c(1, 4, 3, 2)), nrow = dim(x)[1])
  p <- sk$PCA(n_components = n)$fit(flat)
  list(
    features_pca = p$transform(flat),
    explained_variance = p$explained_variance_ratio_,
    principal_components = p$components_,
    center = p$mean_
  )
}

for (nm in c("medoid", "geom")) {
  x <- py[[nm]]
  p <- fit_pca(x)
  out <- file.path(here, paste0(nm, "_pca.rds"))
  saveRDS(p, out)
  cat(sprintf(
    "%s: %d crops, variance explained %s -> %s\n", nm, dim(x)[1],
    paste(sprintf("%.3f", p$explained_variance), collapse = " "), basename(out)
  ))
}

# Where do the cGAN crops land on AutoCDAScorer's own diagnostic? The package ships a PCA
# fitted on each model's training crops and marks a user image out of scope if it falls
# outside the convex hull of that cloud. The cGAN crops are images the models were never
# meant to see, so this asks whether the diagnostic would catch them.

library(ggplot2)
suppressMessages(devtools::load_all("/Users/jowillia/Packages/AutoCDAScorer", quiet = TRUE))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
root <- normalizePath(file.path(here, "..", ".."))
out <- file.path(here, "thesis_plots")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

gan_dir <- file.path(root, "analyses", "07_gan_diversity", "gan_sample")
files <- list.files(gan_dir, pattern = "[.]tif$", recursive = TRUE, full.names = TRUE)
arr <- array(0, dim = c(length(files), 64, 64, 3))
for (i in seq_along(files)) {
  arr[i, , , ] <- as.numeric(tiff::readTIFF(files[i])) |> array(dim = c(64, 64, 3))
}
data <- list(images = arr)

pca <- AutoCDAScorer:::load_result_pca("ensemble")
feats <- AutoCDAScorer:::pca_transform(data, pca)

# Ray-casting point-in-polygon, so the check needs no extra package.
inside <- function(pc_x, pc_y) {
  tr <- pca$features_pca[, c(pc_x, pc_y)]
  h <- chull(tr)
  px <- tr[h, 1]
  py <- tr[h, 2]
  n <- length(h)
  vapply(seq_len(nrow(feats)), function(i) {
    x <- feats[i, pc_x]
    y <- feats[i, pc_y]
    hit <- FALSE
    j <- n
    for (k in seq_len(n)) {
      crosses <- (py[k] > y) != (py[j] > y)
      if (crosses && x < (px[j] - px[k]) * (y - py[k]) / (py[j] - py[k]) + px[k]) {
        hit <- !hit
      }
      j <- k
    }
    hit
  }, logical(1))
}
for (pr in list(c(1, 2), c(1, 3), c(2, 3))) {
  cat(sprintf(
    "PC%d-PC%d: %.1f%% of %d cGAN crops inside the training hull\n",
    pr[1], pr[2], 100 * mean(inside(pr[1], pr[2])), length(files)
  ))
}

row <- cowplot::plot_grid(
  AutoCDAScorer:::pca_diagnostic_convexhull(pca, feats, 1, 2) + labs(x = "PC 1", y = "PC 2"),
  AutoCDAScorer:::pca_diagnostic_convexhull(pca, feats, 1, 3) + labs(x = "PC 1", y = "PC 3"),
  AutoCDAScorer:::pca_diagnostic_convexhull(pca, feats, 2, 3) + labs(x = "PC 2", y = "PC 3"),
  nrow = 1
)
ggsave(file.path(out, "gan_scope_check.png"), row, width = 9, height = 3.2, dpi = 300, bg = "white")
cat("written:", file.path(out, "gan_scope_check.png"), "\n")

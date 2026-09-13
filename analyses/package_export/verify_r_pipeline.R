# End-to-end check that the R side reproduces the chapter's test-split numbers, using only
# what the package will ship: the 17 unmodified geometric-augmentation .keras files, the
# ordinal regression RDS, and crops read from disk in R. Targets are 66.88% for model 6848
# (chapter paragraph 12) and 69.16% for the ensemble (paragraph 15).

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
root <- normalizePath(file.path(here, "..", ".."))
source(file.path(here, "features.R"))
Sys.setenv(RETICULATE_PYTHON = "~/miniforge3/envs/agroinfiltration_scoring/bin/python")

# cv2.resize INTER_LINEAR: sample at (j + 0.5) * scale - 0.5 and interpolate the four
# neighbours, clamping at the edges. magick's resize uses a different filter and shifts
# 14 of the 308 test predictions, so the package needs this rule rather than that one.
resize_bilinear <- function(img, n = 64L) {
  ax <- function(s) {
    x <- (seq_len(n) - 0.5) * s / n - 0.5
    i0 <- floor(x)
    a <- x - i0
    a[i0 < 0 | i0 + 1 > s - 1] <- 0
    list(i0 = pmin(pmax(i0, 0), s - 1) + 1L, i1 = pmin(pmax(i0 + 1, 0), s - 1) + 1L, a = a)
  }
  h <- ax(dim(img)[1])
  w <- ax(dim(img)[2])
  lerp <- function(A, B, a, m) sweep(A, m, 1 - a, "*") + sweep(B, m, a, "*")
  lerp(lerp(img[h$i0, w$i0, , drop = FALSE], img[h$i0, w$i1, , drop = FALSE], w$a, 2),
       lerp(img[h$i1, w$i0, , drop = FALSE], img[h$i1, w$i1, , drop = FALSE], w$a, 2),
       h$a, 1)
}

split <- read.csv(file.path(root, "data", "datasets", "medoid_64", "medoid_64.csv"))
te <- split[split$split == "test", ]
paths <- file.path(root, "data", "cropped_images_medoid", te$label, te$filename)
stopifnot(all(file.exists(paths)))

# One pass per crop: the native-resolution image feeds the features, the 64x64 BGR/256
# version feeds the networks. cv.imread returns uint8, so the resize is rounded first.
X <- array(0, dim = c(nrow(te), 64, 64, 3))
F <- matrix(0, nrow(te), length(FEATURE_NAMES), dimnames = list(NULL, FEATURE_NAMES))
for (i in seq_along(paths)) {
  img <- load_img(paths[i])
  F[i, ] <- cda_features(img)
  X[i, , , ] <- round(resize_bilinear(img * 255))[, , 3:1] / 256
}

ordinal <- readRDS(file.path(here, "ordinal_regression.rds"))
P_ord <- ordinal_probs(ordinal, F)

suppressMessages(library(keras3))
dirs <- sort(list.dirs(file.path(root, "models/07_Medoid_CNN_GeomAug/01_MatchLargestClass",
                                 "results"), recursive = FALSE))
P_cnn <- array(0, dim = c(length(dirs), nrow(te), 7))
for (i in seq_along(dirs)) {
  m <- keras3::load_model(file.path(dirs[i], "model.keras"))
  P_cnn[i, , ] <- predict(m, X, verbose = 0)
}
best <- grep("_6848$", basename(dirs))

y <- te$label
acc <- function(p) mean(max.col(p, ties.method = "first") - 1L == y)
near <- function(p) mean(abs(max.col(p, ties.method = "first") - 1L - y) <= 1)
mean_cnn <- apply(P_cnn, 2:3, mean)

cat(sprintf("%d models, %d test crops\n", length(dirs), nrow(te)))
cat(sprintf("6848            exact %.4f  near %.4f   (target 0.6688 / 0.9903)\n",
            acc(P_cnn[best, , ]), near(P_cnn[best, , ])))
cat(sprintf("ordinal         exact %.4f  near %.4f   (target 0.6331 / 0.9643)\n",
            acc(P_ord), near(P_ord)))
cat(sprintf("geom + ordinal  exact %.4f  near %.4f   (target 0.6916 / 0.9838)\n",
            acc((mean_cnn + P_ord) / 2), near((mean_cnn + P_ord) / 2)))

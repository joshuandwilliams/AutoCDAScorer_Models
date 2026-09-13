# The twelve hand-crafted features the ordinal regression is fitted on, ported verbatim
# from uv-agroinfiltration-dataset/analyses/10_segmentation_and_feature_extraction (all
# but entropy_raw) and 09_image_entropy (entropy_raw). Run as a script to diff the port
# against the published table; source() it to get the functions alone.

load_img <- function(path) {
  img <- tiff::readTIFF(path, info = FALSE)
  if (length(dim(img)) == 2) img <- array(img, dim = c(dim(img), 3))
  img[, , 1:3, drop = FALSE]
}

leaf_mask <- function(img) (img[, , 1] + img[, , 2] + img[, , 3]) / 3 >= 0.15

greenness <- function(img) img[, , 2] - img[, , 1]

# Otsu threshold, plus separability = between-class / total variance at that threshold.
otsu_sep <- function(x, nbins = 64) {
  rng <- range(x)
  if (diff(rng) < 1e-6) {
    return(c(thr = rng[1], sep = 0, frac_hi = 0))
  }
  br <- seq(rng[1], rng[2], length.out = nbins + 1)
  mids <- (br[-1] + br[-length(br)]) / 2
  h <- tabulate(cut(x, br, include.lowest = TRUE), nbins)
  p <- h / sum(h)
  mu <- sum(p * mids)
  tot <- sum(p * (mids - mu)^2)
  w1 <- cumsum(p)
  m1 <- cumsum(p * mids)
  bcv <- (mu * w1 - m1)^2 / (w1 * (1 - w1) + 1e-12)
  c(
    thr = br[which.max(bcv) + 1], sep = max(bcv) / (tot + 1e-12),
    frac_hi = mean(x > br[which.max(bcv) + 1])
  )
}

# Otsu on blurred greenness within the leaf, then median filter at a radius that scales
# with crop size, largest connected component, holes filled.
segment_cda <- function(img, sigma = 2) {
  leaf <- leaf_mask(img)
  g <- greenness(img)
  gs <- (g - min(g)) / (max(g) - min(g) + 1e-9)
  gb <- EBImage::gblur(gs, sigma = sigma)
  mask <- leaf & (gb > EBImage::otsu(matrix(gb[leaf]), range = c(0, 1)))
  b <- max(1, round(5 * mean(dim(img)[1:2]) / 264))
  mask <- EBImage::fillHull(EBImage::medianFilter(mask, 2 * b)) & leaf
  lab <- EBImage::bwlabel(mask)
  if (max(lab) > 1) mask <- lab == which.max(tabulate(lab[lab > 0]))
  list(leaf = leaf, mask = EBImage::fillHull(mask))
}

shannon_entropy_vec <- function(v) {
  q <- as.integer(round(pmin(pmax(v, 0), 1) * 255))
  counts <- tabulate(q + 1L, nbins = 256L)
  p <- counts[counts > 0] / length(q)
  -sum(p * log2(p))
}

region_entropy <- function(img, mask) {
  if (!any(mask)) {
    return(NA_real_)
  }
  shannon_entropy_vec(c(img[, , 1][mask], img[, , 2][mask], img[, , 3][mask]))
}

FEATURE_NAMES <- c(
  "ent_gap", "green_gap", "cda_green", "sep", "cda_green_sd", "cda_bright",
  "cda_bright_sd", "leaf_green_sd", "mean_green", "log_gr", "leaf_bright",
  "entropy_raw", "frac_g_gt_r"
)

# One crop at native resolution in, thirteen numbers out (the twelve model features plus
# frac_g_gt_r, which is a baseline predictor rather than a model input).
cda_features <- function(img) {
  s <- segment_cda(img)
  leaf <- s$leaf
  cda <- s$mask & leaf
  healthy <- leaf & !cda
  gmat <- greenness(img)
  bmat <- (img[, , 1] + img[, , 2] + img[, , 3]) / 3
  R <- img[, , 1]
  G <- img[, , 2]
  nsd <- function(x) if (length(x) > 1) sd(x) else 0
  mn <- function(x) if (length(x)) mean(x) else 0
  gv_leaf <- gmat[leaf]
  both <- any(cda) && any(healthy)

  c(
    ent_gap = if (both) region_entropy(img, cda) - region_entropy(img, healthy) else 0,
    green_gap = if (both) mean(gmat[cda]) - mean(gmat[healthy]) else 0,
    cda_green = mn(gmat[cda]),
    sep = unname(otsu_sep(gv_leaf)["sep"]),
    cda_green_sd = nsd(gmat[cda]) * 255,
    cda_bright = mn(bmat[cda]) * 255,
    cda_bright_sd = nsd(bmat[cda]) * 255,
    leaf_green_sd = nsd(gv_leaf) * 255,
    mean_green = mn(G[leaf]) * 255,
    log_gr = log((sum(G[leaf] >= R[leaf]) + 1) / (sum(R[leaf] > G[leaf]) + 1)),
    leaf_bright = mn(bmat[leaf]) * 255,
    entropy_raw = shannon_entropy_vec(img),
    frac_g_gt_r = sum(cda) / sum(leaf)
  )
}

# Proportional odds: P(Y <= k) = plogis(zeta_k - x'beta), from the coefficients and
# cut-points in ordinal_regression.rds, so predicting needs no MASS internals.
ordinal_probs <- function(m, X) {
  eta <- as.vector(scale(X[, m$features, drop = FALSE], m$mu, m$s) %*% m$beta)
  cum <- cbind(0, vapply(m$zeta, function(z) plogis(z - eta), numeric(length(eta))), 1)
  p <- cum[, -1, drop = FALSE] - cum[, -ncol(cum), drop = FALSE]
  colnames(p) <- paste0("p", 0:6)
  p
}

if (sys.nframe() == 0L) {
  crops <- "/Users/jowillia/Documents/GitHub/uv-agroinfiltration-dataset/data/cropped_images_medoid"
  published <- file.path(dirname(crops), "features", "cda_image_features.csv")
  here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
  cache <- file.path(here, "features_port.rds")

  files <- list.files(crops, pattern = "\\.tif$", recursive = TRUE, full.names = TRUE)
  if (file.exists(cache)) {
    got <- readRDS(cache)
  } else {
    m <- t(vapply(files, function(f) cda_features(load_img(f)), numeric(13)))
    got <- data.frame(
      image_id = tools::file_path_sans_ext(basename(files)), m,
      row.names = NULL
    )
    saveRDS(got, cache)
  }

  want <- read.csv(published)
  stopifnot(setequal(got$image_id, want$image_id))
  want <- want[match(got$image_id, want$image_id), ]
  d <- vapply(FEATURE_NAMES, function(k) max(abs(got[[k]] - want[[k]])), numeric(1))
  cat(sprintf("%d crops\n", nrow(got)))
  print(signif(d, 3))
  cat(if (max(d) < 1e-8) "MATCH\n" else "DIFFERENT\n")
}

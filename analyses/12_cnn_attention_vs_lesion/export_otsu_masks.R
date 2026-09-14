# Export Otsu CDA masks for the held-out test crops, for analyses/06.
#
# The segmentation is a verbatim copy of `segment_cda` from the sibling dataset repo
# (uv-agroinfiltration-dataset, analyses/10_segmentation_and_feature_extraction,
# chunk 1.3). It is duplicated rather than sourced because that file is a Quarto
# document, not a library, and because this script must run against a DIFFERENT crop
# set (see below). Any change to the original should be mirrored here.
#
# Why re-run it at all: that analysis caches only 14 scalar features per image
# (segmentation_cache/cda_segmentation_metrics.rds), not the masks themselves.
#
# Crop set: `cropped_images_medoid`, which is what `medoid_64` is built from and so what
# the CNN being explained actually sees. It is also the crop set the dataset repo
# segments, so this matches it directly. A saliency map on one crop compared against a
# mask from a different crop of the same CDA would be comparing two different images.
#
# Segmentation runs at native crop resolution, as in the original -- the median-filter
# radius scales with image size, so segmenting a pre-shrunk 64x64 image would apply a
# different amount of smoothing. The binary mask is downsampled to 64x64 afterwards to
# match the network input.
#
# Run with the dataset repo's R environment, which has EBImage:
#   ~/miniforge3/envs/uv-agroinfiltration/bin/Rscript analyses/12_cnn_attention_vs_lesion/export_otsu_masks.R

suppressPackageStartupMessages({
  library(tiff)
  library(png)
})

# Run from the repo root (the Makefile-free convention used by the other analyses).
repo <- normalizePath(".")
if (!dir.exists(file.path(repo, "data", "cropped_images_medoid"))) {
  stop("run this from the repo root: data/cropped_images_medoid not found at ", repo)
}

split_csv <- file.path(repo, "data", "datasets", "medoid_64", "medoid_64.csv")
crop_dir <- file.path(repo, "data", "cropped_images_medoid")
out_dir <- file.path(repo, "analyses", "12_cnn_attention_vs_lesion", "mask_cache")
dir.create(file.path(out_dir, "masks"), recursive = TRUE, showWarnings = FALSE)

# ---- segmentation (verbatim from the dataset repo, chunk 1.3) ----------------

load_img <- function(path) {
  img <- readTIFF(path, info = FALSE)
  if (length(dim(img)) == 2) img <- array(img, dim = c(dim(img), 3))
  img[, , 1:3, drop = FALSE]
}

leaf_mask <- function(img) (img[, , 1] + img[, , 2] + img[, , 3]) / 3 >= 0.15

greenness <- function(img) img[, , 2] - img[, , 1]

segment_cda <- function(img, sigma = 2) {
  leaf <- leaf_mask(img)
  g <- greenness(img)
  gs <- (g - min(g)) / (max(g) - min(g) + 1e-9)
  gb <- EBImage::gblur(gs, sigma = sigma)
  thr <- EBImage::otsu(matrix(gb[leaf]), range = c(0, 1))
  mask <- leaf & (gb > thr)
  b <- max(1, round(5 * mean(dim(img)[1:2]) / 264))
  mask <- EBImage::fillHull(EBImage::medianFilter(mask, 2 * b))
  mask <- mask & leaf
  lab <- EBImage::bwlabel(mask)
  if (max(lab) > 1) mask <- lab == which.max(tabulate(lab[lab > 0]))
  mask <- EBImage::fillHull(mask)
  list(leaf = leaf, greenness = gb, mask = mask, cda_frac = mean(mask))
}

# ---- select the test crops --------------------------------------------------

split <- read.csv(split_csv, stringsAsFactors = FALSE)
test <- split[split$split == "test", ]
# Score 0 means no cell death, so there is no lesion for a saliency map to agree with.
# Including it would score every model as "off-lesion" by construction.
test <- test[test$label > 0, ]
cat(sprintf("test crops with score >= 1: %d\n", nrow(test)))

rows <- vector("list", nrow(test))
for (i in seq_len(nrow(test))) {
  fn <- test$filename[i]
  lab <- test$label[i]
  path <- file.path(crop_dir, as.character(lab), fn)
  if (!file.exists(path)) {
    stop(sprintf("missing crop: %s", path))
  }
  img <- load_img(path)
  seg <- segment_cda(img)

  # Downsample the binary mask to the 64x64 network input. EBImage::resize
  # interpolates, so re-threshold at 0.5 to get a clean binary mask back.
  #
  # No transpose anywhere in this chain. readTIFF returns [row, col, channel] and
  # writePNG expects [row, col], and EBImage::Image stores the matrix it is given
  # without reordering, so the row/col convention survives end to end. An earlier
  # version transposed on write, which put every mask on the image's diagonal
  # reflection -- caught by checking mean (G - R) inside vs outside the mask, since
  # that difference is the very quantity Otsu thresholds on, and the transposed masks
  # separated it markedly worse (0.178 vs 0.238).
  m64 <- EBImage::resize(EBImage::Image(seg$mask * 1.0), w = 64, h = 64)
  m64 <- as.matrix(m64) > 0.5

  id <- sub("\\.tif$", "", fn)
  writePNG(m64 * 1.0, file.path(out_dir, "masks", paste0(id, ".png")))

  rows[[i]] <- data.frame(
    image_id = id, filename = fn, score = lab,
    cda_frac_native = seg$cda_frac, cda_frac_64 = mean(m64),
    native_h = dim(img)[1], native_w = dim(img)[2],
    stringsAsFactors = FALSE
  )
  if (i %% 25 == 0) cat(sprintf("  %d/%d\n", i, nrow(test)))
}

meta <- do.call(rbind, rows)
write.csv(meta, file.path(out_dir, "mask_index.csv"), row.names = FALSE)
cat(sprintf(
  "wrote %d masks to %s\nmean CDA fraction at 64x64: %.4f\n",
  nrow(meta), file.path(out_dir, "masks"), mean(meta$cda_frac_64)
))

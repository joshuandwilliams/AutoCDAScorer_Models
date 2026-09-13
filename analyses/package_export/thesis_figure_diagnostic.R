# Chapter 3 #fig-autocdascorer-diagnostic. Panel A is the three diagnostic styles on the
# PC1-PC2 pair alone, panel B the convex-hull style over all five components, which is
# what diagnostic_pca() draws. Both use the package's own example crops against the
# ensemble's training cloud, so the figure shows what a user actually gets.

library(ggplot2)
suppressMessages(devtools::load_all("/Users/jowillia/Packages/AutoCDAScorer", quiet = TRUE))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
out <- file.path(here, "thesis_plots")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

crops <- system.file("extdata", "example_dataset", "cropped_images",
  package = "AutoCDAScorer", mustWork = TRUE
)
data <- load_images(crops)
pca <- AutoCDAScorer:::load_result_pca("ensemble")
feats <- AutoCDAScorer:::pca_transform(data, pca)

# The single-pair functions strip their axes and legend for the grid in panel B, so put
# both back here, where the three styles are being compared side by side.
titled <- function(p, title) {
  p + labs(title = title, x = "PC 1", y = "PC 2") +
    theme(
      plot.title = element_text(size = 11, hjust = 0.5, margin = margin(b = 4)),
      axis.title = element_text(size = 10),
      plot.margin = margin(2, 4, 2, 4)
    )
}

row <- cowplot::plot_grid(
  titled(AutoCDAScorer:::pca_diagnostic_target(pca, feats, 1, 2), "Ellipses"),
  titled(AutoCDAScorer:::pca_diagnostic_density(pca, feats, 1, 2), "Density"),
  titled(AutoCDAScorer:::pca_diagnostic_convexhull(pca, feats, 1, 2), "Convex hull"),
  nrow = 1
)

# No legend here: panel B carries the same one immediately below.

# Both panels are saved at the same physical width. The assembler gives them one common
# display width, so equal saved widths mean one common scale factor, and a size-0.5 point
# is then the same size on the page in each. Only the dpi is raised for resolution;
# widening the canvas would shrink every label relative to the panels.
W <- 6
ggsave(file.path(out, "a.png"), row, width = W, height = 2.4, dpi = 600, bg = "white")
# Four components rather than the five the README uses. num_pcs is a required argument
# with no default, so neither is more canonical, and four rows keep the cells large enough
# to read at this size. The canvas is wider than tall so the assembled figure fills the
# page; the panels stay square because the plot functions set aspect.ratio = 1, so a
# wider box only adds space between the columns.
panel_b <- diagnostic_pca("ensemble", data, num_pcs = 4, plot_type = "convexhull")
ggsave(file.path(out, "b.png"), panel_b, width = W, height = 4.6, dpi = 600, bg = "white")

# ggsave keeps whatever margin the plot leaves, and diagnostic_pca()'s grid leaves about a
# tenth of the width empty on the right. Trimming to the drawn content lets both panels
# reach the edges of the figure.
trim <- function(f) {
  img <- png::readPNG(f)
  ink <- which(apply(img[, , 1:3, drop = FALSE], c(1, 2), min) < 250 / 255, arr.ind = TRUE)
  png::writePNG(img[min(ink[, 1]):max(ink[, 1]), min(ink[, 2]):max(ink[, 2]), , drop = FALSE], f)
  d <- dim(png::readPNG(f))
  cat(sprintf("%s  %d x %d px  (aspect %.2f)\n", basename(f), d[2], d[1], d[2] / d[1]))
}
invisible(lapply(file.path(out, c("a.png", "b.png")), trim))

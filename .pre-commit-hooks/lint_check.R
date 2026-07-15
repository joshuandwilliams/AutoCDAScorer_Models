#!/usr/bin/env Rscript
# Lints the given R/Rmd/qmd files with lintr (using .lintr config) and fails
# the hook if any lints are found.
args <- commandArgs(trailingOnly = TRUE)
failed <- FALSE
for (path in args) {
  lints <- lintr::lint(path)
  if (length(lints) > 0) {
    print(lints)
    failed <- TRUE
  }
}
if (failed) quit(status = 1)

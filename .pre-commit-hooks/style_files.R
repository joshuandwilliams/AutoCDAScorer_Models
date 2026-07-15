#!/usr/bin/env Rscript
# Styles the given R/Rmd/qmd files in place with styler's tidyverse style.
args <- commandArgs(trailingOnly = TRUE)
styler::style_file(args, style = styler::tidyverse_style)

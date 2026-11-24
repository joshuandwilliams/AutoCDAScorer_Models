# Rerun: augmented pools built from train+val

## Why

`models/07` (geometric) and `models/08` (cGAN) cross-validated over the training partition
alone, while `models/03`, `04` and `05` cross-validate over train+val pooled by
`base_dataset.combine_train_val`. Two consequences:

- the 455 validation crops were never used by either augmented run;
- the augmented grids folded over 3,818 real crops at `[1627, 262, 262, 422, 508, 419, 318]`
  against the unaugmented grids' 2,289 at 327 per class, so the 2.95 pp cross-validation
  gain attributed to augmentation also contains 1,984 extra real crops and 455 missing ones.

The rerun makes the augmented and unaugmented runs differ only by the augmented copies.

## What changed in the code

- `analyses/01_generate_base_dataset/build_medoid_64_aug.py` and `build_medoid_64_cgan.py`
  now pool train+val. `--train-only` reproduces the original pool.
- `src/random_search.py` records `avg_vaf_orig_macro`: the mean of the seven per-class
  accuracies over the REAL crops of each validation fold, averaged across folds. The real
  crops carry the training partition's own imbalance, so the pooled `avg_vaf_orig` rewards
  accuracy on score 0; macro-averaging removes that.
- Ranking is unchanged: `avg_vaf`, the pooled accuracy over the whole fold, which is what
  `TopNStore` also saves on. `avg_vaf_orig` and `avg_vaf_orig_macro` are recorded for
  reporting, not for selection.

## Run order

The rerun writes to new folders, so the current pools and results are not touched.

1. Build both pools on the Mac. The builders read the raw crops, the raw photographs and
   the medoid box table in uv-agroinfiltration-dataset, which are all there. The
   geometric pool takes about 20 minutes. The cGAN pool takes longer, because it also
   samples the generator.

   ```bash
   python analyses/01_generate_base_dataset/build_trainval_pools.py aug  --target 5000
   python analyses/01_generate_base_dataset/build_trainval_pools.py cgan --target 5000
   ```

   They are written to `data/datasets/medoid_64_aug_trainval/` and
   `data/datasets/medoid_64_cgan_trainval/`.

2. Copy the code and the two new pools to the cluster. `sync_to_hpc.sh --code-only`
   copies the code, including the two new run folders. The pools are in `data/`, which
   `--code-only` skips, so copy just those two folders:

   ```bash
   ./scripts/sync_to_hpc.sh --code-only
   rsync -av data/datasets/medoid_64_aug_trainval data/datasets/medoid_64_cgan_trainval \
       slurm:agroinfiltration_scoring/AutoCDAScorer_Models/data/datasets/
   ```

3. Submit both grids from their new folders, on the cluster:

   ```bash
   cd agroinfiltration_scoring/AutoCDAScorer_Models/models/07_Medoid_CNN_GeomAug/03_MatchLargestClass_TrainVal
   sbatch run_search.slurm.sh
   cd ../../08_Medoid_CNN_CGANAug/03_MatchLargestClass_TrainVal
   sbatch run_search.slurm.sh
   ```

   As for the original runs, resubmit the same command until each
   `random_search_results.csv` has 8,640 rows. `per_class 0` means "match the largest
   class". The pool is now train+val, so that target rises from 1,627 to the size of the
   largest class in the pooled split.

Cost, from `train_seconds` in the existing CSVs: the geometric grid's 6,720 settings took a
median 1,678 s each over 5 folds, so about 3,100 CPU-hours spread over 200 tasks. The pool
grows by roughly a fifth, so budget the same again plus 20%.

## After the rerun

Pull the results back with `./scripts/sync_from_hpc.sh --path models/07_Medoid_CNN_GeomAug`
and `--path models/08_Medoid_CNN_CGANAug`. The new results are in the
`03_MatchLargestClass_TrainVal` folders. The analyses still read `01_MatchLargestClass`,
so compare the two before changing anything that the thesis reports.

If the new results replace the old ones, rebuild the ensembling cache
(`python analyses/14_ensembling/build_cache.py geo cgan`), re-render `analyses/04, 08, 09,
10, 12, 14` and rebuild the thesis figures. Every number in Chapter 3's P9, P11, P12, P15,
P16 and P17 would change.

## What the chapter says until then

The submitted version reports the train-only pools. The Methods sentence to carry is that
the augmented runs cross-validate over the training partition topped up by augmentation,
while the unaugmented grids cross-validate over train+val, so the comparison is between
pipelines rather than an isolated measurement of augmentation.

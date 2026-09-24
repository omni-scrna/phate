# phate

Omnibenchmark **module** wrapping [KrishnaswamyLab/PHATE](https://github.com/KrishnaswamyLab/PHATE)
2.0.0 for the `PHATE` stage of
[split-stages-plan](https://github.com/omni-scrna/split-stages-plan).

| entrypoint | function | reads | writes |
|---|---|---|---|
| `phate` | `phate.PHATE(...).fit_transform(X)` | `normalized_selected_h5` | `{name}_embedding.tsv` (`cell_id`, `dim_1..dim_k`) |

A sibling of the `PCA` / `ISOMAP` / `CNTFCT` stages: same input, same
`embedding_tsv` contract, so the embedding metrics and kNN stages fan out over
it unchanged. There is no `loadings_tsv`, so `INTG8` does not consume it.

Parameters (all required): `--n_components`, `--knn`, `--decay`,
`--n_landmark`, `--n_pca`, `--random_seed`. `t` is left at PHATE's `"auto"`.
`n_jobs` comes from `OMP_NUM_THREADS`, which Snakemake sets to the rule's
threads.

The entrypoint is `run_phate.py`, not `phate.py`, because a script named
`phate.py` would shadow the upstream package on `import phate`.

## Environment

linux-64 only. `phate` comes from PyPI rather than bioconda: the bioconda
2.0.0 recipe still depends on `scprep`, which caps pandas below 2.1 and so
Python at 3.11. See `pixi.toml`.

```sh
pixi run check        # imports work
pixi run -e test test
pixi run export-env   # regenerate envs/phate.yml
pixi run -e dev sync  # refresh src/common/ from boilerplate + the plan's schema/
```

The module is MIT; upstream PHATE is GPL-2.0-only and is installed, not vendored.

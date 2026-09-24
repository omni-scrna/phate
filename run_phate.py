#!/usr/bin/env python3
"""PHATE module for omnibenchmark.

Sibling of the PCA / ISOMAP / CNTFCT stages, not downstream of them: reads the
same normalized_selected_h5 matrix and produces the same embedding_tsv output
contract, so EMBED-M/NNG/CLUST-E consume it like any other embedding. No
loadings_tsv (PHATE has no gene -> dimension map), so INTG8 drops it.

Output
------
File: {output_dir}/{name}_embedding.tsv
  `cell_id  dim_1  ...  dim_{n_components}`, one row per cell -- the same shape
  and column naming as sklearn's Isomap embedding.

PHATE's own landmarking (--n_landmark) keeps the diffusion operator at
(n_landmark, n_landmark), so this scales with cell count the way L-Isomap does.
PHATE also runs its own PCA (--n_pca) on the gene-selected input before
building the kNN graph.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import phate
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common) + module-local writers
from common import cli  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PHATE module")
    cli.add_base_args(p)
    cli.add_stage_args(p, "PHATE")
    p.add_argument("--n_components", type=int, required=True,
                   help="Number of PHATE embedding dimensions")
    p.add_argument("--knn", type=int, required=True,
                   help="Nearest neighbors for the alpha-decay kernel")
    p.add_argument("--decay", type=float, required=True,
                   help="Alpha-decay kernel exponent")
    p.add_argument("--n_landmark", type=int, required=True,
                   help="Landmarks for the diffusion operator")
    p.add_argument("--n_pca", type=int, required=True,
                   help="PCs PHATE computes internally before the kNN graph")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Seed for PCA, landmarking and SGD-MDS")
    return p.parse_args()


def load_matrix(h5_path) -> tuple[list[str], sp.csr_matrix]:
    """Read the normalized, gene-selected matrix as a sparse cells-by-genes
    CSR array. Same on-disk layout and orientation fix as sklearn/isomap.py's
    load_matrix: the file stores genes-by-cells CSC, transposed here."""
    with h5py.File(h5_path, "r") as h5:
        g = h5["matrix"]
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        shape = tuple(g["shape"][:])
        cell_ids = g["barcodes"][:].astype(str)

    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # cells x genes
    return list(cell_ids), X


def run_phate(X, n_components, knn, decay, n_landmark, n_pca, random_seed,
              n_jobs=1) -> np.ndarray:
    """Fit PHATE on X (cells x features, dense or sparse) and return the
    (n_cells, n_components) embedding. t stays at PHATE's "auto" (knee of the
    von Neumann entropy curve)."""
    if X.shape[0] <= knn:
        raise ValueError(
            f"knn ({knn}) must be smaller than the number of cells ({X.shape[0]})."
        )

    op = phate.PHATE(
        n_components=n_components, knn=knn, decay=decay,
        n_landmark=n_landmark, n_pca=n_pca, random_state=random_seed,
        n_jobs=n_jobs, verbose=1,
    )
    embedding = np.asarray(op.fit_transform(X), dtype=np.float64)

    assert embedding.shape == (X.shape[0], n_components), embedding.shape
    if not np.isfinite(embedding).all():
        raise ValueError("PHATE returned non-finite coordinates.")
    return embedding


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "normalized_selected_h5", "n_components",
              "knn", "decay", "n_landmark", "n_pca", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    # Snakemake exports OMP_NUM_THREADS = <rule threads>; PHATE's n_jobs=-1
    # would take every core on the host instead.
    n_jobs = int(os.environ.get("OMP_NUM_THREADS", "1"))
    print(f"  n_jobs: {n_jobs}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cell_ids, X = load_matrix(args.normalized_selected_h5)
    print(f"  input: {X.shape[0]} cells x {X.shape[1]} genes")

    embedding = run_phate(
        X, n_components=args.n_components, knn=args.knn, decay=args.decay,
        n_landmark=args.n_landmark, n_pca=args.n_pca,
        random_seed=args.random_seed, n_jobs=n_jobs,
    )

    col_names = [f"dim_{i + 1}" for i in range(embedding.shape[1])]
    embedding_out = Path(args.output_dir) / f"{args.name}_embedding.tsv"
    write_embeddings(Embedding(embedding, cell_ids, col_names), embedding_out)
    print(f"  wrote: {embedding_out}")


if __name__ == "__main__":
    main()

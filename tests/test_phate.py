from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from run_phate import load_matrix, run_phate

PARAMS = dict(knn=5, decay=40, n_landmark=50, n_pca=10, random_seed=42)


def _write_matrix_h5(path: Path, X: sp.csr_matrix, cell_ids, gene_ids):
    """Genes-by-cells CSC on disk, matching the normalized_selected_h5 contract."""
    Xt = X.T.tocsc()  # genes x cells
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        g.create_dataset("data", data=Xt.data)
        g.create_dataset("indices", data=Xt.indices)
        g.create_dataset("indptr", data=Xt.indptr)
        g.create_dataset("shape", data=np.array(Xt.shape))
        g.create_dataset("genes", data=np.array(gene_ids, dtype="S"))
        g.create_dataset("barcodes", data=np.array(cell_ids, dtype="S"))


def _blobs(n_cells=200, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5, size=(3, n_genes))
    X = centers[rng.integers(3, size=n_cells)] + rng.normal(size=(n_cells, n_genes))
    return sp.csr_matrix(np.clip(X, 0, None))


def test_load_matrix_reads_cells_by_genes_sparse(tmp_path: Path):
    X = sp.csr_matrix(np.array(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
    ))
    h5_path = tmp_path / "fixture_normalized_selected.h5"
    _write_matrix_h5(h5_path, X, ["cell_a", "cell_b", "cell_c"], ["g1", "g2", "g3"])

    cell_ids, loaded = load_matrix(h5_path)

    assert cell_ids == ["cell_a", "cell_b", "cell_c"]
    assert sp.issparse(loaded)
    np.testing.assert_allclose(loaded.toarray(), X.toarray())


@pytest.mark.parametrize("n_components", [2, 10])
def test_run_phate_shape_and_finite(n_components):
    embedding = run_phate(_blobs(), n_components=n_components, **PARAMS)

    assert embedding.shape == (200, n_components)
    assert np.isfinite(embedding).all()


def test_run_phate_is_seeded():
    X = _blobs()
    np.testing.assert_allclose(run_phate(X, n_components=2, **PARAMS),
                               run_phate(X, n_components=2, **PARAMS))


def test_run_phate_rejects_knn_not_below_n_cells():
    with pytest.raises(ValueError, match="knn"):
        run_phate(_blobs(n_cells=5), n_components=2, **PARAMS)

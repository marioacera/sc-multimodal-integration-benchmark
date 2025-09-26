#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap

# ---------------- CONFIG ----------------
BASE_DIR = Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/reck")
RNA_FILE = BASE_DIR / "rna.h5ad"
FRAG_DIR = BASE_DIR / "fragments"
OUTDIR   = Path("./outputs")
GENOME   = "hg38"
BIN_SIZE = 500
MIN_FRAG = 1000
N_HVG    = 3000
N_COMPS  = 30
MNC_DIMS = 30

(OUTDIR / "tmp_import").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(OUTDIR / "tmp_import"))

print(f"[info] SnapATAC2={snap.__version__}  Scanpy={sc.__version__}", file=sys.stderr)
print(f"[info] RNA: {RNA_FILE}", file=sys.stderr)
print(f"[info] FRAGMENTS dir: {FRAG_DIR}", file=sys.stderr)
OUTDIR.mkdir(parents=True, exist_ok=True)

# 1) Collect fragment files
frag_map = {}
for p in sorted(FRAG_DIR.glob("*.tsv.gz")):
    if str(p).endswith(".gz.1"):
        continue
    m = re.search(r"Library([1-6])_atac_fragments\.tsv\.gz$", p.name)
    if m:
        frag_map[f"l{m.group(1)}"] = p
if not frag_map:
    raise FileNotFoundError(f"No fragment files *.tsv.gz found in {FRAG_DIR}")
print(f"[info] Libraries found: {sorted(frag_map)}", file=sys.stderr)

h5ads=[]
print(OUTDIR)
for sample_key, frag in sorted(frag_map.items()):
    out_h5ad = OUTDIR / f"{sample_key}_atac_bin.h5ad"
    # adatas[i].write(out_h5ad)
    h5ads.append((sample_key, out_h5ad))

# 3) AnnDataSet + global features
ds_file = OUTDIR / "atac_merged_HORIZONTAL.h5ads"
atac_ds = snap.AnnDataSet(adatas=h5ads, filename=str(ds_file))
#atac_ds = atac_ds.to_adata()
atac_ds = atac_ds.to_adata()
snap.pp.select_features(atac_ds, n_features=200000)
atac_ds = atac_ds[:,atac_ds.var['selected']].copy()

print(f"[info] ATAC cells total: {len(list(atac_ds.obs_names))}", file=sys.stderr)

# 4) RNA prep (HVGs if missing) + sample from prefix
rna = sc.read_h5ad(str(RNA_FILE))
# if "sample" not in rna.obs:
rna_names = list(map(str, rna.obs_names))
samples = [s.split("_", 1)[0] if "_" in s else "" for s in rna_names]
rna.obs["sample"] = samples  # plain strings (safe)

if "highly_variable" in rna.var:
    hv = rna.var["highly_variable"].to_numpy() if hasattr(rna.var["highly_variable"], "to_numpy") else rna.var["highly_variable"].values
    if hv.dtype == bool and hv.sum() > 0:
        rna = rna[:, hv].copy()
else:
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=N_HVG)
    rna = rna[:, rna.var["highly_variable"].to_numpy()].copy()

print(f"[info] RNA cells: {rna.n_obs}  genes: {rna.n_vars}", file=sys.stderr)

# 5) Intersect cells and align order
atac_names = list(map(str, atac_ds.obs_names))
rna_names  = list(map(str, rna.obs_names))
aset = set(atac_names)
inter = [n for n in rna_names if n in aset]
if not inter:
    raise ValueError("No shared barcodes between RNA and ATAC.")
rna = rna[inter, :].copy()
atac_ds = atac_ds[inter,:].copy()



data = atac_ds
snap.tl.spectral(
    data,
    n_comps=30,
    features="selected",
    random_state=0,
    distance_metric="cosine",
)  # :contentReference[oaicite:4]{index=4}

# ---- MNC-correct en el embedding espectral ----
snap.pp.mnc_correct(
    data,
    batch="sample",
    use_rep="X_spectral",
    use_dims=30,
    key_added="X_spectral_mnn",
    n_jobs=8,
)  # :contentReference[oaicite:5]{index=5}

# ---- export CSVs ----
def _to_csv(mat_key: str, out_csv: Path):
    M = np.asarray(data.obsm[mat_key])
    idx = np.asarray(list(map(str, data.obs_names)))
    cols = [f"X{i+1}" for i in range(M.shape[1])]
    pd.DataFrame(M, index=idx, columns=cols).to_csv(out_csv, index=True)

_to_csv("X_spectral", OUTDIR / "atac_spectral.csv")
_to_csv("X_spectral_mnn", OUTDIR / "atac_spectral_mnc.csv")


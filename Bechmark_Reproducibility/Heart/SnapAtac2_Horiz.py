from pathlib import Path
import pandas as pd
import numpy as np
import snapatac2 as snap
import scanpy as sc

# ---------------------------------------------------------------------
# 1) Inputs
# ---------------------------------------------------------------------

def sample_from_fname(p: Path) -> str:
    name = p.name
    for suf in ("_atac_fragments.tsv.gz", "_atac_fragments.tsv.bgz"):
        if name.endswith(suf):
            return name[:-len(suf)]
    raise ValueError(f"Unexpected filename: {p.name}")

def build_prefixed_metadata(rna_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Build a metadata table indexed by 'sample:barcode' with the two fields:
    'cell_type' and 'cell_state'. Missing values are allowed.
    """
    # Validación mínima
    for col in ("cell_type", "cell_state", "sample", "bare_barcode"):
        if col not in rna_obs.columns:
            raise KeyError(f"Missing column '{col}' in RNA_CSV table")

    prefixed = (
        rna_obs
        .assign(pref_id = rna_obs["sample"].astype(str) + ":" + rna_obs["bare_barcode"].astype(str))
        .set_index("pref_id")
        [["cell_type", "cell_state"]]
        .copy()
    )
    return prefixed

def assign_metadata_to_ad(ad_obj, meta_df: pd.DataFrame, default_ct="unknown", default_cs="unknown"):
    """
    Align meta_df (indexed by prefixed obs_names) to ad_obj.obs_names and attach
    'cell_type' and 'cell_state' columns. Works with backed AnnData from snapatac2.
    """
    idx = pd.Index(ad_obj.obs_names.astype(str))
    m = meta_df.reindex(idx)
    # Evitar dtype pandas 'string' que a veces molesta en backed
    ct = m["cell_type"].astype("object").fillna(default_ct).values
    cs = m["cell_state"].astype("object").fillna(default_cs).values
    ad_obj.obs["cell_type"] = pd.Series(ct, index=idx, dtype="object")
    ad_obj.obs["cell_state"] = pd.Series(cs, index=idx, dtype="object")

def main(first_load=False):

    FRAG_DIR = Path("/mnt/beegfs/macera/CZI/external_ref/HeartAtlas/fragments_dir/")
    RNA_CSV  = Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATAC/objects/Heart_Atac_obs.csv")
    OUT_DIR  = Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATAC/objects/")
    KEEP_DIR = OUT_DIR
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_TILE=False
    # ---------------------------------------------------------------------
    # 2) Discover fragment files (keep .tsv.gz / .bgz, skip .tbi)
    # ---------------------------------------------------------------------
    fragment_files = sorted([
        p for p in FRAG_DIR.glob("*_atac_fragments.tsv.gz*")
        if p.suffix in {".gz", ".bgz"} and all(suf != ".tbi" for suf in p.suffixes)
    ])


    sample_names = [sample_from_fname(p) for p in fragment_files]

    # ---------------------------------------------------------------------
    # 3) Load keep-list CSV
    #     - index: <combinedID>_<barcode> (e.g., HCAHeartST..._GCGG...-1)
    #     - we need two fields:
    #         * sample == combinedID
    #         * bare_barcode == text after last '_' in the index
    # ---------------------------------------------------------------------
    rna_obs = pd.read_csv(RNA_CSV, index_col=0)
    rna_obs["sample"] = rna_obs["combinedID"]
    rna_obs["bare_barcode"] = rna_obs.index.to_series().str.rsplit("_", n=1).str[-1]

    # ---------------------------------------------------------------------
    # 4) Write per-sample keep files with **bare barcodes**
    barcode_files = []
    for samp in sample_names:
        keep_bcs = rna_obs.loc[rna_obs["sample"] == samp, "bare_barcode"]
        txt = KEEP_DIR / f"{samp}_barcodes.txt"
        # Ensure unique + non-empty
        keep_bcs.dropna().drop_duplicates().to_csv(txt, index=False, header=False)
        barcode_files.append(str(txt))
    print(barcode_files)
    # ---------------------------------------------------------------------
    # 5) Import with SnapATAC2 (one AnnData per sample), write to disk too
    #     NOTE: depending on your snapatac2 version, the keyword is `barcodes=`
    #           or `barcodes_file=`. Try `barcodes=` first (most recent).
    # ---------------------------------------------------------------------
    if BUILD_TILE:
        if first_load:
            adatas = snap.pp.import_data(
                [str(p) for p in fragment_files],
                chrom_sizes=snap.genome.GRCh38,
                sorted_by_barcode=False,
                n_jobs=1,
                file=[str(OUT_DIR / f"{s}.h5ad") for s in sample_names],
            )
        else:
            adatas = []
            files = [str(OUT_DIR / f"{s}.h5ad") for s in sample_names]
            for s, f in zip(sample_names, files):
                ad = snap.read(str(f), backed="r+")
                if ad.shape[0] == 0:
                    continue
                # 1) Prefix obs_names once
                idx = pd.Index(ad.obs_names)
                ad.obs_names = pd.Index([f"{s}:{bc}" for bc in idx], name=idx.name)

                # 2) Assign a proper length-matched 'sample' column (backed-safe)
                n = ad.n_obs
                ad.obs["sample"] = pd.Series([s] * n, index=pd.Index(ad.obs_names), dtype="string")

                # 3) Build the prefixed keep-set for THIS sample from RNA_CSV
                keep_prefixed = set(
                    f"{s}:{bc}"
                    for bc in rna_obs.loc[rna_obs["sample"] == s, "bare_barcode"]
                        .dropna()
                        .astype(str)
                        .values
                )

                # 4) Boolean mask over obs_names (fast; works with AnnData backed)
                names_arr = np.asarray(ad.obs_names, dtype=object)
                mask = np.fromiter((nm in keep_prefixed for nm in names_arr), dtype=bool, count=names_arr.size)
                print(ad, np.sum(mask))
                # 5) Subset in place
    #            ad.subset(obs_indices=mask)

                adatas.append(ad)


        # --------------------------------------------------------------------
        snap.pp.add_tile_matrix(adatas, bin_size=5000)

        # ---------------------------------------------------------------------
        # 7) Pack into an AnnDataSet on disk
        # ---------------------------------------------------------------------
        data = snap.AnnDataSet(
            adatas=[(str(s), a) for s, a in zip(sample_names, adatas)],
            filename="/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/HEART_Integrated_raw.h5ads",
        )

    else:
        data = snap.read_dataset("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/Integrated_raw.h5ads")

    data = data.to_adata()
    print(data, data.obs)
    data.write(
        "/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATAC/objects/HEART_Integrated_raw.h5ad",
        compression="gzip",
    )

    print(f"Number of cells: {data.n_obs}")
    print(f"Number of unique barcodes: {np.unique(data.obs_names).size}")

    n_feat = 200000  # 200k is possible but heavy; adjust to taste
    snap.pp.select_features(data, n_features=n_feat)

    snap.tl.spectral(data)  # stores in data.obsm["X_spectral"]

    snap.tl.umap(data)
    data.obsm["X_umap_raw"] = data.obsm["X_umap"]

    # ---------------------------------------------------------------------
    # 9) Batch correction
    #     Build a 'batch_combined' that actually exists: here sample + donor
    # ---------------------------------------------------------------------
    Batch = "sample"

    snap.pp.mnc_correct(data, batch=Batch)  # result in obsm["X_spectral_mnn"]
    snap.tl.umap(data, use_rep="X_spectral_mnn")
    data.obsm["X_umap_spectral_mnn"] = data.obsm["X_umap"]

    snap.pp.harmony(data, batch=Batch, max_iter_harmony=20)
    snap.tl.umap(data,use_rep="X_spectral_harmony")
    data.obsm["X_umap_spectral_harmony"] = data.obsm["X_umap"]


    pd.DataFrame(data.obsm['X_spectral_mnn'], index = data.obs.index).to_csv('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATAC/csv/Snap_Spectral_multi_mnn.csv')
    pd.DataFrame(data.obsm['X_spectral_harmony'], index = data.obs.index).to_csv('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATA/Ccsv/Snap_Spectral_multi_harmony.csv')


    # ---------------------------------------------------------------------
    # 10) Write integrated dataset
    # ---------------------------------------------------------------------
    data.write(
        "/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATAC/objects/HEART_Integrated_combined_corr_200kfeat.h5ad",
        compression="gzip",
    )


if __name__ == "__main__":
    # Don’t force a start method; SnapATAC2 requests "spawn" internally.
    # The guard prevents recursive re-exec on workers.
    main()

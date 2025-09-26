import snapatac2 as snap
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

snap.__version__

def import_all_fragments():
    fragment_files = [
        '/mnt/beegfs/macera/CZI/MULTI/ARC/AGG/outs/atac_fragments.tsv.gz',
        '/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/wilson/fragments.tsv.bgz',
        '/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/muto/fragments.tsv.bgz',
    ]
    sample_names = ['AGG', 'wilson', 'muto']

    # load keep-lists
    df_agg    = pd.read_csv('/mnt/beegfs/macera/CZI/Downstream/HORIZONTAL_Integration/RNA_final/csv/CLUSTERS_METADATA.csv',    index_col=0)
    df_wilson = pd.read_csv('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/wilson/Wilson_atac_obs.csv', index_col=0)
    df_muto   = pd.read_csv('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/muto/Muto_atac_obs.csv',   index_col=0)

    df_agg['donor_id'] = df_agg['sample']
    df_agg['batch']    = '10x Multiome'
    keep_dfs = [df_agg, df_wilson, df_muto]

    # perform import
    adatas = []
    files=['/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/'+ name + '_tile5000_processed.h5ad' for name in sample_names]
    for f in files:
        adatas.append(sc.read(f))
    # ... whatever you do next with adatas and keep_dfs …
    return adatas, keep_dfs

def main():
    sample_names = ["AGG", "wilson", "muto"]
    files = [
        Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/AGG_save.h5ad"),
        Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/wilson_save.h5ad"),
        Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/muto_save.h5ad"),
    ]


    sample_names = ['AGG', 'wilson', 'muto']

    # load keep-lists
    df_agg    = pd.read_csv('/mnt/beegfs/macera/CZI/Downstream/HORIZONTAL_Integration/RNA_final/csv/CLUSTERS_METADATA.csv',index_col=0)
    df_wilson = pd.read_csv('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/wilson/Wilson_atac_obs.csv', index_col=0)
    df_muto   = pd.read_csv('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/data/muto/Muto_atac_obs.csv',   index_col=0)
    print(df_agg.shape)

    df_agg = df_agg.loc[df_agg['batch']=='snRNA'].copy()
    df_agg.index = [i.replace('-snRNA','') for i in df_agg.index]
    df_agg['batch']    = '10x Multiome'
    df_agg['assay']    = '10x Multiome'
    df_agg['donor_id'] = df_agg['sample']

    print(df_agg.shape, df_agg)
    keep_dfs = [df_agg, df_wilson, df_muto]

    adatas = []
    files_=['/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/'+ name +'.h5ad' for name in sample_names]
    for f in files_:
        adatas.append(sc.read(f))

    # Sanity: make sure they exist
    for f in files:
        assert f.exists(), f"Missing file: {f}"

    for adata, sample, df in zip(adatas, sample_names, keep_dfs):
        
        adata.obs['dataset'] = str(sample)
        # rename obs_names to "sample:barcode":
        print(df.index, adata.obs)
        mask = adata.obs_names.isin(df.index.astype(str))
        adata = adata[mask].copy()

        df = df[df.index.isin(adata.obs.index)]
        df = df.loc[adata.obs.index]
        for col in df.columns:
            adata.obs[col]=df[col]
        adata.obs['sample'] = adata.obs['donor_id']
        adata.obs['batch'] = adata.obs['assay']
        adata.obs_names = [f"{sample}:{bc}" for bc in adata.obs_names]

        adata.write(f'/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/{sample}_save.h5ad')

    # Build dataset (stores links to the component files, does NOT copy them)
    dataset_path = Path("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/Integrated.h5ads")


    snap.pp.add_tile_matrix(adatas, bin_size=5000)


    data = snap.AnnDataSet(
        adatas=[(name, str(path)) for name, path in zip(sample_names, files)],
        filename=str(dataset_path),
        add_key="sample",        # will add .obs['sample'] with your keys
        # backend="hdf5",        # optional; default is fine
    )

    n_feat = 50000

    print(f'Number of cells: {data.n_obs}')
    print(f'Number of unique barcodes: {np.unique(data.obs_names).size}')
    ###----------------###

    snap.pp.select_features(data, n_features=n_feat)

    snap.tl.spectral(data)

    #sc.pp.neighbors(data, n_neighbors=50, use_rep='X_spectral')

    snap.tl.umap(data)

    data.obsm['X_umap_raw'] = data.obsm['X_umap']

    ###----------------###

    Batch = 'sample'
#    data.obs['batch_combined'] = data.obs['sample'].astype(str) + "_" + data.obs['dataset'].astype(str)

    # Materialise both columns as strings
  #  sample_arr  = data.obs["sample"].to_numpy().astype(str)
   # dataset_arr = data.obs["dataset"].to_numpy().astype(str)

    # Concatenate element-wise
    #batch_combined = pd.Series(sample_arr + "_" + dataset_arr, index=data.obs_names)

    snap.pp.mnc_correct(data, batch=Batch)

    #sc.pp.neighbors(data, n_neighbors=50, use_rep='X_spectral_mnn')

    snap.tl.umap(data, use_rep='X_spectral_mnn')

    data.obsm['X_umap_spectral_mnn_sample'] = data.obsm['X_umap']

    snap.pp.harmony(data, batch=Batch, max_iter_harmony=20) # Stored in X_spectral_harmony
    
    snap.tl.umap(data, use_rep='X_spectral_harmony')

    data.obsm['X_umap_spectral_harmony_sample'] = data.obsm['X_umap']

    #data.write("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/Integrated_combined_corr.h5ads", compression='gzip')

    try:
        data = data.to_adata()
        data.write("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/Integrated_corr_50k.h5ads", compression='gzip')
    except:
        data.close()

    ###----------------###

    Batch = 'dataset'
    snap.pp.mnc_correct(data, batch=Batch)

    #sc.pp.neighbors(data, n_neighbors=50, use_rep='X_spectral_mnn')

    snap.tl.umap(data)

    data.obsm['X_umap_spectral_mnn_dataset'] = data.obsm['X_umap']

    snap.pp.harmony(data, batch="sample", max_iter_harmony=20) # Stored in X_spectral_harmony

    data.obsm['X_umap_spectral_harmony_dataset'] = data.obsm['X_umap']
    try:
        data = data.to_adata()
        data.write("/mnt/beegfs/macera/CZI/Downstream/REVIEWS/objects/ATAC/Integrated_corr_50k.h5ads", compression='gzip')
    except:
        data.close()
if __name__ == "__main__":
    main()


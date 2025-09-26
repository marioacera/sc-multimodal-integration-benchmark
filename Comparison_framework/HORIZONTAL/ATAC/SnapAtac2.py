import snapatac2 as snap
import scanpy as sc
import pandas as pd
import numpy as np

snap.__version__

df = pd.read_csv('csv/atac_cells.csv', index_col=0)
df_multi = pd.read_csv('csv/multiome_cells.csv', index_col=0)

data = snap.pp.import_data(
    '/mnt/beegfs/macera/CZI/MULTI/ARC/AGG/outs/atac_fragments.tsv.gz',
    chrom_sizes=snap.genome.GRCh38,
    file='objects/snapatac_fragments_Ch38.h5ad',  # Optional
    sorted_by_barcode=False,n_jobs=-1
)
data.close()

data = snap.read('objects/snapatac_fragments_Ch38.h5ad', backed=None)

data = data[data.obs.index.isin(list(df.index))].copy()

snap.metrics.tsse(data, snap.genome.hg38)
print('adata pre tile', data)

snap.pp.add_tile_matrix(data, bin_size=5000)

print('adata post tile', data)

data.write('objects/snap_basic.h5ad', compression='gzip')

data = data[data.obs.index.isin(list(df.index))].copy()

feat = 200000

snap.pp.select_features(data, n_features=feat)

data.var[f'selected_{feat}'] = data.var['selected'].copy()

snap.pp.scrublet(data)

condition = (data.obs['doublet_probability'] > 0.5) & (~data.obs.index.isin(df_multi.index))

data = data[~condition].copy()

snap.tl.spectral(data)

snap.pp.mnc_correct(data, batch="sample")

snap.pp.harmony(data, batch="sample", max_iter_harmony=20)

sc.pp.neighbors(data, n_neighbors=50, use_rep='X_spectral_mnn')

sc.tl.umap(data)

data.obsm['X_umap_spectral_mnn'] = data.obsm['X_umap']

snap.pp.harmony(data, batch="sample", max_iter_harmony=20) # Stored in X_spectral_harmony

data.write(f'objects/snapatac_FINAL.h5ad', compression='gzip')


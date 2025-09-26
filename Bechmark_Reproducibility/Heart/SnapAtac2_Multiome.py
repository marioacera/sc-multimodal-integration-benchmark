from __future__ import annotations

import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix as sparse

from muon import MuData
import muon as mu
import snapatac2 as snap
from scipy.sparse import csc_matrix
import scipy.sparse as sp


import re
import numpy as np
import pandas as pd

def _canonicalize_sparse_inplace(ad):
    # Canonicalize .X
    if sp.issparse(ad.X):
        X = ad.X.tocsr()
        X.sum_duplicates()
        X.sort_indices()
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        ad.X = X
    # Canonicalize any sparse layers too (if SnapATAC2 stored tiles in a layer)
    if hasattr(ad, "layers") and len(ad.layers):
        for k in list(ad.layers.keys()):
            M = ad.layers[k]
            if sp.issparse(M):
                M = M.tocsr()
                M.sum_duplicates()
                M.sort_indices()
                if M.dtype != np.float32:
                    M = M.astype(np.float32)
                ad.layers[k] = M

adata = snap.read('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART/objects/query.h5ad', backed=None)
adata = adata[adata.obs['kit_10x'].isin(['Multiome-v1'])].copy()
print(adata.obs.index[0:10])

data = snap.read('/mnt/beegfs/macera/CZI/Downstream/REVIEWS/HEART_ATAC/objects/HEART_Integrated_raw.h5ad',backed=None)
print(data.obs.index[0:10])


# -------------------------------
# Apply to your in-memory objects
# -------------------------------
# Force both to share the same suffix policy; set to None to keep as-is

# ---- Apply ----
adata.obs.index = [adata.obs.loc[i,'combinedID']+'_'+i.split('_')[-1] for i in adata.obs.index]
print(adata.obs.index[0:10])

data.obs.index = [i.split(':')[0]+'_'+i.split(':')[-1] for i in data.obs.index]
print(data.obs.index[0:10])

# Optional: verify overlap (after standardisation, obs_names should match exactly for matched cells)
common = np.intersect1d(adata.obs_names.values, data.obs_names.values)
print(f"Common cell IDs (sample:barcode): {len(common)}")

print(adata.shape, data.shape)

for ad in [adata, data]:
    _canonicalize_sparse_inplace(ad)


n_feat = 200000  # 200k is possible but heavy; adjust to taste
snap.pp.select_features(data, n_features=n_feat)
data = data[:,data.var['selected']].copy()

mdata = MuData({"rna": adata, "atac": data})
mdata
print(mdata)
print(mdata['rna'].obs,mdata['atac'].obs)
mdata.obs['sample'] = mdata['atac'].obs['sample']


mu.pp.intersect_obs(mdata)

sc.pp.highly_variable_genes(mdata["rna"], n_top_genes=4000, subset=True, flavor ='seurat_v3')
sc.pp.normalize_total(mdata['rna'], target_sum=1e4)
sc.pp.log1p(mdata['rna'])
print(mdata)

embedding = snap.tl.multi_spectral([mdata['rna'], mdata['atac']], features=None)[1]

mdata.obsm['X_spectral'] = embedding

mdata['rna'].obsm['X_spectral'] = embedding
mdata['rna'].obs['sample'] = mdata.obs['sample']
snap.pp.mnc_correct(mdata['rna'], batch="sample",use_rep='X_spectral')
snap.pp.harmony(mdata['rna'], batch="sample", max_iter_harmony=20,use_rep='X_spectral')

snap.tl.umap(mdata['rna'], use_rep="X_spectral_mnn")
snap.tl.umap(mdata['rna'], use_rep="X_spectral_harmony")

pd.DataFrame(embedding, index = mdata.obs.index).to_csv('csv/Snap_Spectral_multi_200k.csv')
pd.DataFrame(mdata['rna'].obsm['X_spectral_mnn'], index = mdata.obs.index).to_csv('csv/Snap_Spectral_multi_mnn_200k.csv')
pd.DataFrame(mdata['rna'].obsm['X_spectral_harmony'], index = mdata.obs.index).to_csv('csv/Snap_Spectral_multi_harmony_200k.csv')
mdata.var = mdata.var.astype(str)
mdata['atac'].var = mdata['atac'].var.astype(str)
mdata['rna'].var = mdata['rna'].var.astype(str)
col_dict={}
for col in mdata['rna'].var.columns:
    col_dict[col] = str(col)
mdata['rna'].var.rename(columns = col_dict, inplace=True)
mdata.write('objects/Snap_Multi.h5mu', compression='gzip')
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


adata = sc.read('objects/snRNA_raw.h5ad')
adata.obs.index = [i.replace('-snRNA','') for i in adata.obs.index] # Standarize barcode for consistent multiome mergue


data = snap.read('objects/snap_basic.h5ad', backed=None)

df = pd.read_csv('csv/atac_cells.csv', index_col=0)

data = data[data.obs.index.isin(list(df.index))].copy()

atac = sc.read('objects/atac.h5ad')
data.obs[list(atac.obs.columns)] = atac.obs[list(atac.obs.columns)] # Propagate atac metadata

snap.pp.select_features(data, n_features=200000)

data = data[:,data.var['selected']].copy() # Subset highly variable features

print(adata.X, data.X)
mdata = MuData({"rna": adata, "atac": data})
mdata
print(mdata)
print(mdata['rna'].obs,mdata['atac'].obs)
mdata.obs['sample'] = mdata['atac'].obs['sample']

mu.pp.intersect_obs(mdata)

sc.pp.highly_variable_genes(mdata["rna"], n_top_genes=5000, subset=True, flavor ='seurat_v3')
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

pd.DataFrame(embedding, index = mdata.obs.index).to_csv('csv/Snap_Spectral_multi.csv')
pd.DataFrame(mdata['rna'].obsm['X_spectral_mnn'], index = mdata.obs.index).to_csv('csv/Snap_Spectral_multi_mnn.csv')
pd.DataFrame(mdata['rna'].obsm['X_spectral_harmony'], index = mdata.obs.index).to_csv('csv/Snap_Spectral_multi_harmony.csv')

# Modifications to avoid object saving bugs

mdata.var = mdata.var.astype(str)
mdata['atac'].var = mdata['atac'].var.astype(str)
mdata['rna'].var = mdata['rna'].var.astype(str)
col_dict={}
for col in mdata['rna'].var.columns:
    col_dict[col] = str(col)
mdata['rna'].var.rename(columns = col_dict, inplace=True)

mdata.write('objects/Snap_Multi.h5mu', compression='gzip')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import scanpy as sc
import anndata
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

from scipy import sparse
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)


sc.settings.verbosity =0


import harmonypy as hm
import scanpy.external as sce
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scanorama


def split_batches(adata, batch, hvg=None, return_categories=False):
    """Split batches and preserve category information

    :param adata:
    :param batch: name of column in ``adata.obs``. The data type of the column must be of ``Category``.
    :param hvg: list of highly variable genes
    :param return_categories: whether to return the categories object of ``batch``
    """
    split = []
    batch_categories = adata.obs[batch].cat.categories
    if hvg is not None:
        adata = adata[:, hvg]
    for i in batch_categories:
        split.append(adata[adata.obs[batch] == i].copy())
    if return_categories:
        return split, batch_categories
    return split

def merge_adata(*adata_list, **kwargs):
    """Merge adatas from list while remove duplicated ``obs`` and ``var`` columns

    :param adata_list: ``anndata`` objects to be concatenated
    :param kwargs: arguments to be passed to ``anndata.AnnData.concatenate``
    """

    if len(adata_list) == 1:
        return adata_list[0]

    # Make sure that adatas do not contain duplicate columns
    for _adata in adata_list:
        for attr in ("obs", "var"):
            df = getattr(_adata, attr)
            dup_mask = df.columns.duplicated()
            if dup_mask.any():
                print(
                    f"Deleting duplicated keys `{list(df.columns[dup_mask].unique())}` from `adata.{attr}`."
                )
                setattr(_adata, attr, df.loc[:, ~dup_mask])

    return anndata.AnnData.concatenate(*adata_list, **kwargs)

def compute_latents(adata, mod, n_hvg = 4000, subset='', batch = 'sample'):
    if subset != '':        
        save_key = mod 
    else:
        save_key = mod 
    sc.pp.normalize_total(adata)     
    sc.pp.log1p(adata)               
    
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor = 'cell_ranger')
    
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata)

    sc.tl.pca(adata, svd_solver='arpack')

    pd.DataFrame(adata.obsm['X_pca'], index = adata.obs.index).to_csv(f'csv/{save_key}_Unintegrated.csv')
    
    index_number = [i.split('-')[-2] for i in adata.obs.index]
    adata = adata[adata.obs.iloc[np.argsort(index_number)].index,]

    split, categories = split_batches(adata.copy(), batch, return_categories=True)
    corrected = scanorama.correct_scanpy(split, return_dimred=True)
    corrected = merge_adata(
        *corrected, batch_key=batch, batch_categories=categories, index_unique=None
    )
    corrected = corrected[adata.obs.index]
    adata.obsm["X_scanorama"] = corrected.obsm["X_scanorama"]

    pd.DataFrame(adata.obsm['X_scanorama'], index = adata.obs.index).to_csv(f'csv/{save_key}_Scanorama.csv')
    

    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, batch)

    harmonized = pd.DataFrame(ho.Z_corr.T, index = adata.obs.index)

    # Write the adjusted PCs to a new file.
    adata.obsm['X_Harmony'] = harmonized.values
    adata.obsm['X_Harmony'] 

    pd.DataFrame(adata.obsm['X_Harmony'], index = adata.obs.index).to_csv(f'csv/{save_key}_Harmony.csv')



for mod in ['snRNA','scRNA','scRNA5p']:
    save_key = mod
    adata = sc.read(f'../objects/{mod}_raw.h5ad', compression='gzip')
    adata.X = adata.layers['counts'].copy()
    compute_latents(adata, mod,  n_hvg = 5000)


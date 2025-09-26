
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle

import scanpy as sc
import anndata
import pandas as pd
import numpy as np


from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)

sc.settings.verbosity =0

import harmonypy as hm
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


mod = 'RNA_STWG'
save_key = mod
n_hvg=5000

tech_dict = {'snRNA':'10x Multiome','scRNA':'10x scRNA(scCv3)','scRNA5p':'10x scRNA 5p(scCv2)'} 
sus_dict = {'snRNA':'nucleus','scRNA':'cell','scRNA5p':'cell'} 

adatas = []
for mod_ in ['snRNA','scRNA','scRNA5p']:
    adata = sc.read(f'objects/{mod_}_raw.h5ad', compression='gzip')
    adata.layers['counts'] = adata.X.copy()
    adata.obs['Technology'] = tech_dict[mod_]
    adata.obs['Suspension_type'] = sus_dict[mod_]

    obs = pd.read_csv('objects/Multi_obs.csv', index_col=0)
    obs = obs[obs['batch'].isin([mod_])]
    obs.index = [i.split('_')[0] for i in obs.index]
    adata.obs['In_final_obj'] = [i in obs.index for i in adata.obs.index]
    print(adata.obs['In_final_obj'].value_counts())
    
    adatas.append(adata)
adata = anndata.concat(adatas, join='outer')

adata.obs['Study'] = 'STWG'
adata.obs['Donor_id'] = adata.obs['sample']
adata.layers['counts'] = adata.X.copy()

adata = sc.read(f'objects/{save_key}.h5ad')


adata.var.to_csv(f'objects/var_{save_key}.csv')


batch_key='Technology'
batches = adata.obs[batch_key].unique()

# Create a boolean mask for genes that are expressed in all batches
gene_mask = np.ones(len(adata.var_names), dtype=bool)

for batch in batches:
    # Subset the data for the current batch
    batch_data = adata[adata.obs[batch_key] == batch]

    # Check for non-zero expression in each gene
    batch_gene_expressed = (batch_data.X > 0).sum(axis=0) > 0

    # Update the gene mask
    gene_mask = gene_mask & batch_gene_expressed

# Filter the AnnData object to keep only the genes expressed in all batches
adata = adata[:, gene_mask]

adata.obs['Scanorama_batch'] = adata.obs[batch_key].astype(str) + "_" + adata.obs['Donor_id'].astype(str)
adata.obs['Scanorama_batch'] = adata.obs['Scanorama_batch'].astype('category')

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor = 'seurat_v3',batch_key = batch_key, layer='counts')

adata = adata[:, adata.var.highly_variable].copy()

sc.pp.scale(adata)

sc.tl.pca(adata, svd_solver='arpack')

pd.DataFrame(adata.obsm['X_pca'], index = adata.obs.index).to_csv(f'csv/{save_key}_Unintegrated.csv')

ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, vars_use = [batch_key,'Suspension_type','Donor_id'])

harmonized = pd.DataFrame(ho.Z_corr.T, index = adata.obs.index)

# Write the adjusted PCs to a new file.
adata.obsm['X_Harmony'] = harmonized.values
adata.obsm['X_Harmony'] 

pd.DataFrame(adata.obsm['X_Harmony'], index = adata.obs.index).to_csv(f'csv/{save_key}_Harmony.csv')



index_number = [i.split('-')[-2] for i in adata.obs.index]
adata = adata[adata.obs.iloc[np.argsort(index_number)].index,].copy()

split, categories = split_batches(adata.copy(), batch_key, return_categories=True)

corrected = scanorama.correct_scanpy(split, return_dimred=True)
try:
    corrected = merge_adata(
        *corrected, batch_key=batch, batch_categories=categories, index_unique=None)
    corrected = corrected[adata.obs.index]
    adata.obsm["X_scanorama"] = corrected.obsm["X_scanorama"]
    pd.DataFrame(adata.obsm['X_scanorama'], index = adata.obs.index).to_csv(f'csv/{save_key}_Scanorama.csv')
    
except:
    
    with open(f'{save_key}_scanorama.pickle', 'wb') as handle:
        pickle.dump(corrected, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)

sc.settings.verbosity =0

import harmonypy as hm
import scanpy.external as sce
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scvi


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

adata.write(f'objects/{save_key}.h5ad')

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


sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor = 'seurat_v3',batch_key = batch_key)


scvi.model.SCVI.setup_anndata(adata,layer='counts', batch_key='Technology',categorical_covariate_keys=['Suspension_type','Donor_id'])

scvi_model = scvi.model.SCVI(adata,n_hidden=256, n_latent=30, n_layers=2,)

scvi_model.view_anndata_setup()

scvi_model.train(max_epochs=5000, use_gpu=True, early_stopping=True)

# adata.write('objects/Integration_scVI.h5ad', compression='gzip')

scvi_model.save(f'models/{save_key}', overwrite=True)

adata.obsm["X_scVI"] = scvi_model.get_latent_representation(adata)
pd.DataFrame(adata.obsm['X_scVI'], index = adata.obs.index).to_csv(f'csv/{save_key}_scVI.csv')

# sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_scVI')                         
# sc.tl.umap(adata, spread=1.3, min_dist=0.3)


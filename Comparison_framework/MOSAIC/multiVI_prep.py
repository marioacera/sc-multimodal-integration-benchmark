
#!/usr/bin/env python
# coding: utf-8
 
# In[1]:

# # LOAD DATA TO INTEGRATE

# In[8]:


#import gdown
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
#import seaborn as sns
import numpy as np
import anndata as an
# from scvi.external import CellAssign
import snapatac2 as snap



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (17,17)

import pickle
sc.settings.verbosity =0

current_directory = os.getcwd()
print("The script is running in:", current_directory)


atac = sc.read('../../HORIZONTAL_Integration/ATAC/objects/snapatac_FINAL.h5ad')

adata = sc.read('../../DIAGONAL_Integration/objects/RNA_STWG_final.h5ad')

adata.obs.index = [i.replace('-snRNA','-snMulti') for i in adata.obs.index]

adata.X = adata.layers['counts'].copy()

adata.obs['Technology'] = adata.obs['batch'].tolist()

adata.obs['Technology']  = [i.replace('snRNA','snMulti') for i in adata.obs['Technology'] ]

adata.var['modality'] = 'Gene Expression'

sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3", batch_key ='batch')


snap.pp.select_features(atac, n_features=200000)

atac.var['highly_variable'] = atac.var['selected']

#atac.X = atac.layers['counts'].copy()

atac.var['modality'] = 'Peaks'

atac.obs['Technology'] = 'snMulti'

atac.obs['batch'] = 'snATAC'

atac.obs.index = [i+'-snMulti' for i in atac.obs.index]


adata = adata[:,adata.var['highly_variable'] ].copy()
atac = atac[:,atac.var['highly_variable'] ].copy()

atac_feat = atac.var.index.tolist()
rna_feat = adata.var.index.tolist()

paired = np.intersect1d(adata.obs.index, atac.obs.index)

adata_mvi = []

adata_mvi.append(adata[~adata.obs.index.isin(paired)].copy())

adata = adata[adata.obs.index.isin(paired)].copy()

del atac.varm

adata

obs = adata.obs.copy()

adata = an.concat([adata, atac[atac.obs.index.isin(paired)]], axis = 1)

adata.obs = obs

adata.obs['batch'] = 'snMulti'

adata_mvi.append(adata)

del adata

atac = atac[~atac.obs.index.isin(paired)].copy()

adata_mvi.append(atac)

del atac

with open('../objects/multiVI_adatas.pkl', 'wb') as file:
    pickle.dump(adata_mvi, file)



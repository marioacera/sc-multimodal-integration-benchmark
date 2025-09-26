#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.remove('/mnt/beegfs/macera/.local/lib/python3.11/site-packages')

import pickle
import scvi
from scvi.data import organize_multiome_anndatas
from scvi.model import MULTIVI


sys.path.append('/mnt/beegfs/macera/.conda/envs/scanpy/lib/python3.9/site-packages')
sys.path.append('/mnt/beegfs/macera/.conda/envs/scanpy/lib/python3.9')
sys.path.append('/mnt/beegfs/macera/.conda/envs/scanpy/lib/python3.9/lib-dynload')
sys.path.append('/mnt/beegfs/macera/.conda/envs/scanpy/lib/python39.zip')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import sparse
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)
import seaborn as sns
from scipy import sparse

sc.settings.verbosity =0



import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
#import scvi


def compute_latents(adata, mod, n_hvg = 4000, subset='', batch = 'sample'):
    if subset != '':        
        save_key = mod 
    else:
        save_key = mod 
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor = 'cell_ranger')
    
    adata = adata[:, adata.var.highly_variable].copy()

    scvi.model.SCVI.setup_anndata(adata,layer='counts', batch_key=batch)

    scvi_model = scvi.model.SCVI(adata,n_hidden=256, n_latent=30, n_layers=2)

    scvi_model.view_anndata_setup()

    scvi_model.train(max_epochs=500, use_gpu=True, early_stopping=False)

    # adata.write('objects/Integration_scVI.h5ad', compression='gzip')

    scvi_model.save(f'models/HORIZONTAL_{save_key}', overwrite=True)

    adata.obsm["X_scVI"] = scvi_model.get_latent_representation(adata)
    pd.DataFrame(adata.obsm['X_scVI'], index = adata.obs.index).to_csv(f'csv/{save_key}_scVI.csv')


for mod in ['snRNA','scRNA','scRNA5p']:
    save_key = mod
    adata = sc.read(f'../objects/{mod}_raw.h5ad', compression='gzip')
    adata.X = adata.layers['counts'].copy()
    compute_latents(adata, mod,  n_hvg = 5000)

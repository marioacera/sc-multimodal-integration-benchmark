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
import argparse

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


def compute_latents(adata, save_key, n_hvg, batch):

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    try:

        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            flavor='seurat_v3', 
            layer='counts', batch_key=batch
        )
    except:
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_hvg, 
            layer='counts'
        )
    adata = adata[:, adata.var.highly_variable].copy()


    scvi.model.SCVI.setup_anndata(
        adata,
        layer='counts',
        batch_key=batch, categorical_covariate_keys=['suspension_type','assay']
    )

    model = scvi.model.SCVI(
        adata,
        n_hidden=256,
        n_latent=30,
        n_layers=2
    )
    model.train(
        max_epochs=500,
#        use_gpu=True,
        early_stopping=False
    )

    model.save(f'models/{save_key}', overwrite=True)
    latents = model.get_latent_representation(adata)
    df = pd.DataFrame(latents, index=adata.obs_names)
    df.to_csv(f'objects/{save_key}_scVI.csv')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',  default='all', help='Dataset mod: muto|wilson')
    p.add_argument('--batch', default='donor_id')
    args = p.parse_args()

    # load
    adata = sc.read(
        f'/mnt/beegfs/macera/CZI/Downstream/REVIEWS/scripts/RNA/reck/objects/rna.h5ad',
        compression='gzip'
    )
    adata = adata.raw.to_adata()
    adata.layers['counts'] = adata.X.copy()

    # subset if needed
    if args.dataset.lower() != 'all':
        adata = adata[adata.obs['dataset'] == args.pop].copy()
    else:
        print('all')
    n_hvg=5000
    save_key = f"{args.dataset}"


    compute_latents(adata, save_key, n_hvg, args.batch)

if __name__ == '__main__':
    main()


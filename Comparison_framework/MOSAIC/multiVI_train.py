
# # LOAD DATA TO INTEGRATE

import sys
import pickle
import scvi
from scvi.data import organize_multiome_anndatas
from scvi.model import MULTIVI
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as an
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (17,17)
import scipy
import pickle
print(pickle.__file__)
sc.settings.verbosity =0

current_directory = os.getcwd()
print("The script is running in:", current_directory)

with open('../objects/multiVI_adatas.pkl', 'rb') as file:
    adata_mvi = pickle.load(file)

atac_feat = adata_mvi[2].var.index.tolist()
rna_feat = adata_mvi[0].var.index.tolist()

adata_mvi= organize_multiome_anndatas(adata_mvi[1], rna_anndata = adata_mvi[0], atac_anndata = adata_mvi[2],)

adata_mvi.var.loc[rna_feat,'modality'] = 'Gene Expression'
adata_mvi.var.loc[atac_feat,'modality'] = 'Peaks'

adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()

adata_mvi.obs['batch'] = adata_mvi.obs['batch'].astype('category')
adata_mvi.obs['modality'] = adata_mvi.obs['modality'].astype('category')
adata_mvi.obs['Technology'] = adata_mvi.obs['Technology'].astype('category')

MULTIVI.setup_anndata(adata_mvi, batch_key="modality", categorical_covariate_keys=['Technology','sample'])

mvi = MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
    modality_weights="universal"
)

mvi.view_anndata_setup()


mvi.train(max_epochs = 1000, early_stopping=True)


mvi.save("model_multiVI", overwrite=True)



mvi = scvi.model.MULTIVI.load("model_multiVI", adata=adata_mvi)


adata_mvi.obsm["MultiVI"] = mvi.get_latent_representation()
sc.pp.neighbors(adata_mvi, use_rep="MultiVI", n_neighbors=50)
sc.tl.umap(adata_mvi, spread = 1, min_dist=0.5)

adata_mvi.write('../objects/multiVI.h5ad', compression='gzip')




import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import torch
import pandas as pd

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sc.set_figure_params(figsize=(4, 4), frameon=False)

adata = sc.read('objects/atac.h5ad')

save_key='ATAC'

min_cells = int(adata.shape[0] * 0.01)

sc.pp.filter_genes(adata, min_cells=min_cells)

scvi.model.PEAKVI.setup_anndata(adata, batch_key='sample')

model = scvi.model.PEAKVI(adata, n_hidden = 512, n_latent = 30)

model.train(max_epochs=1000, early_stopping=True)

model_dir = "models/peakVI_deep"

model.save(model_dir, overwrite=True)

#model = scvi.model.PEAKVI.load(model_dir, adata=adata)

PEAKVI_LATENT_KEY = "X_peakVI"

latent = model.get_latent_representation()

adata.obsm[PEAKVI_LATENT_KEY] = latent

pd.DataFrame(adata.obsm[PEAKVI_LATENT_KEY], index = adata.obs.index).to_csv(f'csv/{save_key}_PeakVI.csv')

adata.write('objects/{save_key}_PeakVI.h5ad', compression='gzip')

PEAKVI_CLUSTERS_KEY = "clusters_peakvi"

sc.pp.neighbors(adata, use_rep=PEAKVI_LATENT_KEY)
# compute the umap
sc.tl.umap(adata, min_dist=0.2)
# cluster the space (we use a lower resolution to get fewer clusters than the default)
sc.tl.leiden(adata, key_added=PEAKVI_CLUSTERS_KEY, resolution=0.2)

adata.write(f'objects/{save_key}_PeakVI.h5ad', compression='gzip')



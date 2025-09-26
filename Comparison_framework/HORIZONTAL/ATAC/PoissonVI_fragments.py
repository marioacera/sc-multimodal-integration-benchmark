
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import torch
import pandas as pd
import numpy as np
import snapatac2 as snap

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

save_key = 'ATAC'

sc.set_figure_params(figsize=(4, 4), frameon=False)

torch.set_float32_matmul_precision("high")

save_dir = tempfile.TemporaryDirectory()


adata = sc.read('objects/atac.h5ad')

print('Read counts:\n')

print('== 1:',(adata.X == 1).sum())
print('== 2:',(adata.X == 2).sum())

scvi.data.reads_to_fragments(adata)

print('== 1:',(adata.layers["fragments"] == 1).sum())
print('== 2:',(adata.layers["fragments"] == 2).sum())

adata.write('objects/ATAC_Frag_Approx.h5ad', compression='gzip')
print("# regions before filtering:", adata.shape[-1])

# compute the threshold: 1% of the cells
min_cells = int(adata.shape[0] * 0.01)
# in-place filtering of regions
sc.pp.filter_genes(adata, min_cells=min_cells)

print("# regions after filtering:", adata.shape[-1])

scvi.external.POISSONVI.setup_anndata(adata, layer="fragments",batch_key='sample')

model = scvi.external.POISSONVI(adata,n_hidden=512, n_latent=32)
model.train(max_epochs=1000,early_stopping=True)

LATENT_KEY = "X_poissonVI"

latent = model.get_latent_representation()
adata.obsm[LATENT_KEY] = latent
latent.shape

pd.DataFrame(adata.obsm[LATENT_KEY], index = adata.obs.index).to_csv(f'csv/{save_key}_Poisson_fragments.csv')


model_dir = "models/poissonVI_fragments"
model.save(model_dir, overwrite=True)


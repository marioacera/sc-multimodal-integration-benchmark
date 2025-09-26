
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI
import scib_metrics
from scipy import sparse
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)
import seaborn as sns
from scipy import sparse

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

import time


#latents = ['snRNA_scVI','snRNA_Harmony','snRNA_Unintegrated','snRNA_Scanorama','Spectral','Spectral_Harmony','Spectral_mnn','PeakVI', 'PoissonVI','ATAC_LSI_Seurat_WNN', 'RNA_SCT_PCA_Seurat_WNN']
latents = ['Spectral','Spectral_Harmony','Spectral_MNN','PeakVI', 'PoissonVI','PoissonVI_fragments','LSI_Harmony']
mod = '10xMultiome'
batch = 'sample'

adata = sc.read(f'objects/atac_multi.h5ad')
adata.obsm['Spectral_MNN'] = adata.obsm['Spectral_mnn'].copy()
adata.obsm['LSI_Harmony'] = adata.obsm['ATAC_LSI_Seurat_WNN'].copy()

biocons = BioConservation(isolated_labels=False)
batchcors = BatchCorrection(silhouette_batch=True, ilisi_knn=True, kbet_per_label=True, graph_connectivity=True, pcr_comparison=False)
adata = adata.copy()

start = time.time()

bm = Benchmarker(
    adata,
    batch_key=batch,
    label_key="Deepscore_HCA_l3_Clean",                                              
    embedding_obsm_keys=latents,
    bio_conservation_metrics = biocons,
    batch_correction_metrics = batchcors,
    n_jobs=-1,
)
bm.benchmark()
end = time.time()
print(f"Time: {int((end - start) / 60)} min {int((end - start) % 60)} sec")


bm.plot_results_table(min_max_scale=False, show=False)
plt.savefig(f'figures/{mod}_Benchmark_l2_ATAC_Bio.png',dpi=300)
plt.savefig(f'figures/{mod}_Benchmark_l2_ATAC_Bio.pdf',dpi=300)

bm.plot_results_table(show=False)
plt.savefig(f'figures/{mod}_Benchmark_l2_scaled_ATAC_Bio.png',dpi=300)
plt.savefig(f'figures/{mod}_Benchmark_l2_scaled_ATAC_Bio.pdf',dpi=300)

df = bm.get_results(min_max_scale=False)
df.to_csv(f'csv/{mod}_Benchmark_l2_ATAC.csv')

#mod = 'ATAC'
start = time.time()

bm = Benchmarker(
    adata,
    batch_key=batch,
    label_key="Deepscore_HCA_l1_Clean",                                              
    embedding_obsm_keys=latents,
    bio_conservation_metrics = biocons,
    batch_correction_metrics = batchcors,
    n_jobs=-1,
)
bm.benchmark()
end = time.time()
print(f"Time: {int((end - start) / 60)} min {int((end - start) % 60)} sec")


bm.plot_results_table(min_max_scale=False, show=False)
plt.savefig(f'figures/{mod}_Benchmark_l1_ATAC_Bio.png',dpi=300)
plt.savefig(f'figures/{mod}_Benchmark_l1_ATAC_Bio.pdf',dpi=300)

bm.plot_results_table(show=False)
plt.savefig(f'figures/{mod}_Benchmark_l1_scaled_ATAC_Bio.png',dpi=300)

df = bm.get_results(min_max_scale=False)
df.to_csv(f'csv/{mod}_Benchmark_l1_ATAC.csv')



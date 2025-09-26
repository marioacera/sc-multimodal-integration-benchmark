
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

from scipy import sparse
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (12,12)
import seaborn as sns
from scipy import sparse


from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import time



for mod in ['snRNA','scRNA5p','scRNA']:
    save_key = mod


    adata = sc.read(f'../objects/{save_key}_raw.h5ad')
    latents = []
    for i in ['Harmony','Unintegrated','Scanorama','scVI']:
        l = pd.read_csv(f'csv/{mod}_{i}.csv', index_col=0)
        l = l.loc[adata.obs.index]
        adata.obsm[i] = l.values
        latents.append(i)

    # In[11]:
    ds = pd.read_csv(f'csv/deepscore/Deepscore_HCA_l1_{mod}.csv',index_col=0)
    adata.obs['Deepscore_HCA_l1'] = ds['Deepscore_HCA']
    adata.obs['Deepscore_HCA_l3'] = adata.obs['Deepscore_HCA_l1'].tolist()
    for ct in ['IMM', 'DCT', 'PT', 'CNT', 'TAL', 'IC', 'EC', 'FIB', 'PC', 'VSM/P']:
        c = ct.replace('/','_')
        ds = pd.read_csv(f'csv/deepscore/Deepscore_HCA_{mod}_{c}_l3_no_deg.csv',index_col=0)
        batch_mask = adata.obs['Deepscore_HCA_l1'] == ct
        adata.obs.loc[adata[batch_mask].obs.index, ['Deepscore_HCA_l3']] = ds['Deepscore_HCA']
        
    biocons = BioConservation(isolated_labels=False)
    batchcors = BatchCorrection(silhouette_batch=True, ilisi_knn=True, kbet_per_label=True, graph_connectivity=True, pcr_comparison=False)

    start = time.time()
    bm = Benchmarker(
        adata,
        batch_key="sample",
        label_key="Deepscore_HCA_l1",                                              
        embedding_obsm_keys=latents,
        bio_conservation_metrics = biocons,
        batch_correction_metrics = batchcors,
        n_jobs=-1,
    )
    bm.benchmark()
    
    end = time.time()
    print(f"Time: {int((end - start) / 60)} min {int((end - start) % 60)} sec")


    bm.plot_results_table(min_max_scale=False, show=False)
    plt.savefig(f'figures/{mod}_Benchmark_l1_NEW.png',dpi=600)


    bm.plot_results_table(show=False)
    plt.savefig(f'figures/{mod}_Benchmark_l1_scaled_NEW.png',dpi=600)

    df = bm.get_results(min_max_scale=False)
    df.to_csv(f'csv/{mod}_Benchmark_l1_NEW.csv')
    
    
    start = time.time()
    bm = Benchmarker(
        adata,
        batch_key="sample",
        label_key="Deepscore_HCA_l3",                                              
        embedding_obsm_keys=latents,
        bio_conservation_metrics = biocons,
        batch_correction_metrics = batchcors,
        n_jobs=-1,
    )
    bm.benchmark()
    end = time.time()
    print(f"Time: {int((end - start) / 60)} min {int((end - start) % 60)} sec")


    bm.plot_results_table(min_max_scale=False, show=False)
    plt.savefig(f'figures/{mod}_Benchmark_l3_NEW.png',dpi=600)


    bm.plot_results_table(show=False)
    plt.savefig(f'figures/{mod}_Benchmark_l3_scaled_NEW.png',dpi=600)

    df = bm.get_results(min_max_scale=False)
    df.to_csv(f'csv/{mod}_Benchmark_l3_NEW.csv')


import umap
import os
import scanpy as sc
import anndata
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

sc.settings.verbosity =0


import harmonypy as hm
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import anndata

def downsample_adata(
    adata,
    cell_type_var,
    qc_vars,
    target_cells,
    max_attempts=10,
    quantiles_to_mantain = [0.2, 0.4, 0.6, 0.8],
    quantile_margin=0.05
):
    """
    Downsample an AnnData object while maintaining cell type proportions
    and the quantile distribution of QC variables using stratified sampling.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - cell_type_var: str, the column name in adata.obs corresponding to cell type annotations.
    - qc_vars: list of str, column names in adata.obs to use as quality control variables.
    - target_cells: int, the number of cells to downsample to.
    - max_attempts: int, maximum number of attempts to achieve desired quantile matching.
    - quantile_margin: float, allowable deviation in quantiles for QC metrics.

    Returns:
    - downsampled_adata: AnnData object with downsampled cells.
    """
   
    aggregated_data = adata 

    # Step 1: Calculate the cell type proportions in the aggregated data
    cell_type_counts = aggregated_data.obs[cell_type_var].value_counts()
    total_cells = len(aggregated_data)
    cell_type_proportions = cell_type_counts / total_cells

    # Step 2: Determine the number of cells per cell type for downsampling
    cell_type_target_counts = (cell_type_proportions * target_cells).astype(int)
    
    # Step 3: Ignore cell types with fewer than 50 target cells
    cell_type_target_counts = cell_type_target_counts[cell_type_target_counts >= 50]
    print(cell_type_target_counts, np.sum(cell_type_target_counts))
    downsampled_indices = []

    # Step 4: Perform stratified sampling using StratifiedShuffleSplit
    for attempt in range(max_attempts):
        temp_indices = []
        sss = StratifiedShuffleSplit(n_splits=1, train_size=target_cells, random_state=np.random.randint(500))

        stratify_labels = aggregated_data.obs[cell_type_var].apply(
            lambda x: x if x in cell_type_target_counts.index else 'Other'
        )
        
        # Fit and transform
        for train_idx, _ in sss.split(aggregated_data.obs.index, stratify_labels):
            sampled_indices = aggregated_data.obs.iloc[train_idx].index
            temp_indices.extend(sampled_indices)

        # Step 5: Check if the downsampled data matches the original quantile distributions
        temp_downsampled_adata = adata[temp_indices].copy()
        quantile_check = True

        for qc_var in qc_vars:
            original_quantiles = aggregated_data.obs[qc_var].quantile(quantiles_to_mantain)
            downsampled_quantiles = temp_downsampled_adata.obs[qc_var].quantile(quantiles_to_mantain)

            # Check if the quantiles of the downsampled data are within ±5% of the original quantiles
            if any(abs((downsampled_quantiles - original_quantiles) / original_quantiles) > quantile_margin):
                quantile_check = False
                break

        if quantile_check:
            downsampled_indices = temp_indices
            break
        elif attempt == max_attempts - 1:
            print("Warning: Could not maintain quantile distribution within ±5% margin after 10 attempts.")
    
    # Step 6: Create the downsampled AnnData object
    downsampled_adata = adata[adata.obs_names.isin(downsampled_indices)].copy()
    
    return downsampled_adata


for i in range(10):
    for batch in ['scRNA','snRNA','scRNA5p']:
        for n_cells in [7500]:#,10000,12500,15000,17500,20000]:

                adata = sc.read(f'objects/byTech/{batch}.h5ad')
                adata = adata[~adata.obs['Deepscore_HCA_l1_Clean'].isin(['NEU'])].copy()
                if adata[adata.obs['batch'].isin([batch])].shape[0] > n_cells: 
                    adata = downsample_adata(
                        adata=adata,
                        cell_type_var='Deepscore_HCA_l1_Clean',
                        qc_vars=['total_counts', 'n_genes_by_counts'],
                        target_cells=n_cells,
                    )
                    adata.write(f'objects/subsample/{batch}_{n_cells}cells_{i}.h5ad', compression = 'gzip')
                    for counts in [20000,15000,10000,7500,5000,3000]:
                        if os.path.exists(f"DEG/downsampling/markers_{batch}_{n_cells}cells_{counts}counts_run{i}_.csv"):
                            continue
                        else:
                            adata.X = adata.layers['counts'].copy()
                            adata = sc.pp.downsample_counts(adata, counts_per_cell=counts, copy=True)
                            sc.pp.calculate_qc_metrics(adata, inplace=True)
                            print(adata, adata.obs['total_counts'].mean())
                            # markers_dict = find_markers(adata, save_key=f'{batch}_{n_cells}cells_{counts}counts')
                            sc.pp.filter_genes(adata, min_cells=1)
                            sc.pp.normalize_total(adata, target_sum=10e4)
                            sc.pp.log1p(adata)
                            # Compute differential expression
                            sc.tl.rank_genes_groups(adata, groupby='Deepscore_HCA_l1_Clean', method='wilcoxon', use_raw=False)
                            
                            # Collect results
                            markers_df = sc.get.rank_genes_groups_df(adata, group=None)   
                            print(markers_df) 
                            # Optionally save results to CSV
                            markers_df.to_csv(f"DEG/downsampling/markers_{batch}_{n_cells}cells_{counts}counts_run{i}_.csv")
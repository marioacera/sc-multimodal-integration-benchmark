import sys
import snapatac2 as snap
import scanpy as sc
import pickle
import pandas as pd
import numpy as np


if __name__ == "__main__":
    tech = sys.argv[1] 

    adata = sc.read(f'../objects/{tech}_raw.h5ad')
    
    adata.X = adata.layers['counts'].copy()
    
    sc.pp.normalize_total(adata, target_sum=10e4)
    sc.pp.log1p(adata)

    adata.layers['logcounts'] = adata.X.copy()

    metadata = pd.read_csv('../csv/CLUSTERS_METADATA.csv', index_col=0)

    adata.obs[metadata.columns] = metadata
    print(adata)
    print(adata.obs)
    print(adata.X.data)
    sc.tl.rank_genes_groups(adata, method='wilcoxon',groupby='Deepscore_HCA_l1_Clean', layer='logcounts', use_raw=False)

    adata.obs['cell_type'] =  adata.obs['Deepscore_HCA_l1_Clean']

    result = adata.uns["rank_genes_groups"]

    groups = result["names"].dtype.names

    for group in groups:

        df = pd.DataFrame(
            {
                key: result[key][group]
                for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']
            }
        )
        df.index = df['names'].tolist()
        df = df[df['logfoldchanges']>0]
        df = df[df['pvals']<0.01]
        df = df[['logfoldchanges', 'pvals', 'pvals_adj']]
        df.columns = ['log2(fold_change)','p-value','adjusted p-value']
        df['mean_Counts'] = np.ravel(np.mean(adata[adata.obs['cell_type'].isin([group]), df.index.to_numpy()].X, axis = 0))
        df['sum_Counts'] = np.ravel(np.sum(adata[adata.obs['cell_type'].isin([group]), df.index.to_numpy()].X, axis = 0))
        df['pct_cells_IN_group'] = np.ravel(np.sum(adata[adata.obs['cell_type'].isin([group]), df.index.to_numpy()].X > 0, axis=0))/adata[adata.obs['cell_type'].isin([group])].shape[0]
        df['pct_cells_OUT_group'] = np.ravel(np.sum(adata[~adata.obs['cell_type'].isin([group]), df.index.to_numpy()].X > 0, axis=0))/adata[~adata.obs['cell_type'].isin([group])].shape[0]
        group1 = group.replace('/','_')

        df.to_csv(f'csv/l1/{tech}/DEG_{group1}.csv', index_label='Feature_name')






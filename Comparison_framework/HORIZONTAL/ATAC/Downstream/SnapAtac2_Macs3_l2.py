import snapatac2 as snap
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

snap.__version__

if __name__ == '__main__':
    data = snap.read('objects/snapatac_FINAL.h5ad', backed=None)
    print(data)

    metadata = pd.read_csv('../../CLUSTERS_METADATA.csv', index_col=0)
    metadata = metadata[metadata['batch'].isin(['snRNA'])]
    metadata.index = [i.replace('-snRNA','') for i in metadata.index]

    print(np.sum(data.obs.index.isin(metadata.index)))
    data.obs['Deepscore_HCA_l3_Clean'] = metadata['Deepscore_HCA_l3_Clean']

    clean_obs = data.obs.dropna(subset=['Deepscore_HCA_l3_Clean'])
    data = data[data.obs.index.isin(clean_obs.index)]

    data.obs['Deepscore_HCA_l3_Clean'] = [i.replace('/','_') for i in data.obs['Deepscore_HCA_l3_Clean']]
    data.obs['Deepscore_HCA_l3_Clean'] = data.obs['Deepscore_HCA_l3_Clean'].astype('category')



    #data = data[:,data.var['selected']].copy()
    print(data.obs.columns)
    #data.obs['test'] = 'test'
    snap.tl.macs3(data, groupby='Deepscore_HCA_l3_Clean', qvalue=0.05, n_jobs = 8)
    data.write('objects/MACS3_l2.h5ad', compression='gzip')
    merged_peaks = snap.tl.merge_peaks(data.uns['macs3'], chrom_sizes=snap.genome.hg38)

    peak_mat = snap.pp.make_peak_matrix(data, use_rep=merged_peaks['Peaks'])
    import pickle
    with open('macs_peak_l2.pickle', 'wb') as handle:
        pickle.dump(peak_mat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('merged_peaks_l2.pickle', 'wb') as handle:
        pickle.dump(merged_peaks, handle, protocol=pickle.HIGHEST_PROTOCOL)


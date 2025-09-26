import sys
import snapatac2 as snap
import scanpy as sc
import pickle
import pandas as pd
import numpy as np

import sys
import os

def main(param_set):
    # Your script logic here
    print(f"Running with parameter set: {param_set}")
def parse_index(index):
    chrom, positions = index.split(':')
    start, end = positions.split('-')
    return chrom, int(start), int(end)

def csv_to_bed(csv_file, bed_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, index_col=0)
    
    # Parse the index to extract chromosome, start, and end positions
    parsed_data = [parse_index(idx) for idx in df.index]
    bed_df = pd.DataFrame(parsed_data, columns=['chromosome', 'start', 'end'])
    
    # Save to BED format (tab-separated)
    bed_df.to_csv(bed_file, sep='\t', header=False, index=False)
    print(f"Converted {csv_file} to {bed_file}")





if __name__ == "__main__":

    group1 = sys.argv[1]  # Get the parameter from the command line
    main(group1)


    group1_ = group1.replace('_','/')
    with open("../macs_peak_l2.pickle", "rb") as file:
        adata = pickle.load(file)

    with open("../mergued_peaks_l2.pickle", "rb") as file:
        peaks = pickle.load(file)

    # marker_peaks = snap.tl.marker_regions(adata, groupby='Deepscore_HCA_l3_Clean', pvalue=0.01)

    adata.obs['cell_type'] =  adata.obs['Deepscore_HCA_l3_Clean']

    peaks = pd.DataFrame(peaks.to_numpy()[:,1:], index = peaks['Peaks'].to_numpy(), columns = peaks.columns[1:])
    print(list(adata.obs['cell_type'].unique()))

    bar_group_1 = adata.obs['cell_type'] == group1_
    bar_group_2 = ~(adata.obs['cell_type'] == group1_)

    peaks_selected = peaks[group1].to_numpy()



    # Suppress all outputs
    sys.stdout = open(os.devnull, 'w')

    diff_peaks = snap.tl.diff_test(
    adata,
    cell_group1=bar_group_1,
    cell_group2=bar_group_2,
    features=peaks_selected,
    direction="positive",
    min_log_fc=0, min_pct=0.01
    )

    sys.stdout = sys.__stdout__
    diff_peaks = pd.DataFrame(diff_peaks.to_numpy()[:,1:], index = diff_peaks['feature name'].to_numpy(), columns = diff_peaks.columns[1:])

    diff_peaks['mean_Counts'] = np.ravel(np.mean(adata[adata.obs['cell_type'].isin([group1]), diff_peaks.index.to_numpy()].X, axis = 0))
    diff_peaks['sum_Counts'] = np.ravel(np.sum(adata[adata.obs['cell_type'].isin([group1]), diff_peaks.index.to_numpy()].X, axis = 0))
    diff_peaks['pct_cells_IN_group'] = np.ravel(np.sum(adata[adata.obs['cell_type'].isin([group1]), diff_peaks.index.to_numpy()].X > 0, axis=0))/adata[adata.obs['cell_type'].isin([group1])].shape[0]
    diff_peaks['pct_cells_OUT_group'] = np.ravel(np.sum(adata[~adata.obs['cell_type'].isin([group1]), diff_peaks.index.to_numpy()].X > 0, axis=0))/adata[~adata.obs['cell_type'].isin([group1])].shape[0]


    diff_peaks.to_csv(f'csv/l2/DAR_{group1}.csv', index_label='Feature_name')
    
    # Example usage:
    csv_file = f'csv/l2/DAR_{group1}.csv'
    bed_file = f'csv/l2/bed/{group1}.bed'
    csv_to_bed(csv_file, bed_file)





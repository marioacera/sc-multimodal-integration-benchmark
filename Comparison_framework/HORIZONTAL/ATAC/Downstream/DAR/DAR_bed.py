import sys
import snapatac2 as snap
import scanpy as sc
import pickle
import pandas as pd
import numpy as np


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
group1 = sys.argv[1]  # Get the parameter from the command line
# Example usage:
csv_file = f'csv/l1/DAR_{group1}.csv'
bed_file = f'csv/l1/bed/{group1}.bed'
csv_to_bed(csv_file, bed_file)




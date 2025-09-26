
import anndata as ad
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams
import snapatac2 as snap

rna = sc.read('../objects/RNA_STWG_final.h5ad')

sc.pp.highly_variable_genes(rna, n_top_genes=5000, flavor="seurat_v3", batch_key ='batch')

scglue.data.get_gene_annotation(
    rna, gtf="../Homo_sapiens.GRCh38.98.gtf.gz",
    gtf_by="gene_name"
)
print(rna.var["chromStart"].isna().any())
print(rna.var["chromEnd"].isna().any())
print(rna.var["chrom"].isna().any())
print(rna)
rna = rna[:, rna.var.dropna(subset='chromEnd').index]

print(rna)
rna.var["chromStart"] = rna.var["chromStart"].astype(int)
rna.var["chromEnd"] = rna.var["chromEnd"].astype(int)

print(rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head())

atac = sc.read('../../HORIZONTAL_Integration/ATAC/objects/snapatac_FINAL.h5ad')

snap.pp.select_features(atac, n_features=200000)

split = atac.var_names.str.split(r"[:-]")
atac.var["chrom"] = split.map(lambda x: x[0])
atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

atac.var['highly_variable'] = atac.var['selected']

print(atac.var["chromStart"].isna().any())
print(atac.var["chromEnd"].isna().any())
print(atac.var["chrom"].isna().any())
print(atac.var.head())


guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
guidance

scglue.graph.check_graph(guidance, [rna, atac])

rna.write("../objects/GLUE_rna_pp.h5ad", compression="gzip")
atac.write("../objects/GLUE_atac_pp.h5ad", compression="gzip")
nx.write_graphml(guidance, "../objects/guidance.graphml.gz")


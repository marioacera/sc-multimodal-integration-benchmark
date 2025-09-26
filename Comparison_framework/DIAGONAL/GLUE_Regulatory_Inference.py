
import os
import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scglue
import seaborn as sns
from IPython import display
from matplotlib import rcParams
from networkx.algorithms.bipartite import biadjacency_matrix
from networkx.drawing.nx_agraph import graphviz_layout
import torch
device = torch.device("cpu") 


from scglue.utils import config

# Set CPU_ONLY to True
config.CPU_ONLY = True

rna = ad.read_h5ad("../objects/GLUE_rna_emb.h5ad")
atac  = ad.read_h5ad("../objects/GLUE_atac_emb.h5ad")
guidance_hvf = nx.read_graphml("guidance-hvf.graphml.gz")

print('objects loaded')
rna.var["name"] = rna.var_names
atac.var["name"] = atac.var_names

genes = rna.var.query("highly_variable").index
peaks = atac.var.query("highly_variable").index

features = pd.Index(np.concatenate([rna.var_names, atac.var_names]))
feature_embeddings = np.concatenate([rna.varm["X_glue"], atac.varm["X_glue"]])


skeleton = guidance_hvf.edge_subgraph(
    e for e, attr in dict(guidance_hvf.edges).items()
    if attr["type"] == "fwd"
).copy()

print('skelon created')

reginf = scglue.genomics.regulatory_inference(
    features, feature_embeddings,
    skeleton=skeleton, random_state=0
)

print('reginf created', reginf)
gene2peak = reginf.edge_subgraph(
    e for e, attr in dict(reginf.edges).items()
    if attr["qval"] < 0.05
)
print('graph subseted', dict(reginf.edges).items())

import pickle
pickle.dump(
    list(dict(reginf.edges).items()),
    open(os.path.join('../objects', "reg_inf_dic.pkl"), "wb")
)


scglue.genomics.Bed(atac.var).write_bed("../objects/peaks.bed", ncols=3)
scglue.genomics.write_links(
    gene2peak,
    scglue.genomics.Bed(rna.var).strand_specific_start_site(),
    scglue.genomics.Bed(atac.var),
    "../objects/gene2peak.links", keep_attrs=["score"]
)


















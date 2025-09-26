from itertools import chain
import anndata as ad
import networkx as nx
import pandas as pd
import scanpy as sc
import scglue
import seaborn as sns
from matplotlib import rcParams
import snapatac2 as snap
import matplotlib.pyplot as plt
import torch
device = torch.device("cpu") 


from scglue.utils import config

# Set CPU_ONLY to True
config.CPU_ONLY = True

model_name = 'glue'

rna = ad.read_h5ad("../objects/GLUE_rna_pp.h5ad")
atac = ad.read_h5ad("../objects/GLUE_atac_pp.h5ad")
guidance = nx.read_graphml("../objects/guidance.graphml.gz")

print(atac, rna)

rna.obs['batch_sample'] = rna.obs[['batch','sample']].astype(str).agg('_'.join, axis=1)

scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="scVI", use_batch='batch_sample'
)

scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,use_layer="counts",
    use_rep="Spectral_mnn", use_batch = 'sample'
)

guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

glue = scglue.models.load_model(f"{model_name}.dill")

dx = scglue.models.integration_consistency(
    glue, {"rna": rna, "atac": atac}, guidance_hvf
)

_ = sns.lineplot(x="n_meta", y="consistency", data=dx).axhline(y=0.05, c="darkred", ls="--")
plt.savefig('glue_pretrained_consistency.png', dpi=200)

rna.obsm["X_glue"] = glue.encode_data("rna", rna)
atac.obsm["X_glue"] = glue.encode_data("atac", atac)

combined = ad.concat([rna, atac])
sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
sc.tl.umap(combined)

combined.write("../objects/GLUE_comb_emb_FINAL.h5ad", compression="gzip")
rna.write("../objects/GLUE_rna_emb_FINAL.h5ad", compression="gzip")
atac.write("../objects/GLUE_atac_emb_FINAL.h5ad", compression="gzip")
nx.write_graphml(guidance_hvf, "guidance_hvf_FINAL.graphml.gz")

feature_embeddings = glue.encode_graph(guidance_hvf)
feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)

rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()

rna.write("../objects/GLUE_rna_emb.h5ad", compression="gzip")
atac.write("../objects/GLUE_atac_emb.h5ad", compression="gzip")
nx.write_graphml(guidance_hvf, "guidance-hvf.graphml.gz")


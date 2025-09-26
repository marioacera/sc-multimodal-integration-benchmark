library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(patchwork)
library(harmony)
library(dplyr)
library(tibble)



library(Seurat)
library(dplyr)
library(tibble)
library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(patchwork)

setwd('~/Documents/CZI/')


counts <- Read10X_h5(filename = "MULTIOME/data/AGG/outs/filtered_feature_bc_matrix.h5")


annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevels(annotation) <- paste0('chr', seqlevels(annotation))

# create a Seurat object containing the RNA adata
data <- CreateSeuratObject(
  counts = counts$`Gene Expression`,
  assay = "RNA"
)
data[["ATAC"]] <- CreateChromatinAssay(
  counts = counts$Peaks,
  sep = c(":", "-"),
  fragments = 'MULTIOME/data/AGG/outs/atac_fragments.tsv.gz',
  #annotation = annotation
)

metadata <- read.csv(
  file = "MANUSCRIPT_PREP/VERTICAL_INTEGRATION/multiome_cells.csv",
  # header = TRUE,
  col.names = c('index')
)


multiome_path <- "MANUSCRIPT_PREP/VERTICAL_INTEGRATION/objects/multiome_nofilter.rds"

# Save the object as an .rds file
saveRDS(data, file = multiome_path)
data <-subset(data, cells = metadata$index)


multiome_path <- "MANUSCRIPT_PREP/VERTICAL_INTEGRATION/objects/multiome.rds"

# Save the object as an .rds file
saveRDS(data, file = multiome_path)



setwd('/mnt/beegfs/macera/CZI/Downstream/VERTICAL_Integration/')


# Load the Seurat object
multiome_path <- "objects/multiome.rds"


data <- readRDS(multiome_path)

# Load metadata and add it to the Seurat object
metadata <- read.csv(file ='csv/multiome_RNA_metadata.csv')
rownames(metadata) <- metadata$X

data <- AddMetaData(object = data, metadata = metadata)

data$barcode <- colnames(data)

# Load and process RNA embeddings
embeddings_rna <- read.csv("csv/snRNA_scVI.csv")
print(head(embeddings_rna))

embeddings_rna$barcode <- gsub("-snRNA", "", embeddings_rna$barcode)

print(head(embeddings_rna))

rownames(embeddings_rna) <- embeddings_rna$barcode

colnames(embeddings_rna)[2:31] <- paste0("RNA_", 1:30)

print(embeddings_rna)
data <- AddMetaData(object = data, metadata = embeddings_rna)
rna_columns <- grep("RNA_", names(data@meta.data), value = TRUE)
data[['RNA_emb']] <- CreateDimReducObject(embeddings = as.matrix(data@meta.data[, grep("RNA_", names(data@meta.data))]), key = "RNA_emb")

# Load and process ATAC embeddings
embeddings_atac <- read.csv("csv/Snap_Spectral_mnn.csv")
rownames(embeddings_atac) <- embeddings_atac$barcode

colnames(embeddings_atac)[2:31] <- paste0("ATAC_", 1:30)

data <- AddMetaData(object = data, metadata = embeddings_atac)
atac_columns <- grep("ATAC_", names(data@meta.data), value = TRUE)
data[['ATAC_emb']] <- CreateDimReducObject(embeddings = as.matrix(data@meta.data[, grep("ATAC_", names(data@meta.data))]), key = "ATAC_emb")

# Use these embeddings in FindMultiModalNeighbors
data <- FindMultiModalNeighbors(
  object = data,
  reduction.list = c("RNA_emb", "ATAC_emb"),
  dims.list = list(1:ncol(data[["RNA_emb"]]), 1:ncol(data[["ATAC_emb"]])),
  k.nn = 50,
  modality.weight.name = list("RNA.weight","ATAC-weight"),
  verbose = TRUE
)

multiome_path <- "objects/Seurat_WNN_Custom.rds"

# Save the object as an .rds file
saveRDS(data, file = multiome_path)

# build a joint UMAP visualization
data <- RunUMAP(
  object = data,
  k.nn = 50,
  nn.name = "weighted.nn",
  assay = "RNA",
  verbose = TRUE
)

saveRDS(data, file = multiome_path)

data <- readRDS(multiome_path)

write.table(data@active.ident,file = "csv/Seurat_WNN_obs_names.csv",sep=";",dec=",",row.names = T)

write.table(data@reductions$umap@cell.embeddings,file = "csv/Seurat_WNN_UMAP.csv",sep=";",dec=",",row.names = T)

write.table(data@neighbors$weighted.nn@nn.idx,file = "csv/Seurat_WNN_idx.csv",sep=";",dec=",",row.names = T)

write.table(data@neighbors$weighted.nn@nn.dist,file = "csv/Seurat_WNN_dist.csv",sep=";",dec=",",row.names = T)




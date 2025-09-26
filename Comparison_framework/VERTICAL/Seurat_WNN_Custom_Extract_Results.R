library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(patchwork)
library(harmony)
library(dplyr)
library(tibble)


setwd('/mnt/beegfs/macera/CZI/Downstream/VERTICAL_Integration/')

library(Seurat)
library(dplyr)
library(tibble)

# Load the Seurat object
multiome_path <- "objects/Seurat_WNN_Custom.rds"

pbmc <- readRDS(multiome_path)

write.table(pbmc@active.ident,file = "csv/Seurat_WNN_obs_names.csv",sep=";",dec=",",row.names = T)

write.table(pbmc[[,c("RNA.weight","ATAC-weight")]],file = "csv/Seurat_WNN_MOD_WEIGHTS.csv",sep=";",dec=",",row.names = T)

write.table(pbmc@reductions$umap@cell.embeddings,file = "csv/Seurat_WNN_UMAP.csv",sep=";",dec=",",row.names = T)

write.table(pbmc@neighbors$weighted.nn@nn.idx,file = "csv/Seurat_WNN_idx.csv",sep=";",dec=",",row.names = T)

write.table(pbmc@neighbors$weighted.nn@nn.dist,file = "csv/Seurat_WNN_dist.csv",sep=";",dec=",",row.names = T)

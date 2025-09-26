library(Signac)
library(Seurat)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(patchwork)
#library(harmony)

#setwd('/mnt/beegfs/macera/CZI/Downstream/VERTICAL_Integration/')

setwd('~/Documents/CZI/MANUSCRIPT_PREP/VERTICAL_INTEGRATION/')

multiome_path <- "objects/multiome.rds"


data <- readRDS(multiome_path)

metadata <- read.csv(
file ='multiome_RNA_metadata.csv',
)

rownames(metadata) <- metadata$X

data <- AddMetaData(
  object = data,
  metadata = metadata)

#macs2.path <- c("~/miniconda3/bin/macs2/")
#peaks <- CallPeaks(data, macs2.path = macs2.path)

DefaultAssay(data) <- "RNA"
FindVariableFeatures(data)
data <- SCTransform(data)
data <- RunPCA(data)
harmonized_seurat <- RunHarmony(data, 
                                group.by.vars = c("sample"), 
                                reduction = "pca", assay.use = "RNA", reduction.save = "pca_harmony")




DefaultAssay(data) <- "ATAC"
data <- FindTopFeatures(data, min.cutoff = 5)
data <- RunTFIDF(data)
data <- RunSVD(data)

harmonized_seurat <- RunHarmony(data, 
                                group.by.vars = c("sample"), 
                                reduction = "lsi", assay.use = "RNA", reduction.save = "lsi_harmony")


data <- FindMultiModalNeighbors(
  object = data,
  reduction.list = list("pca_harmony", "lsi_harmony"), 
  dims.list = list(1:50, 2:40),
  modality.weight.name = "RNA.weight",
  verbose = TRUE
)

# build a joint UMAP visualization
data <- RunUMAP(
  object = data,
  nn.name = "weighted.nn",
  assay = "RNA",
  verbose = TRUE
)

multiome_path <- "objects/Seurat_WNN.rds"

# Save the object as an .rds file
saveRDS(data, file = multiome_path)


multiome_path <- "objects/Seurat_WNN.rds"
setwd('~/Documents/CZI/MANUSCRIPT_PREP/VERTICAL_INTEGRATION/')

data <- readRDS(multiome_path)

write.table(data@active.ident,file = "csv/Seurat_WNN_obs_names.csv",sep=";",dec=",",row.names = T)

write.table(data@reductions$pca_harmony@cell.embeddings,file = "csv/Seurat_WNN_RNA_SCT_PCA.csv",sep=";",dec=",",row.names = T)

write.table(data@reductions$lsi_harmony@cell.embeddings,file = "csv/Seurat_WNN_ATAC_LSI.csv",sep=";",dec=",",row.names = T)

write.table(data@reductions$lsi@cell.embeddings,file = "csv/Seurat_WNN_ATAC_LSI_raw.csv",sep=";",dec=",",row.names = T)

write.table(data@reductions$umap@cell.embeddings,file = "csv/Seurat_WNN_UMAP.csv",sep=";",dec=",",row.names = T)

write.table(data@neighbors$weighted.nn@nn.idx,file = "csv/Seurat_WNN_idx.csv",sep=";",dec=",",row.names = T)

write.table(data@neighbors$weighted.nn@nn.dist,file = "csv/Seurat_WNN_dist.csv",sep=";",dec=",",row.names = T)






import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import scanpy as sc
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI

from scipy import sparse
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
rcParams['figure.figsize'] = (17,17)
from scipy import sparse

sc.settings.verbosity =0

sample_color_dic = {'lib_09': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 'lib_10': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
 'lib_15': (1.0, 0.4980392156862745, 0.054901960784313725),
 'lib_23': (1.0, 0.7333333333333333, 0.47058823529411764),
 'lib_29': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 'lib_34': (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
 'lib_36': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 'lib_38': (1.0, 0.596078431372549, 0.5882352941176471),
 'lib_51': (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 'lib_54': (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
 'lib_55': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 'lib_56': (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
 'lib_57': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)}

def comparison_heatmap(adata, key1, key2, label_1=None, label_2=None, cmap = 'Reds', annot = True, figsize=(7,7)):
    if label_1==None:
        label_1=key1
    if label_2==None:
        label_2=key2
    expected_df = adata.obs[[key1,key2]].groupby(by=[key2,key1]).size().reset_index(name = 'count')
    counts = np.array(expected_df['count'].tolist())
    df = pd.DataFrame(counts.reshape(((len(adata.obs[key2].cat.categories),len(adata.obs[key1].cat.categories)))), index = expected_df[key2].unique(), columns = expected_df[key1].unique())
    if annot ==True:
        annot_ = df.astype(int)
        sc.settings.set_figure_params(figsize=figsize, color_map='gist_earth')
    else:
        annot_=None
        sc.settings.set_figure_params(figsize=figsize, color_map='gist_earth')
    s = sns.heatmap(df/np.sum(df,axis = 0), cbar_kws={'label': '% cell shared between annotations'}, cmap=cmap, vmax=1, vmin=0, annot = annot_,  fmt='.7g')
    s.set_ylabel(label_2, fontsize=12)
    s.set_xlabel(label_1, fontsize = 12)
    return df

def n_cells_histogram(adata, sample_key='sample', sample_color_dic = sample_color_dic):
    df = pd.DataFrame(adata.obs[sample_key].value_counts().index, index = adata.obs[sample_key].value_counts().index  , columns =[ sample_key ])
    df['n_cells'] = adata.obs[sample_key].value_counts().values.astype(int)
    df.reindex(index=np.argsort(df['n_cells']).index.tolist()[::-1]).plot(kind='bar', x = sample_key, y ='n_cells', color = [sample_color_dic[i] for i in np.argsort(df['n_cells']).index.tolist()[::-1]], title = 'Total nº of Cells' )


# In[4]:


palette_ = { 'PT':'#ff9896',
 'PT_VCAM1':'#c5b0d5',
 'TL':'#279e68',
 'TAL':'#c49c94',
 'DCT1':'#ff7f0e',
 'CNT':'#1f77b4',
 'PC':'#aec7e8',
 'ENDO':'#d62728',
 'MES':'#17becf',
 'FIB':'#aa40fc',
 'ICA':'#8c564b',
 'ICB':'#e377c2',
 'PODO':'#98df8a',
 'PEC':'#ffbb78',
 'LEUK':'#b5bd61',
          }


# In[5]:


palette = {'CNT':'#1f77b4',
 'DCT1':'#ff7f0e',
 'DCT2':'#279e68',
 'TL':'#279e68',
 'DCT':'#ff7f0e',
 'ENDO':'#d62728',
 'FIB':'#aa40fc',
 'ICA':'#8c564b',
 'ICB':'#e377c2',
 'LEUK':'#b5bd61',
 'MES_FIB':'#17becf',
 'MES':'#17becf',
 'VSM':'#f7b6d2',
 'PC':'#aec7e8',
 'PEC':'#ffbb78',
 'PODO':'#98df8a',
 'PT':'#ff9896',
 'PT_VCAM1':'#c5b0d5',
 'TAL':'#c49c94',
 'aPT':'#c5b0d5',
 'Unclassified':'#808080',
 'Unknown':'#000000',
 'Low_Quality_RNA':'#808080'}


# In[6]:


markers_dict = { 'PT': ['LRP2', 'CUBN', 'SLC13A1','SLC5A12','SLC22A7','SLC22A6'],
                 'PT_VCAM1': ['ITGB8', 'DCDC2', 'TPM1','HAVCR1','KIF26B','ACSM3','DLGAP1','CDH6'],
                 'TL': ['CRYAB', 'TACSTD2', 'SLC44A5', 'KLRG2', 'COL26A1', 'BOC'],
                 'TAL': ['CASR', 'SLC12A1', 'UMOD'],
                 'DCT1': ['SLC12A3','CNNM2', 'FGF13', 'KLHL3', 'TRPM6'],
                 'CNT': ['SLC8A1', 'SCN2A', 'HSD11B2', 'CALB1'],
                 'PC': ['GATA3', 'AQP2', 'AQP3'],
                 'ENDO': ['CD34', 'PECAM1', 'PTPRB', 'MEIS2', 'FLT1', 'EMCN'],
                 'MES':['PIP5K1B','ROBO1','DAAM2', 'PHTF2', 'POSTN'],
                 'FIB':['COL1A1', 'COL1A2', 'C7', 'NEGR1', 'FBLN5', 'DCN', 'CDH11'],
                 'ICA': ['SLC4A1', 'SLC26A7', 'CLNK'],
                 'ICB': ['SLC4A9', 'SLC35F3', 'SLC26A4', 'INSRR','TLDC2'],
                 'PODO': ['PTPRQ', 'WT1', 'NPHS2','CLIC5', 'PODXL'],
                 'PEC': ['CLDN1', 'VCAM1', 'CFH', 'RBFOX1', 'ALDH1A2'],
                 'LEUK': ['PTPRC','PRKCB','ARHGAP15','MS4A1','CD96'],}


# In[7]:


markers_dict_ = { 'Proximal Tubule Cell': ['LRP2', 'CUBN', 'SLC13A1','SLC5A12','SLC22A7','SLC22A6'],
                 'Adaptative Proximal Tubule Cell': ['ITGB8', 'DCDC2', 'TPM1','HAVCR1','KIF26B','ACSM3','DLGAP1','CDH6'],
                 'Thin Limb Cell': ['CRYAB', 'TACSTD2', 'SLC44A5', 'KLRG2', 'COL26A1', 'BOC'],
                 'Thick Ascending Limb Cell': ['CASR', 'SLC12A1', 'UMOD'],
                 'Distal Convoluted Tubule Cell': ['SLC12A3','CNNM2', 'FGF13', 'KLHL3', 'LHX1', 'TRPM6'],
                 'Connecting Tubule': ['SLC8A1', 'SCN2A', 'HSD11B2', 'CALB1'],
                 'Principal Cell': ['GATA3', 'AQP2', 'AQP3'],
                 'Endothelial Cell': ['CD34', 'PECAM1', 'PTPRB', 'MEIS2', 'FLT1', 'EMCN'],
                 'Mesenglial':['PIP5K1B','ROBO1','DAAM2', 'PHTF2', 'POSTN'],
                 'Fibroblast': ['COL1A1', 'COL1A2', 'C7', 'NEGR1', 'FBLN5', 'DCN', 'CDH11'],
                 'Intercalated Cell A': ['SLC4A1', 'SLC26A7', 'CLNK'],
                 'Intercalated Cell B': ['SLC4A9', 'SLC35F3', 'SLC26A4', 'INSRR','TLDC2'],
                 'Podocyte': ['PTPRQ', 'WT1', 'NPHS2','CLIC5', 'PODXL'],
                 'Parietal Epithelial Cell': ['CLDN1', 'VCAM1', 'CFH', 'RBFOX1', 'ALDH1A2'],
                 'Immune Cells': ['PTPRC','PRKCB','ARHGAP15','MS4A1','CD96'],}


# # LOAD DATA TO INTEGRATE

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import anndata as an

# LOADING PROCESSED RNA data from 10xMultiome (single nucleai) / scRNA-seq 3' / scRNA-seq 5') - Processed consistently.

adata = sc.read('Integration_scVI_Final.h5ad')

adata = adata.raw.to_adata()

adata = adata[:,adata.var['highly_variable']]

adata.var['modality'] = 'Gene Expression'

del adata.uns
del adata.obsm
del adata.uns
del adata.varm
del adata.obsp

# LOADING ATAC DATA

atac = sc.read('atac.h5ad')

atac.X = atac.layers['counts'].copy()

atac.var['madality'] = 'Peaks'

atac.obs.index = [i+'-snRNA' for i in atac.obs.index]

atac.obs['Deepscore_external'] = atac.obs['Deepscore_markers_RNA']

sc.pp.filter_genes(atac, min_cells=int(atac.shape[0] * 0.01)) # Following authors recomendation

# GENERATE OBJECT WITH SORTED FEATURES

atac_feat = atac.var.index.tolist()
rna_feat = adata.var.index.tolist()

paired = np.intersect1d(adata.obs.index, atac.obs.index)

adata_mvi = []

adata_mvi.append(adata[~adata.obs.index.isin(paired)].copy())

adata = adata[adata.obs.index.isin(paired)].copy()

del atac.varm

adata

obs = adata.obs.copy()

adata = an.concat([adata, atac[atac.obs.index.isin(paired)]], axis = 1)

adata.obs = obs

adata_mvi.append(adata)

del adata

atac = atac[~atac.obs.index.isin(paired)].copy()

adata_mvi.append(atac)

del atac

adata_mvi= scvi.data.organize_multiome_anndatas(adata_mvi[1], adata_mvi[0], adata_mvi[2],)

adata_mvi.var.loc[rna_feat,'modality'] = 'Gene Expression'
adata_mvi.var.loc[atac_feat,'modality'] = 'Peaks'

adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()

adata_mvi.obs['batch'] = [i.split('-')[-1].split('_')[0] for i in adata_mvi.obs.index]

adata_mvi.obs['batch'] = adata_mvi.obs['batch'].astype('category')
adata_mvi.obs['modality'] = adata_mvi.obs['modality'].astype('category')

adata_mvi.obs.to_csv('PRE_multiVI_RAW_OBS.csv')

adata_mvi

adata_mvi.obs = adata_mvi.obs[['sample', 'batch','modality','leiden','Deepscore_external']]

adata_mvi.write('PRE_multiVI_RAW.h5ad', compression ='gzip')

# IMPORTANT !!!
## In our study we are leveraging onlt KIDNEY CORTEX samples, nontheless we find some samples with Kidney Medula biology (e.g. Thin ascending limb).
## Therefore we decide to remove this populations.

adata_mvi.obs['kidney_area'] = 'Cortex'

adata_mvi.obs.loc[adata_mvi[(adata_mvi.obs['sample'].isin(['lib_23','lib_38']) & adata_mvi.obs['batch'].isin(['snRNA'])) & ~(adata_mvi.obs['sample'].isin(['lib_776','lib_23']) & adata_mvi.obs['batch'].isin(['scRNA']))].obs.index,'kidney_area'] = 'Medula'

adata_mvi.write('PRE_multiVI_RAW_Cortex.h5ad', compression='gzip')

adata_mvi = adata_mvi[adata_mvi.obs['kidney_area'].isin(['Cortex'])].copy()

# SETUP & TRAIN

scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality", categorical_covariate_keys=['batch','sample'])


mvi = scvi.model.MULTIVI(
    adata_mvi,
    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
)
mvi.view_anndata_setup()


mvi.train(max_epochs = 5000, early_stopping=True)


mvi.save("model_cortex", overwrite=True)


#mvi = scvi.model.MULTIVI.load("model_cortex", adata=adata_mvi)

adata_mvi.obsm["MultiVI_latent"] = mvi.get_latent_representation()
sc.pp.neighbors(adata_mvi, use_rep="MultiVI_latent", n_neighbors=100)
sc.tl.umap(adata_mvi, spread = 1.5, min_dist=0.2)
sc.pl.umap(adata_mvi, color="modality", frameon=False, save='Modality.png')
adata_mvi.write('multiVI_Cortex.h5ad', compression='gzip')


adata_mvi.obs['Annotation_bySample'] = adata_mvi.obs['Annotation_bySample'].tolist()
adata_mvi.obs.loc[adata_mvi[adata_mvi.obs['modality'].isin(['accessibility'])].obs.index,'Annotation_bySample'] = adata_mvi[adata_mvi.obs['modality'].isin(['accessibility'])].obs['bySample_leiden_Celltype']
sc.pl.umap(adata_mvi,color = 'Annotation_bySample', frameon=False, save='Annotation.png')


adata_mvi.write('multiVI_Cortex.h5ad', compression='gzip')


# Benchmark framework & mBDRC dataset
This repository contains the code for reproducibility of results in our publication: ["Systematic evaluation of single-cell multimodal data integration for comprehensive human reference atlas"](https://www.biorxiv.org/content/10.1101/2025.03.06.637075v1). Using the human kidney as a model for a complex tissue, we generated a unique benchmarking dataset for the multimodal characterization of renal cortex by integrating 3' and 5' scRNA-seq, with joint snRNA-seq and snATAC-seq data, encompassing 119,744 high-quality nuclei/cells from 18 donors.

![Project overview](/Project_scheme.png)

Following depicted guidelines we generated a unique multimodal benchmarking dataset for renal cortex characterization (mBDRC). In house developed interpretable machine learning tool for reference-based cell-type classification, scOMM ([single-cell Omics Multimodal Mapping](https://github.com/mereulab/scOMM)), has been used to anchor mBDRC to previous human kidney references to produce two layer of annotation. This dataset can be found in the CZ CELLxGENE Discover platform (link).

![Dataset](/mBDRC.png)

## Citation

If you use this code, please cite:

**Paper**
Acera-Mateos M, Adiconis X, Li J-K, Marchese D, Caratù G, Hon C-C, Tiwari P, Kojima M, Vieth B, Murphy MA, Simmons SK, Lefevre T, Claes I, O’Connor CL, Menon R, Otto EA, Ando Y, Vandereyken K, Kretzler M, Bitzer M, Fraenkel E, Voet T, Enard W, Carninci P, Heyn H, Levin JZ, Mereu E. *Systematic evaluation of single-cell multimodal data integration for comprehensive human reference atlas.* **bioRxiv** (2025). doi:10.1101/2025.03.06.637075

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17475964.svg)](https://doi.org/10.5281/zenodo.17475964)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# scanpy

# %% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, fnmatch
import re
import pickle
import seaborn as sns
import scanpy as sc
import sklearn # scikit-misc
import scvi
import leidenalg

# %% set cwd and read files

cwd = r"/home/ubuntu/data/" # "r" means the string will be treated as raw string

# %% scanpy default settings 

sc.settings.verbosity = 3 # verbosity: errors(0), warnings(1), info(2), hints(3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=100, facecolor='white', color_map="viridis")

# %% load concatenate clean dataset
concat_adata = sc.read_h5ad(cwd + "concat_adata.h5ad")
concat_adata.obs_names_make_unique()

# %% normalization
# normalize counts in each cell. So, total counts ends up to the same value. 
# normallize data matrix to 10.000 reads per cell, so that counts become comparable among cells.
concat_adata.layers["counts"] = concat_adata.X.copy() # save raw data before doing integration using scvi

sc.pp.normalize_total(concat_adata, target_sum=1e4)
sc.pp.log1p(concat_adata) # log n+1 transform
concat_adata.raw = concat_adata

# %% integration with scvi (all genes)
# setup_anndata() alerts scvi-tools to the locations of various matrices inside the anndata. 
# so, scvi is notified that your dataset has batches, annotations, etc.
scvi.model.SCVI.setup_anndata(concat_adata, layer="counts",
                              categorical_covariate_keys=["Sample", "Dataset"],
                              continuous_covariate_keys=["pct_counts_mt", "total_counts"])

model = scvi.model.SCVI(concat_adata) # initialize model
model.train() # may take a while without GPU
model.save(cwd + "model") # save model
model = scvi.model.SCVI.load(cwd + "model", concat_adata) # load model

concat_adata.obsm["X_scVI"] = model.get_latent_representation() # obtaining model outputs
# concat_adata.obsm["X_scVI"].shape
concat_adata.layers["scvi_normalized"] = model.get_normalized_expression(concat_adata, library_size = 10e4) # save scvi_normalized expression values to the adata object
# concat_adata.layers["scvi_normalized"][:5, :5]

# without batch correction (run PCA then generate UMAP plots)
sc.tl.pca(concat_adata)
sc.pp.neighbors(concat_adata, n_pcs=30, n_neighbors=20)
sc.tl.umap(concat_adata, min_dist=0.3)
sc.pl.umap(concat_adata, color=["Dataset", "Condition"], ncols=2, 
           legend_fontsize="xx-small")
plt.savefig(cwd + "../figures/concat_adata_BeforeIntegration.pdf", bbox_inches="tight")

# with batch correction (use scVI latent space for UMAP generation)
sc.pp.neighbors(concat_adata, use_rep="X_scVI")
sc.tl.umap(concat_adata)
sc.pl.umap(concat_adata, color=["Dataset", "Condition"], ncols=2,
           legend_fontsize="xx-small")
plt.savefig(cwd + "../figures/concat_adata_AfterIntegration.pdf", bbox_inches="tight")

# clustering on the scVI latent space
# neighbors were already computed using scVI
sc.tl.leiden(concat_adata, resolution=0.5)
sc.pl.umap(concat_adata, color=["leiden"], frameon=False, layer="scvi_normalized")

concat_adata.write_h5ad(cwd + "integrated.h5ad")

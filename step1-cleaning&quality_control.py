#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, fnmatch
import re
import pickle
import scanpy as sc
import scvi

# %% set cwd and read files

cwd = r"/home/ubuntu/data/"

# %% scanpy default settings

sc.settings.verbosity = 3 # verbosity: errors(0), warnings(1), info(2), hints(3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=100, facecolor='white', color_map="viridis")

# %% cleaning and quality control

ds = "XXX"

# get the sample name
sample = pd.read_excel(cwd + "lung_samples.xlsx")
sample = sample[sample["Dataset"] == ds]
sample["cnd"] = sample["Sample"] + "_" + sample["Condition"]

# rename samples
files = fnmatch.filter(os.listdir(cwd + ds), "*.h5") # get a list of .h5 files

for i in files:
    print(i)
    for j in sample["cnd"]:
        print(j)
        if (i.split("_")[0] == j.split("_")[0]):
            os.rename(os.path.join(cwd + ds + "/" + i),
                      os.path.join(cwd + ds + "/" + ds + "_" + j.split("_")[0] + "_" + j.split("_")[1] + ".h5"))

new_files = fnmatch.filter(os.listdir(cwd + ds), "*.h5")

# %%% pp for .h5 files (preprocessing, remove doublets, and quality control)

# -------------------------------- annotate cells and remove doublets using SOLO --------------------------------
# wget.download("https://figshare.com/ndownloader/files/34701991")
# !unzip TS_Lung.h5ad.zip
cell_ref = sc.read_h5ad(cwd + "cell_ref.h5ad")
cell_ref.X = cell_ref.layers["raw_counts"] # scvi works on the raw counts

adatas = []
for file in new_files:
    print(file)
    adata = sc.read_10x_h5(cwd + ds + "/" + file)
    adata.var_names_make_unique()

    sc.pp.filter_genes(adata, min_cells=10) # get rid of genes which express in less than 10 cells to narrow down the adata for scvi

    adata = adata.concatenate(cell_ref)

    # run feature selection to reduce the number of features (genes in this case)
    # For scVI, we recommend anywhere from 1,000 to 10,000 HVGs, but it will be context-dependent.
    # the number of genes should be at least half of the number of cells and not more for scvi
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, flavor="seurat_v3")
    # setup_anndata(), alerts scvi-tools to the locations of various matrices inside the anndata.
    # If batches are registered with scvi-tools, the subsequent model will correct for batch effects.
    # If the adata is modified, run setup_anndata again before creating another instance of a model.
    scvi.model.SCVI.setup_anndata(adata) # set up scvi model
    vae = scvi.model.SCVI(adata)
    vae.train() # train the model. get latent representation using vae.get_latent_representation()

    adata.obs["cell_ontology_class"] = adata.obs["cell_ontology_class"].cat.add_categories("Unknown") # add a new category  adata.obs = adata.obs.fillna(value = {"cell_ontology_class": "Unknown"}) # fill na with "Unknown"
    adata.obs = adata.obs.fillna(value = {"cell_ontology_class": "Unknown"}) # fill na with "Unknown"
    lvae = scvi.model.SCANVI.from_scvi_model(vae, adata=adata, unlabeled_category="Unknown", labels_key="cell_ontology_class")
    lvae.train()
    adata.obs["predicted"] = lvae.predict(adata) # get the predicted labels
    adata.obs["Barcode_2"] = adata.obs.index.map(lambda x: x[:-2]) # Barcode_2 columns is everything in the index except the last 2 characters
    cell_mapper = dict(zip(adata.obs.Barcode_2, adata.obs.predicted)) # make a dictionary with Barcode_2 and predicted columns

    solo = scvi.external.SOLO.from_scvi_model(vae) # train SOLO model to predict doublets using vae model
    solo.train()
    df = solo.predict()
    df["prediction"] = solo.predict(soft=False) # create a dataframe with prediction column. Passing "soft=False" would return predicted values
    df.index = df.index.map(lambda x: x[:-2]) # remove the last two characters from barcodes
    df["dif"] = df.doublet - df.singlet # difference between doublet and singlet prediction
    # sns.displot(df[df.prediction == "doublet"], x="dif") # plot the distribution
    doublets = df[(df.prediction == "doublet") & (df.dif > 1)] # create doublet dataframe from barcodes which have high certaintly to be doublets

    # run a fresh adata, add metadata to adata, and keep singlets
    adata = sc.read_10x_h5(cwd + ds + "/" + file)
    adata.var_names_make_unique()
    adata.obs[["Dataset", "Sample", "Condition"]] = re.split(("_|.h5"), file)[0:3]
    adata.obs["Barcode"] = adata.obs.index
    adata.obs["cell.type_scvi"] = adata.obs.index.map(cell_mapper) # add cell_mapper to the original adata.obs
    adata.obs["doublet"] = adata.obs.index.isin(doublets.index) # add a true/false column called doublet that includes barcodes matched with doublet.index
    adata = adata[~adata.obs.doublet] # remove doublets and keep singlets

    adatas.append(adata)

# save adatas
with open("data/adatas_pkl", "wb") as f:
    pickle.dump(adatas, f)

# --------------------------------- QC before filtering --------------------------------------------------------
with open("data/adatas_pkl", "rb") as f:
    adatas = pickle.load(f)

adata = sc.concat(adatas)

adata.var["mt"] = adata.var_names.str.startswith("MT-") # get the list of mito genes
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True) # log1p is defined as log(x+1)
sc.pl.violin(adata, keys=["n_genes_by_counts", "total_counts", "pct_counts_mt"], groupby="Sample",
             jitter=0.4, multi_panel=True, rotation=90, show=False)
plt.savefig(cwd + "../figures/beforeQC.pdf", bbox_inches="tight")

# --------------------------------- QC after filtering -----------------------------------------------------------
with open("data/adatas_pkl", "rb") as f:
    adatas = pickle.load(f)

for i in range(len(adatas)):
    print(i)

    sc.pp.filter_cells(adatas[i], min_genes=200) # get rid of cells with fewer than 200 genes
    sc.pp.filter_genes(adatas[i], min_cells=3) # get rid of genes that are found in fewer than 3 cells
    adatas[i].var["mt"] = adatas[i].var_names.str.startswith("MT-") # get the list of mito genes
    sc.pp.calculate_qc_metrics(adatas[i], qc_vars=["mt"], percent_top=None, log1p=False, inplace=True) # log1p is defined as log(x+1)
    # n_genes_by_counts: the number of genes expressed in the count matrix
    # total_counts: the total counts per cell"
    # pct_counts_matrix: the percentage of counts in mitochondrial genes
    upper_lim = np.quantile(adatas[i].obs.n_genes_by_counts.values, .98) # 98 percentile
    adatas[i] = adatas[i][adatas[i].obs.n_genes_by_counts < upper_lim]
    adatas[i] = adatas[i][adatas[i].obs.pct_counts_mt < 15]

adata = sc.concat(adatas)

sc.pl.violin(adata, keys=["total_counts", "n_genes_by_counts", "pct_counts_mt"], groupby="Sample",
             jitter=0.4, multi_panel=True, rotation=90, show=False)
plt.savefig(cwd + "../figures/afterQC.pdf", bbox_inches="tight")

adata.write_h5ad(cwd + "cln_ds.h5ad")

adata = sc.read_h5ad(cwd + "cln_ds.h5ad")

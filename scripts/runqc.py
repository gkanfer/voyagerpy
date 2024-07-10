import numpy as np
import geopandas as gpd
import pandas as pd

import scanpy as sc
import voyagerpy as vp
import seaborn as sns
import os
import pickle
from matplotlib.pyplot import imread
from collections import OrderedDict
import json
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pathout = "/data/kanferg/Sptial_Omics/VoyagerPy_fork/voyagerpy/out"

path_016 = "/data/kanferg/Sptial_Omics/playGround/Data/Visium_HD_Mouse_Brain_square_example/square_016um"
andata016_ = sc.read_visium(path=path_016)

def print_all_keys(d, parent_key=''):
    if isinstance(d, dict) or isinstance(d, OrderedDict):
        for key, value in d.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            print(new_key)
            if isinstance(value, dict) or isinstance(value, OrderedDict):
                print_all_keys(value, new_key)

# Example usage with andata016_.uns
print_all_keys(andata016_.uns)
andata016_.var_names_make_unique()
andata016_.obsm['spatial'] = np.array(andata016_.obsm['spatial'], dtype=np.float64)
andata016_.uns['spatial']['img'] = andata016_.uns['spatial']['Visium_HD_Mouse_Brain'].pop("images")
andata016_.uns['spatial']['scale'] = andata016_.uns['spatial']['Visium_HD_Mouse_Brain'].pop("scalefactors")
andata016_.uns['spatial']['metadata'] = andata016_.uns['spatial']['Visium_HD_Mouse_Brain'].pop("metadata")
andata016_.uns['spatial'].pop("Visium_HD_Mouse_Brain")
print_all_keys(andata016_.uns)

#_ = vp.plt.imshow(andata016_)

is_mt = andata016_.var_names.str.startswith('mt')
vp.utils.add_per_cell_qcmetrics(andata016_, subsets={'mito': is_mt})

spot_diameter_fullres = andata016_.uns['spatial']['scale'].pop('spot_diameter_fullres')
andata016_.uns['spatial']['scale']['spot_diameter_fullres'] = {'pxl_col_in_fullres':spot_diameter_fullres,'pxl_row_in_fullres':spot_diameter_fullres}

andata016_.uns['spatial']['scale']

scale = 1
scale_dict = andata016_.uns["spatial"].get("scale", {})
spot_diam = scale_dict.get("spot_diameter_fullres")
visium_spots = gpd.GeoSeries.from_xy(andata016_.obsm['spatial'][:,0], andata016_.obsm['spatial'][:,1]).scale(scale, scale, origin=(0, 0))

# _ = vp.spatial.set_geometry(andata016_, geom="spot_poly", values=visium_spots)

# qc_features = ["sum", "detected", "subsets_mito_percent"]
# axs = vp.plt.plot_spatial_feature(
#     andata016_,
#     qc_features,
#     image_kwargs=dict(),
#     subplot_kwargs=dict(figsize=(8,8), dpi=100, layout='tight')
# )

# ax = vp.plotting.plot_barcodes_bin2d(
#     andata016_,
#     x='sum',
#     y='detected',
#     bins=76,
#     figsize=(4, 4)
# )

# The original count data
andata016_.layers['counts'] = andata016_.X.copy()
# Log-normalize the adata.X matrix
vp.utils.log_norm_counts(andata016_, inplace=True)
andata016_.layers['logcounts'] = andata016_.X.copy()

andata016_.obs['var'] = np.var(andata016_.X.todense(), axis=1)

# gene_means = andata016_.obs['sum']
# var = andata016_.obs['var']

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# sns.scatterplot(x=andata016_.obs['sum'], y=andata016_.obs['var'],ax = axs[0])
# sns.scatterplot(x=andata016_.obs['detected'], y=andata016_.obs['var'],ax = axs[1])

# # Set plot labels and title
# axs[0].set_xlabel('Mean Total Signal')
# axs[0].set_ylabel('Variance')
# axs[0].set_title('Variance as a Function of \n Mean Total Signal')

# axs[1].set_xlabel('Mean Total Signal')
# axs[1].set_ylabel('Variance')
# axs[1].set_title('Variance as a Function of \n  Mean Total Signal')
# plt.subplots_adjust(wspace=0.5)

gene_var = vp.utils.model_gene_var(andata016_.layers['logcounts'], gene_names=andata016_.var_names)
hvgs = vp.utils.get_top_hvgs(gene_var)

andata016_.var['highly_variable'] = False
andata016_.var.loc[hvgs, 'highly_variable'] = True

andata016_.X = vp.utils.scale(andata016_.X, center=True)
sc.tl.pca(andata016_, use_highly_variable=True, n_comps=30, random_state=1337)
andata016_.X = andata016_.layers['logcounts'].copy()



with PdfPages(os.path.join(pathout, f'pca_elboPlottest.pdf')) as pdf:
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    vp.plt.elbow_plot(andata016_, ndims=30)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pdf.savefig()
    plt.close()

    
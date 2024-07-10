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
from tqdm import tqdm

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
sc.pp.filter_cells(andata016_, min_counts = 100)
sc.pp.filter_cells(andata016_, min_genes = 100)
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

andata016_.layers['counts'] = andata016_.X.copy()
# Log-normalize the adata.X matrix
vp.utils.log_norm_counts(andata016_, inplace=True)
andata016_.layers['logcounts'] = andata016_.X.copy()

andata016_.obs['var'] = np.var(andata016_.X.todense(), axis=1)

gene_var = vp.utils.model_gene_var(andata016_.layers['logcounts'], gene_names=andata016_.var_names)
hvgs = vp.utils.get_top_hvgs(gene_var)

andata016_.var['highly_variable'] = False
andata016_.var.loc[hvgs, 'highly_variable'] = True

andata016_.X = vp.utils.scale(andata016_.X, center=True)
sc.tl.pca(andata016_, use_highly_variable=True, n_comps=30, random_state=1337)
andata016_.X = andata016_.layers['logcounts'].copy()

andata016_.uns['config'] = OrderedDict()
andata016_.uns["config"]["secondary_var_names"] = andata016_.var_names

from leidenalg import ModularityVertexPartition

sc.pp.neighbors(
    andata016_,
    n_pcs=9,
    use_rep='X_pca',
    method='gauss',
    n_neighbors=80
)
sc.tl.leiden(
    andata016_,
    random_state=29,
    resolution=None,
    key_added='cluster',
    partition_type=ModularityVertexPartition
)

# The code snippet you provided is performing spatial analysis using the `vp.spatial.local_moran`
# function and then visualizing the results using `vp.plt.plot_local_result`.
# qc_features = ['Mbp','Prkcd','Nrgn']
# vp.spatial.local_moran(andata016_,feature = qc_features)   
# vp.plt.plot_local_result(
#         andata016_,
#         obsm = 'local_moran',
#         features=['Mbp','Prkcd','Nrgn'],
#         )

hvgs = vp.utils.get_top_hvgs(gene_var,n =100)

andata016_.var['highly_variable'] = False
andata016_.var.loc[hvgs, 'highly_variable'] = True

markers = vp.utils.find_markers(andata016_, hvg = True)
andata016_.var['symbol'] = andata016_.var['gene_ids'].values
marker_genes = [
    marker.sort_values(by='p.value').iloc[0].name
    for _, marker in sorted(markers.items())]

marker_genes_symbols = andata016_.var.loc[marker_genes, "symbol"].tolist()
andata016_.var.loc[marker_genes, ["symbol"]]  

pathout = "/data/kanferg/Sptial_Omics/VoyagerPy_fork/voyagerpy/out"

with PdfPages(os.path.join(pathout, f'Gene_DE.pdf')) as pdf:
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    _ = vp.plt.plot_expression(andata016_,marker_genes[:4],groupby='cluster',show_symbol=True,layer='logcounts',figsize=(9, 6), scatter_points=False) 
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pdf.savefig()
    plt.close()
    
    
sc.pp.neighbors(
    andata016_,
    n_neighbors=20,
    n_pcs=9,
    use_rep='X_pca',
    knn=True,
    random_state=29,
    method='gauss', # one of umap, gauss, rapids
    metric='euclidean', # many metrics available,
    key_added='knn'
)

dist = andata016_.obsp['knn_distances'].copy()
dist.data = 1 / dist.data

# row normalize the matrix, this makes the matrix dense.
dist /= dist.sum(axis=1)

# convert dist back to sparse matrix
from scipy.sparse import csr_matrix
andata016_.obsp["knn_weights"] = csr_matrix(dist)

del dist

knn_graph = "knn_weights"

# adata.obsp["knn_connectivities"] represent the edges, while adata.opsp["knn_weights"] represent the weights
andata016_.obsp["knn_connectivities"] = (andata016_.obsp[knn_graph] > 0).astype(int)
vp.spatial.set_default_graph(andata016_, knn_graph)
vp.spatial.to_spatial_weights(andata016_, graph_name=knn_graph)

qc_features = ["sum", "detected", "subsets_mito_percent"]
morans = vp.spatial.moran(andata016_, qc_features, graph_name=knn_graph)
andata016_.uns['spatial']['moran'][knn_graph].loc[qc_features, ["I"]]


vp.spatial.compute_spatial_lag(
    andata016_,
    qc_features,
    graph_name=knn_graph,
    inplace=True
)

with PdfPages(os.path.join(pathout, f'Moran_sum.pdf')) as pdf:
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    ax = vp.plt.moran_plot(andata016_, feature='sum', color_by='cluster', alpha=0.8)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pdf.savefig()
    plt.close()

with PdfPages(os.path.join(pathout, f'Moran_detected.pdf')) as pdf:
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    ax = vp.plt.moran_plot(andata016_, feature='detected', color_by='cluster', alpha=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pdf.savefig()
    plt.close()
    
with PdfPages(os.path.join(pathout, f'Moran_mito_percent.pdf')) as pdf:
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    ax = vp.plt.moran_plot(andata016_, feature='subsets_mito_percent',color_by="cluster",alpha=0.5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pdf.savefig()
    plt.close()
    
_ = vp.spatial.local_moran(andata016_, qc_features, graph_name=knn_graph)

with PdfPages(os.path.join(pathout, f'local_Morn.pdf')) as pdf:
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    ax = vp.plt.plot_local_result(andata016_, obsm="local_moran", features=qc_features, figsize=(6,6),divergent=True ,s=1,figtitle="Local Moran's I")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pdf.savefig()
    plt.close()

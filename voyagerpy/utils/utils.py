#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def is_highres(adata: AnnData) -> bool:
    if "hires" in adata.uns["spatial"]["img"]:
        return True
    if "lowres" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


def is_lowres(adata: AnnData) -> bool:
    if "lowres" in adata.uns["spatial"]["img"]:
        return True
    if "hires" in adata.uns["spatial"]["img"]:
        return False
    raise ValueError("Cannot find image data in .uns['spatial']")


def make_unique(items: List) -> List:
    items = items[:]
    for i in range(len(items) - 1, -1, -1):
        if items.count(items[i]) > 1:
            items.pop(i)
    return items


def get_scale(adata: AnnData, res: Optional[str] = None) -> float:
    if res not in [None, "hi", "hires", "lo", "lowres"]:
        raise ValueError(f"Unrecognized value {res} for res.")

    scale_dict = adata.uns["spatial"].get("scale", {})
    scale_key = None

    if is_lowres(adata) and res in [None, "lowres", "lo"]:
        scale_key = "tissue_lowres_scalef"
    elif is_highres(adata) and res in [None, "hires", "hi"]:
        scale_key = "tissue_hires_scalef"

    if scale_key is None:
        raise ValueError("Invalid resolution. Make sure the correct image is loaded.")
    elif scale_key not in scale_dict:
        raise KeyError(f"Could not find scale factor {scale_key} for {res}")

    return scale_dict[scale_key]


def add_per_gene_qcmetrics(adata: AnnData, subsets: Dict[str, np.ndarray], force: bool = False) -> None:
    if "sum" not in adata.var.keys() or force:
        adata.var["sum"] = adata.X.sum(axis=0).T  # type: ignore

    if "detected" not in adata.var.keys() or force:
        adata.var["detected"] = np.diff(adata.X.tocsc().indptr)  # type: ignore

    for key, subset in subsets.items():
        sum_key = f"subsets_{key}_sum"
        detected_key = f"subsets_{key}_detected"
        percent_key = f"subsets_{key}_percent"

        ss = adata[subset, :].X

        if sum_key not in adata.var.keys() or force:
            adata.var[sum_key] = ss.sum(axis=0).T  # type: ignore

        if detected_key not in adata.var.keys() or force:
            adata.var[detected_key] = np.diff(ss.tocsc().indptr)  # type: ignore

        if percent_key not in adata.var.keys() or force:
            adata.var[percent_key] = adata.var[sum_key] / adata.var["sum"] * 100


def add_per_cell_qcmetrics(adata: AnnData, subsets: Dict[str, np.ndarray], force: bool = False) -> None:
    if "sum" not in adata.obs.keys() or force:
        adata.obs["sum"] = adata.X.sum(axis=1)  # type: ignore

    if "detected" not in adata.obs.keys() or force:
        adata.obs["detected"] = np.diff(adata.X.tocsr().indptr)  # type: ignore

    for key, subset in subsets.items():
        sum_key = f"subsets_{key}_sum"
        detected_key = f"subsets_{key}_detected"
        percent_key = f"subsets_{key}_percent"

        subset_X = adata[:, subset].X
        if sum_key not in adata.obs.keys() or force:
            adata.obs[sum_key] = subset_X.sum(axis=1)  # type: ignore

        if detected_key not in adata.obs.keys() or force:
            adata.obs[detected_key] = np.diff(subset_X.tocsr().indptr)  # type: ignore

        if percent_key not in adata.obs.keys() or force:
            adata.obs[percent_key] = adata.obs[sum_key] / adata.obs["sum"] * 100


def log_norm_counts(
    adata: Union[np.ndarray, sp.csr_matrix, sp.csr_matrix, AnnData],
    layer: Optional[str] = None,
    inplace: bool = False,
    base: Union[None, int, bool] = 2,
    pseudocount: int = 1,
    zero_to_zero: bool = True,
) -> Union[np.ndarray, sp.csr_matrix, sp.csr_matrix]:
    """\
Compute log-normalized counts. If ``adata`` is of type AnnData and ``layer`` is not ``None``, the layer is used instead of ``adata.X``.
Otherwise, ``adata`` is assumed to be a sparse matrix or a dense matrix. All rows are normalized to sum to :math:`\\bar{N}`, then log-transformed,
where :math:`\\bar{N}` is the mean of the total counts across all cells. If `zero_to_zero` is `True`, then zeros in the input matrix will map to zeros in the output matrix.
If pseudocount is not 1 and zero_to_zero is False, will add pseudocount to all values before log-transforming. This makes the matrix dense in an intermediate step
and may take a long time with large memory footprint.

:param adata: The matrix or AnnData object to normalize.
:type adata: Union[np.ndarray, sp.csr_matrix, sp.csr_matrix, AnnData]
:param layer: If not None, normalize this layer, defaults to None
:type layer: Optional[str], optional
:param inplace: Whether to normalize the matrix in-place, defaults to False
:type inplace: bool, optional
:param base: The logarithm base to use, defaults to 2. If None, use natural logarithm. If False, do not log-transform.
:type base: Union[None, int, bool], optional
:param pseudocount: Pseudocounts to use. If 1, compute log1p, defaults to 1
:type pseudocount: int, optional
:param zero_to_zero: If True, zeros in the input matrix will map to zeros in the output matrix, defaults to True
:type zero_to_zero: bool, optional
:raises TypeError: if adata is not AnnData, np.ndarray, sp.csr_matrix, or scipy.sparse.csc_matrix
:return: The log-normalized counts
:rtype: Union[np.ndarray, sp.csr_matrix, sp.csr_matrix]
    """

    # Roughly equivalent to:
    # target_sum = adata.X.sum(axis=1).mean()
    # sc.pp.normalize_total(adata, target_sum=target_sum)
    # sc.pp.log1p(adata, base=base)

    if isinstance(adata, AnnData):
        X = adata.X if layer is None else adata.layers[layer]
    elif isinstance(adata, (np.ndarray, sp.csr_matrix, sp.csc_matrix)):
        X = adata
    else:
        raise TypeError("adata must be AnnData, np.ndarray, sp.csr_matrix, or scipy.sparse.csc_matrix")

    if not inplace:
        X = X.copy()

    cell_sums = np.ravel(X.sum(axis=1))
    cell_sums /= cell_sums.mean()

    # Normalize matrix in-place
    if sp.issparse(X):
        if sp.isspmatrix_csr(X):
            X.data /= np.repeat(cell_sums, np.diff(X.indptr))
        elif sp.isspmatrix_csc(X):
            X.data /= cell_sums[X.indices]

        # Add pseudocount
        if pseudocount != 1:
            if zero_to_zero:
                X.data += pseudocount - 1
            else:
                # Let's try to avoid this state
                cls_ = type(X)
                X = cls_(X.A + pseudocount - 1)
    else:
        X /= np.expand_dims(cell_sums, axis=1)

        if pseudocount != 1:
            if zero_to_zero:
                nonzero = np.where(X != 0)
                X[nonzero] += pseudocount - 1
            else:
                X += pseudocount - 1

    if base is False:
        return X

    # in-place log1p
    data, where = (X.data, True) if sp.issparse(X) else (X, X > 0)
    np.log1p(data, out=data, where=where)

    if base is not None and base is not True:
        data /= np.log(base)

    return X


def scale(
    X: Union[sp.spmatrix, np.ndarray, np.matrix],
    center: bool = True,
    unit_variance: bool = True,
    center_before_scale: bool = True,
    ddof: int = 1,
) -> np.ndarray:

    if sp.issparse(X):  # or isinstance(X, np.matrix):
        A = X.todense()  # type: ignore
    elif isinstance(X, np.ndarray):
        A = X.copy()
    else:
        raise TypeError("X must be of type np.ndarray or sp.spmatrix.")

    del X

    # if not isinstance(A, np.ndarray) or isinstance(A, np.matrix):
    #     raise RuntimeError("A must be a numpy array. This should not happen.")

    kwargs = dict(axis=0, keepdims=True)
    if isinstance(A, np.matrix):
        kwargs.pop("keepdims")

    if center and center_before_scale:
        A -= A.mean(**kwargs)

    if unit_variance:
        std = A.std(ddof=ddof, **kwargs)
        w = np.where(std < 1e-8)
        std[w] = 1
        A = np.divide(A, std)

    if center and not center_before_scale:
        A -= A.mean(axis=0)

    return A


def normalize_csr(X: sp.csr_matrix, byrow: bool = True) -> sp.csr_matrix:
    axis = int(byrow)
    sum_ = sp.csr_matrix(X.sum(axis=axis))
    sum_.eliminate_zeros()
    sum_.data = 1 / sum_.data
    sum_ = sp.diags(sum_.toarray().ravel())
    return sum_.dot(X) if byrow else X.dot(sum_)


def kurtosis(x, method: str = "moments"):
    if method != "moments":
        raise NotImplementedError('Only method="moments" is currently implemented')

    n = x.size
    x_bar = x.mean()
    # From asbio::kurt in R:
    # methods of moments kurtosis is
    #   m_4 / m_2^2  with m_j = sum((x-x_mean)**j)/n

    m_2 = np.square(x - x_bar).mean()
    m_4 = np.power(x - x_bar, 4).mean()

    return m_4 / m_2**2


def listify(
    x: Union[None, int, str, Iterable[str], Iterable[int]],
    size: Optional[int] = None,
) -> List[Any]:
    """Converts a string or an iterable of strings to a list of strings.

    Parameters
    ----------
    x : Union[str, Iterable[str]]
        The string or iterable to convert.

    Returns
    -------
    List[str]
        The list of strings.
    """
    nontype = type(None)
    size = size if size is not None else 1
    return [x] * size if isinstance(x, (int, float, str, nontype)) else list(x)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Iterable,
    Optional,
    Tuple,
    Union,
)

import geopandas as gpd
import numpy as np
from anndata import AnnData
from cv2 import (
    CHAIN_APPROX_SIMPLE,
    COLOR_BGR2GRAY,
    COLOR_RGB2BGR,
    RETR_TREE,
    arcLength,
    contourArea,
    cvtColor,
    findContours,
    pointPolygonTest,
    threshold,
)
from shapely.geometry import Point, Polygon

from voyagerpy import utils as utl


# create spatial functions with shapely
def get_approx_tissue_boundary(adata: AnnData, size: str = "hires", paddingx: int = 0, paddingy: int = 0) -> Tuple[int, int, int, int]:
    if size == "hires":
        scl = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    else:
        scl = adata.uns["spatial"]["scale"]["tissue_lowres_scalef"]
    bot = int(np.max(adata.obs[adata.obs["in_tissue"] == 1]["pxl_row_in_fullres"]) * scl)
    top = int(np.min(adata.obs[adata.obs["in_tissue"] == 1]["pxl_row_in_fullres"]) * scl)
    right = int(np.max(adata.obs[adata.obs["in_tissue"] == 1]["pxl_col_in_fullres"]) * scl)
    left = int(np.min(adata.obs[adata.obs["in_tissue"] == 1]["pxl_col_in_fullres"]) * scl)
    if paddingx != 0:
        left = left - paddingx
        right = right + paddingx
    if paddingy != 0:
        top = top - paddingy
        bot = bot + paddingy

    return top, bot, left, right


Contour = Any
# %%


def get_tissue_contour_score(cntr: Contour, adata: AnnData, size: str = "hires") -> float:

    scl = utl.get_scale(adata, res=size)

    # tissue_barcodes = adata.obs[adata.obs["in_tissue"] == 1]
    # non_tissue_barcodes  = adata.obs[adata.obs["in_tissue"] != 1]
    # total = tissue_barcodes.shape[0]

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(adata.obs.shape[0]):
        # print([int(tissue_barcodes.iloc[i,3]*0.2),int(tissue_barcodes.iloc[i,4]*0.2)])
        # print(cv2.pointPolygonTest(big_cntrs[0], (int(tissue_barcodes.iloc[i,4]*0.2),int(tissue_barcodes.iloc[i,3]*0.2)), False) )
        test_pt = (int(adata.obs["pxl_col_in_fullres"][i] * scl), int(adata.obs["pxl_row_in_fullres"][i] * scl))
        polytest = pointPolygonTest(cntr, test_pt, False)
        if polytest == 1:
            if adata.obs["in_tissue"][i] == 1:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if adata.obs["in_tissue"][i] == 1:
                fn = fn + 1
            else:
                tn = tn + 1

    # method youden j....whynot
    # print([tp,fp,tn,fn])
    J = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
    # print(J)
    return J


def detect_tissue_threshold(adata: AnnData, size: str = "hires", low: int = 200, high: int = 255) -> Tuple[int, Optional[Contour]]:

    bgr_img = cvtColor(adata.uns["spatial"]["img"][size], COLOR_RGB2BGR)
    bgr_img = (bgr_img * 255).astype("uint8")  # type: ignore
    imgray = cvtColor(bgr_img, COLOR_BGR2GRAY)
    px_thrsh = low
    thrsh_score_all = 0
    best_thrsh = 0
    best_cntr_all = None
    for i in range(px_thrsh, high):
        ret, thresh = threshold(imgray, i, 255, 0)
        # contours
        contours, contours2 = findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE)
        # filter contours by size

        big_cntrs = []
        # marked = bgr_img.copy();
        for contour in contours:
            area = contourArea(contour)
            if area > 10000:
                # print(area);
                big_cntrs.append(contour)
        # print(len(big_cntrs))
        score = 0
        best_cntr = None
        for j in range(len(big_cntrs)):
            new_score = get_tissue_contour_score(big_cntrs[j], adata)
            if new_score > score:
                best_cntr = big_cntrs[j]
                score = new_score

        if score > thrsh_score_all:
            # print("score is " ,thrsh_score_all)
            if best_cntr_all is not None:
                # print("ratio is " ,cv2.arcLength(best_cntr_all, True)/ cv2.arcLength(best_cntr, True))

                if arcLength(best_cntr_all, True) / arcLength(best_cntr, True) < 0.9:
                    if abs(thrsh_score_all - score) < 0.1:
                        break

            best_thrsh = i
            best_cntr_all = best_cntr
            thrsh_score_all = score
            # print("score is " ,thrsh_score_all)
            # if(best_cntr_all is not None):

            #    print(cv2.arcLength(best_cntr_all, True)/cv2.arcLength(best_cntr, True))

    return best_thrsh, best_cntr_all


def get_tissue_boundary(
    adata: AnnData,
    threshold_low: int = None,
    size: str = "hires",
    strictness: Optional[int] = None,
    inplace: bool = False,
    # detect_threshold: bool = False,
) -> Polygon:

    # TODO: Do we want assert that size is either 'lowres' or 'hires'?
    res = "hires" if size == "hires" else "lowres"
    scl = utl.get_scale(adata, res=res)

    # load image

    bgr_img = cvtColor(adata.uns["spatial"]["img"][res], COLOR_RGB2BGR)
    bgr_img = (bgr_img * 255).astype("uint8")  # type: ignore

    # rescale
    # scale = 0.25
    # h, w = img.shape[:2]
    # h = int(h*scale)
    # w = int(w*scale)
    # img = cv2.resize(img, (w,h))

    # hsv
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    # h, s, v = cv2.split(hsv);

    # thresh
    # thresh = cv2.inRange(h, 140, 179);
    if threshold_low is not None:
        imgray = cvtColor(bgr_img, COLOR_BGR2GRAY)
        ret, thresh = threshold(imgray, threshold_low, 255, 0)
        # contours
        contours, contours2 = findContours(thresh, RETR_TREE, CHAIN_APPROX_SIMPLE)
        # filter contours by size
        big_cntrs = []
        # marked = bgr_img.copy();
        for contour in contours:
            area = contourArea(contour)
            if area > 10000:
                # print(area);
                big_cntrs.append(contour)
        # for all contours check if all points are within countour and all points outside it
        score = 0
        best_cntr: Optional[np.ndarray] = None
        for i in range(len(big_cntrs)):
            new_score = get_tissue_contour_score(big_cntrs[i], adata)
            if new_score > score:
                score = new_score
                best_cntr = big_cntrs[i]
    else:
        thrsh, best_cntr = detect_tissue_threshold(adata, size=size)

        # not_tissue_barcodes = adata.obs[adata.obs["in_tissue"] == 0]

        # ts_out_p = []

    # if(strictness == "strict"):

    # cv2.drawContours(marked, big_cntrs, -1, (0, 255, 0), 3);

    # # create a mask of the contoured image
    # mask = np.zeros_like(imgray);
    # mask = cv2.drawContours(mask, big_cntrs, -1, 255, -1);

    # # crop out
    # out = np.zeros_like(bgr_img) # Extract out the object and place into output image
    # out[mask == 255] = bgr_img[mask == 255];

    # if(plot):
    # # show
    #     cv2.imshow("Original", brg_img);
    #     cv2.imshow("thresh", thresh);
    #     cv2.imshow("Marked", marked);
    #     cv2.imshow("out", out);

    #     cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

    assert best_cntr is not None

    contour = np.squeeze(best_cntr)
    polygon = Polygon(contour)
    # print(polygon.wkt)

    return polygon  # ,out

    # create outline of tissue sample


def get_geom(adata: AnnData, threshold: int = None, inplace: bool = True, res: str = "hires") -> AnnData:

    if "geom" not in adata.uns["spatial"]:
        adata.uns["spatial"]["geom"] = {}

    # add spot points to geom
    # Create a geometry column from x & ly
    scale = utl.get_scale(adata, res=res)
    spot_diam = adata.uns["spatial"]["scale"]["spot_diameter_fullres"]

    adata.obs["spot_poly"] = adata.obs.apply(
        lambda x: Point(float(x.pxl_col_in_fullres * scale), float(x.pxl_row_in_fullres * scale)).buffer((spot_diam / 2) * 0.2),  # type: ignore
        axis=1,
    )

    # Create a GeoDataFrame from adata.obs
    adata.obs = gpd.GeoDataFrame(adata.obs, geometry=adata.obs.spot_poly)  # type: ignore

    # add boundary and tissue poly to geom
    tissue_poly = get_tissue_boundary(adata, threshold)
    adata.uns["spatial"]["geom"]["tissue_poly"] = tissue_poly
    adata.uns["spatial"]["geom"]["tissue_boundary"] = gpd.GeoSeries(tissue_poly).boundary

    return adata


def get_spot_coords(
    adata: AnnData,
    tissue: bool = True,
    as_tuple: bool = True,
    as_df: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], "pd.DataFrame"]:

    h_sc = utl.get_scale(adata)
    cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    if tissue:
        coords = adata.obs.loc[adata.obs["in_tissue"] == 1, cols] * h_sc
    else:
        coords = adata.obs.loc[:, cols] * h_sc

    if as_df:
        return coords
    coords = coords.values
    # if utl.is_highres(adata):
    #     h_sc = adata.uns["spatial"]["scale"]["tissue_hires_scalef"]
    # else:
    #     h_sc = adata.uns["spatial"]["scale"]["tissue_lowes_scalef"]
    # if tissue:
    #     return np.array(
    #         [h_sc * adata.obs[adata.obs["in_tissue"] == 1].iloc[:, 4], h_sc * adata.obs[adata.obs["in_tissue"] == 1].iloc[:, 3]]
    #     )
    # else:
    #     return np.array(h_sc * adata.obs.iloc[:, 4]), np.array(h_sc * adata.obs.iloc[:, 3])

    return (coords[:, 0], coords[:, 1]) if as_tuple else coords


def apply_rotation(
    adata: AnnData,
    k: Optional[int] = None,
    pxl_col_name: str = "pxl_col_in_fullres",
    pxl_row_name: str = "pxl_row_in_fullres",
    res: str = "all",
    purge: bool = False,
) -> bool:

    res_vals = ("lowres", "hires", "all")
    assert res in res_vals
    res_vals = res_vals[:2] if res == "all" else (res,)

    k_vals = (k % 4,) if k else tuple(range(4))
    obs_cols = adata.obs.columns

    for k in k_vals:
        pxl_rotnames = [f"pxl_col_in_fullres_rot{k}", f"pxl_row_in_fullres_rot{k}"]

        if pxl_rotnames[0] in obs_cols and pxl_rotnames[1] in obs_cols:
            adata.obs[pxl_col_name] = adata.obs[pxl_rotnames[0]]
            adata.obs[pxl_row_name] = adata.obs[pxl_rotnames[1]]
            adata.obs.drop(pxl_rotnames, axis=1, inplace=True)
        else:
            continue

        img_rot_existed = False
        rot_dict = adata.uns["spatial"].get("rotation", {})
        for res in res_vals:
            img_name = f"{res}_rot{k}"
            if img_name in adata.uns["spatial"]["img"]:
                img_rot_existed = True
                adata.uns["spatial"]["img"][res] = adata.uns["spatial"]["img"][img_name]
                rot_dict[res] = (rot_dict.get(res, 0) + k * 90) % 360
                del adata.uns["spatial"]["img"][img_name]

        if img_rot_existed:
            if purge:
                cancel_rotation(adata)
            return True
    return False


def cancel_rotation(adata: AnnData, k: Union[None, int, Iterable[int]] = None, res: str = "all") -> None:
    """Cancel an unapplied rotation of the tissue image and coordinates.
    Effectively, it removes any added columns to adata.obs, and the rotated image.

    Parameters
    ----------
    adata : AnnData
        The AnnData object which to cancel rotation for.
    k : Union[None, int, Iterable[int]], optional
        Cancel unapplied rotations created with k. If None, all rotations are cancelled, by default None
    res : str, optional
        The resolution of images to cancel rotation for, by default "all"
    """
    res_vals = ("lowres", "hires", "all")
    assert res in res_vals
    res_vals = res_vals[:2] if res == "all" else (res,)

    k_vals = [k] if isinstance(k, int) else list(range(4) if k is None else k)
    k_vals = [k % 4 for k in k_vals]

    for k in k_vals:
        rotnames = [f"pxl_col_in_fullres_rot{k}", f"pxl_row_in_fullres_rot{k}"]
        adata.obs.drop(rotnames, axis=1, inplace=True, errors="ignore")
        img_names = [f"{res}_rot{k}" for res in res_vals]
        for img in img_names:
            adata.uns["spatial"]["img"].pop(img, None)


def rotate_img90(adata: AnnData, k: int = 1, apply: bool = True, res: str = "all") -> bool:
    """Rotate the tissue image and the coordinates of the spots by k*90 degrees. If apply is True,
    then adata.uns['spatial']['rotation'][res] will contain the degrees between the original image (and coordinates)
    and the rotated version.

    Parameters
    ----------
    adata : AnnData
        The AnnData whose image and spot coordinates are to be rotated.
    k : int, optional
        The number of times the image should rotated by 90 degrees by default 1
    apply : bool, optional
        Whether to apply the rotation to the image and coordinates, by default True. If False, the
        rotated image will be stored under adata.uns['spatial']['img'] with a key "{res}_rot{k}" for all
        resolutions `res` that exist. The rotated coordinates are stored under adata.uns with keys
        "pxl_col_in_fullres_rot{k}" and "pxl_row_in_fullres_rot{k}" if `apply` is False.

    res : str, optional
        One of 'lowres', 'hires', 'all', the resolution to rotatae, by default "all". If "all", all existing resolutions of the
        image are rotated.

    Returns
    -------
    bool
        True if any image was rotated. False if no image with resolution `res` exists.
    """
    rotation_mats = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]]),
    ]

    k = k % 4

    rot = rotation_mats[k]
    res_keys = adata.uns["spatial"]["img"].keys()
    res_vals = ("lowres", "hires") if res == "all" else (res,)

    def rotator(res):
        img = adata.uns["spatial"]["img"][res]
        img = np.rot90(img, k=k)
        n_rows, n_cols, _ = img.shape

        # Rotate all spot coordinates
        coords = get_spot_coords(adata, tissue=False, as_tuple=False)
        center = np.array([n_rows, n_cols]) / 2

        # We rotate around the center of the image: translate to Origin > rotate > translate back
        coords = np.matmul(coords - center, rot)
        # If k is odd, then the center of the image is transposed (x, y) -> (y, x)
        coords += center[::-1] if k % 2 else center

        return img, coords

    ret = False
    rot_dict = adata.uns["spatial"].get("rotation", {})
    for res in res_vals:
        if res in res_keys:
            img, coords = rotator(res)
            scale = utl.get_scale(adata, res)

            coords = (coords / scale).astype(int)
            col_pos, row_pos = coords[:, 0], coords[:, 1]

            if apply:
                adata.uns["spatial"]["img"][res] = img
                adata.obs["pxl_col_in_fullres"] = col_pos
                adata.obs["pxl_row_in_fullres"] = row_pos
                rot_dict[res] = (rot_dict.get(res, 0) + 90 * k) % 360
            else:
                adata.uns["spatial"]["img"][f"{res}_rot{k}"] = img
                adata.obs[f"pxl_col_in_fullres_rot{k}"] = col_pos
                adata.obs[f"pxl_row_in_fullres_rot{k}"] = row_pos
            ret = True
    adata.uns["spatial"]["rotation"] = rot_dict
    return ret


def mirror_img(adata: AnnData, axis: int, apply: bool = True, res: str = "all") -> bool:
    """Mirror the tissue image and spot coordinates in the direction given by axis.

    Parameters
    ----------
    adata : AnnData
        The AnnData whose spatial data should be mirrored.
    axis : int
        0 mirrors the data horizontally
        1 mirrors the data vertically
        2 mirrors the data in both directions
    apply : bool, optional
        Whether to apply the mirror, by default True
    res : str, optional
        One of 'lowres', 'hires', 'all', the resolution to mirror, by default "all". If "all", all existing resolutions of the
        image are mirrored.

    Returns
    -------
    bool
        True if any image was mirrored, False if no image with resolution `res` exists.
    """

    if axis not in range(3):
        raise ValueError("Invalid mirror axis, must be one of [0, 1, 2].")
    res_keys = adata.uns["spatial"]["img"].keys()
    res_vals = ("lowres", "hires") if res == "all" else (res,)

    def mirror(res: str, axis: int) -> Tuple[np.ndarray, np.ndarray]:
        img = adata.uns["spatial"]["img"][res]
        n_rows, n_cols, _ = img.shape

        # This returns (cols, rows)
        coords: np.ndarray = get_spot_coords(adata, tissue=False, as_tuple=False)  # type: ignore

        if axis % 2 == 0:
            # mirror cols around a vertical axis
            img = img[::-1, :, :]
            coords[:, 1] = n_rows - 1 - coords[:, 1]

        if axis > 0:
            # mirror rows around a horizontal axis
            img = img[:, ::-1, :]
            coords[:, 0] = n_cols - 1 - coords[:, 0]

        return img, coords

    ret = False
    mirror_dict = adata.uns["spatial"].get("mirror", {})
    for res in res_vals:
        if res not in res_keys:
            continue
        img, coords = mirror(res, axis)
        scale = utl.get_scale(adata, res)

        coords = (coords / scale).astype(int)
        col_pos, row_pos = coords[:, 0], coords[:, 1]

        if apply:
            adata.uns["spatial"]["img"][res] = img
            adata.obs["pxl_col_in_fullres"] = col_pos
            adata.obs["pxl_row_in_fullres"] = row_pos
            mirror_dict[res] = get_mirror_val(mirror_dict.get(res), axis)
        else:
            adata.uns["spatial"]["img"][f"{res}_mirror{axis}"] = img
            adata.obs[f"pxl_col_in_fullres_mirror{axis}"] = col_pos
            adata.obs[f"pxl_row_in_fullres_mirror{axis}"] = row_pos
        ret = True
    adata.uns["spatial"]["mirror"] = mirror_dict
    return ret


def apply_mirror(
    adata,
    axis: Optional[int] = None,
    res: str = "all",
    pxl_col_name: str = "pxl_col_in_fullres",
    pxl_row_name: str = "pxl_row_in_fullres",
    purge: bool = False,
) -> bool:

    res_vals = ("lowres", "hires", "all")
    assert res in res_vals
    res_vals = res_vals[:2] if res == "all" else (res,)

    axes = (axis % 3,) if axis else tuple(range(3))
    obs_cols = adata.obs.columns

    for ax in axes:
        pxl_mirrnames = [f"pxl_col_in_fullres_mirror{ax}", f"pxl_row_in_fullres_mirror{ax}"]

        if pxl_mirrnames[0] in obs_cols and pxl_mirrnames[1] in obs_cols:
            adata.obs[pxl_col_name] = adata.obs[pxl_mirrnames[0]]
            adata.obs[pxl_row_name] = adata.obs[pxl_mirrnames[1]]
            adata.obs.drop(pxl_mirrnames, axis=1, inplace=True)
        else:
            continue

        img_mirr_existed = False
        mirr_dict = adata.uns["spatial"].get("mirror", {})
        for res in res_vals:
            img_name = f"{res}_mirror{ax}"
            if img_name in adata.uns["spatial"]["img"]:
                img_mirr_existed = True
                adata.uns["spatial"]["img"][res] = adata.uns["spatial"]["img"][img_name]
                mirr_dict[res] = get_mirror_val(mirr_dict.get(res), ax)
                del adata.uns["spatial"]["img"][img_name]

        if img_mirr_existed:
            if purge:
                cancel_mirror(adata)
            return True
    return False


def cancel_mirror(
    adata: AnnData,
    axis: Union[None, int, Iterable[int]] = None,
    res: str = "all",
) -> None:
    """Cancel an unapplied mirroring of an image

    Parameters
    ----------
    adata : AnnData
        The AnnData object to cancel mirroring for
    axis : Union[None, int, Iterable[int]], optional
        The axis to cancel the mirroring for.
    res : str, optional
        The resolution of images to cancel mirroring for, by default "all"
    """
    res_vals = ("lowres", "hires", "all")
    assert res in res_vals
    res_vals = res_vals[:2] if res == "all" else (res,)
    axis = [axis] if isinstance(axis, int) else list(range(3) if axis is None else axis)

    for ax in axis:
        flipnames = [f"pxl_col_in_fullres_mirror{ax}", f"pxl_row_in_fullres_mirror{ax}"]
        adata.obs.drop(flipnames, axis=1, inplace=True, errors="ignore")
        img_names = [f"{res}_mirror{ax}" for res in res_vals]
        for img in img_names:
            adata.uns["spatial"]["img"].pop(img, None)


def cancel_transformations(
    adata: AnnData,
    axis: Union[None, int, Iterable[int]] = None,
    k: Union[None, int, Iterable[int]] = None,
    res: str = "all",
) -> None:
    """Cancel unapplied transformations of an image

    Parameters
    ----------
    adata : AnnData
        The AnnData object to cancel transformations for
    axis : Union[None, int, Iterable[int]], optional
        The axis to cancel mirroring for, by default None
    k : Union[None, int, Iterable[int]], optional
        The rotation to cancel rotation for, by default None
    res : str, optional
        The resolution of images to cancel transformationr for, by default "all"
    """
    cancel_mirror(adata, axis, res)
    cancel_rotation(adata, k, res)


def get_mirror_val(curr: Optional[int], ax: Optional[int]) -> Optional[int]:
    """Get the result of mirroring image by ax, if already mirrored by curr.

    The values can be None, 0, 1, or 2.
    * None means no mirror.
    * 0 means mirror along a horizontal axis
    * 1 means mirror along a vertical axis
    * 2 means mirror along a both axis

    Parameters
    ----------
    curr : Optional[int]
        One of None, 0, 1, 2. The current mirror condition. None means no mirror.
    ax : Optional[int]
        One of None, 0, 1, 2. The mirror axis. None means no mirroring.

    Returns
    -------
    Optional[int]
        One of None, 0, 1, 2. None means the effictive mirror is none.
    """
    if curr == ax:
        # ax cancels out
        return None

    # Change the values if None to fit the return statement
    ax = (3 - 2 * curr) if ax is None else ax  # type: ignore
    curr = (3 - 2 * ax) if curr is None else curr  # type: ignore

    # The function is symmetric. Only options we consider are (0, 1), (0, 2), and (1,2).
    return 3 - (curr + ax)

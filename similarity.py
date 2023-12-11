"""Functions to compute similarity between images, based on cluster, peaks, voxels
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

from nibabel import Nifti1Image

from scipy.spatial.distance import cdist

def bin_img(img):
    return image.math_img('img1.astype(bool)', img1=img)

def thr_img(path, thr):
    return image.math_img('img > %s' % thr, img=image.load_img(path))

def percent_overlap(path_img1, path_img2, thr_img1, thr_img2, mask=None):
    """Voxel-wise percent overlap
    (Jaccard and Dice similarity coefficient)
    Based on Maitra 2009: https://doi.org/10.1016/j.neuroimage.2009.11.070

    Parameters
    ----------
    path_img1 : str
        (Path to) first image
    path_img2 : str
        (Path to) second image
    mask : str
        Path to GM mask to constrain true negative calculation to

    Returns
    -------
    jaccard : float
        The resulting Jaccard similarity coefficient
    dice : float
        Additional Dice index
    sensitivity : float
        Sensitivity based on img2 is all positive (TP, FN)
    specificity : float
        Specificity
    precision : float
        Precision
    v_t : int 
        Number of sign. voxels of img1
    v_j : int
        Number of sign. voxel of img2
    v_jt : int
        Number of sign. voxel overlapping between img1 and img2
    """

    if isinstance(path_img1, Nifti1Image):
        img1 = path_img1
    else:
        img1 = nib.load(path_img1)

    if isinstance(path_img2, Nifti1Image):
        img2 = path_img2
    else:
        img2 = nib.load(path_img2)

    # resample to img1 if unequal shape
    if img1.shape != img2.shape:
        print('unequal shapes... resampling of image 2')
        img2 = image.resample_to_img(img2, img1, interpolation='nearest')
    
    # threshold and binarise
    img1 = thr_img(path=img1, thr=thr_img1)
    img2 = thr_img(path=img2, thr=thr_img2)
    img1 = bin_img(img=img1)
    img2 = bin_img(img=img2)

    # nifti as numpy.arrays
    map1 = img1.get_fdata()
    map2 = img2.get_fdata()

    # reduce to 3d (4th dim. will be deleted)
    if map1.ndim == 4:
        map1 = map1[:, :, :, 0]
    if map2.ndim == 4:
        map2 = map2[:, :, :, 0]

    # sum map: know which voxels are in both (1+1=2)
    sum_map = map1 + map2

    # union of activated voxels
    v_jt = np.count_nonzero(sum_map == 2)
    # number of activated voxels in map1
    v_t = np.count_nonzero(map1 == 1)
    # number of activated voxels in map2
    v_j = np.count_nonzero(map2 == 1)

    precision = v_jt / v_t

    if mask is None:
        specificity = 'No GM mask defined'
        sensitivity = v_jt / v_j
    else:
        mask = nib.load(mask)
        if mask.shape != img1.shape:
            print('unequal shapes... resampling of mask')
            mask = image.resample_to_img(mask, img1, interpolation='nearest')
        mask = bin_img(img=mask)
        map_mask = mask.get_fdata()
        if map_mask.ndim == 4:
            map_mask = map_mask[:, :, :, 0]
        mask_n_sum = map_mask + sum_map
        mask_n_map2 = map_mask + map2
        v_tn = np.count_nonzero(mask_n_sum == 1)
        v_n = np.count_nonzero(mask_n_map2 == 1)
        v_p = np.count_nonzero(mask_n_map2 == 2)
        sensitivity = v_jt / v_p
    jaccard = v_jt / (v_j + v_t - v_jt)
    return jaccard, sensitivity, precision, v_t, v_j, v_jt


def calculate_voxel_similarity(img1, img2, gm_mask, thr_img1=0.000001, thr_img2=0.000001):
    jaccard, sensitivity, precision, n_vox1, n_vox2, n_voxolp = percent_overlap(
        path_img1=img1, path_img2=img2, mask=gm_mask, thr_img1=thr_img1, thr_img2=thr_img2)
    vox_comp = {
        'jaccard': round(jaccard, 3),
        'sensitivity': round(sensitivity, 3),
        'precision': round(precision, 3),
        'voxel_map1': n_vox1,
        'voxel_map2': n_vox2,
        'voxel_overlap': n_voxolp}
    return vox_comp


def mean_distance_to_next_peak(coords_img1, coords_img2, max_dist=None, mean_or_median='mean'):
    """Calculates the mean distance between peaks of img1 and img2

    Parameters
    ----------
    coords_img1 : np.array
        The coordinates (peaks) of first nii in MNI space (mm)
    coords_img2 : np.array
        The coordinates (peaks) of second nii in MNI space (mm)
    max_dist : int
        Maximum distance between peaks to be considered
    mean_or_median : str
        Whether to return mean or median distance
        
    Returns
    -------
    mean_distance : float
        Mean distance between peaks of img1 and img2
    """
    dist_matrix = cdist(coords_img1, coords_img2, 'euclidean')
    # find shortest distance (i.e., to the next peak)
    dist_matrix_img1 = np.min(dist_matrix, axis=1)
    dist_matrix_img2 = np.min(dist_matrix, axis=0)

    if max_dist is not None:
        dist_matrix_img1 = np.where(dist_matrix_img1 > max_dist, np.nan, dist_matrix_img1)
        dist_matrix_img2 = np.where(dist_matrix_img2 > max_dist, np.nan, dist_matrix_img2)
    if mean_or_median == 'mean':
        mean_distance_img1 = np.nanmean(dist_matrix_img1)
        mean_distance_img2 = np.nanmean(dist_matrix_img2)
    elif mean_or_median == 'median':
        mean_distance_img1 = np.nanmedian(dist_matrix_img1)
        mean_distance_img2 = np.nanmedian(dist_matrix_img2)
    return mean_distance_img1, mean_distance_img2


def calculate_peak_similarity(img1, img2, max_dist=None, mean_or_median='median'):
    lmax_img2 = img2[['x', 'y', 'z']].to_numpy().astype(float)
    lmax_img1 = img1[['x', 'y', 'z']].to_numpy().astype(float)
    mean_dist = mean_distance_to_next_peak(coords_img1=lmax_img1, coords_img2=lmax_img2, max_dist=max_dist, mean_or_median=mean_or_median)
    return mean_dist


def calculate_clus_similarity(img1, img2):
    """Cluster (blob) wise comparison of two images

    Parameters
    ----------
    img1 : str
        path to first cluster indexed image
    img2 : str
        path to second cluster indexed image

    Returns
    -------
    cluster_comp : dict
        num_clus1 : int
            Number of cluster in image 1
        olp_img1 : int
            Number of overlapping cluster (minimum 1 voxel) of image 1 with blobs of image 2
        num_clus2 : int
            Number of cluster in image 2
        olp_img2 : int
            Number of overlapping cluster (minimum 1 voxel) of image 2 with blobs of image 1
    """
    if isinstance(img1, str):
        img1 = nib.load(img1)
    if isinstance(img2, str):
        img2 = nib.load(img2)
    img_1 = img1.get_fdata().astype(int)
    img_2 = img2.get_fdata().astype(int)

    num_clus1 = np.max(img_1)
    num_clus2 = np.max(img_2)
    count_clus1 = np.array([])
    for id_clus in range(1, num_clus2 + 1):
        clus = img_2 == id_clus
        match_clus = clus * img_1
        # count cluster of img1 overlapping with cluster of img2
        count_clus1 = np.append(count_clus1, np.unique(match_clus))
    olp_img1 = np.count_nonzero(np.unique([count_clus1]))

    count_clus2 = np.array([])
    for id_clus in range(1, num_clus1 + 1):
        clus = img_1 == id_clus
        match_clus = clus * img_2
        # count cluster of img2 overlapping with cluster of img1
        count_clus2 = np.append(count_clus2, np.unique(match_clus))
    olp_img2 = np.count_nonzero(np.unique([count_clus2]))
    cluster_comp = {'num_clus1': num_clus1, 'olp_img1': olp_img1,
                    'num_clus2': num_clus2, 'olp_img2': olp_img2}
    return cluster_comp

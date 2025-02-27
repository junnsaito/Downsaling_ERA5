#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:42:44 2025

@author: junsaito
"""

# velocity.py
import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import geopandas as gpd

def load_velocity_components(vx_path, vy_path):
    """
    速度データ (vx, vy) を読み込み、変換情報、スケール情報（dx, dy）、およびCRSを返す
    """
    with rasterio.open(vx_path) as src_vx:
        Vx = src_vx.read(1)
        transform = src_vx.transform
        crs = src_vx.crs
    with rasterio.open(vy_path) as src_vy:
        Vy = src_vy.read(1)
    # dx, dy の抽出（dyは負の値の場合があるので符号反転）
    dx = transform[0]
    dy = -transform[4]
    return Vx, Vy, dx, dy, transform, crs

def load_uncertainty_maps(vx_error_path, vy_error_path):
    """
    vx, vy の不確実性マップを読み込み、各ピクセルで大きい方を返す
    """
    with rasterio.open(vx_error_path) as src_vx_err:
        Vx_error = src_vx_err.read(1)
    with rasterio.open(vy_error_path) as src_vy_err:
        Vy_error = src_vy_err.read(1)
    return np.maximum(Vx_error, Vy_error)

def apply_variable_gaussian_filter(Vx, Vy, uncertainty_map, min_sigma=1, max_sigma=6):
    """
    不確実性に応じたガウシアンフィルタによる平滑化
    """
    uncertainty_map_resized = resize(uncertainty_map, Vx.shape, mode='reflect', anti_aliasing=True)
    sigma_values = np.linspace(min_sigma, max_sigma, num=5)
    Vx_smooth = Vx.copy()
    Vy_smooth = Vy.copy()
    for i, sigma in enumerate(sigma_values):
        mask = (uncertainty_map_resized >= i/5) & (uncertainty_map_resized < (i+1)/5)
        if np.any(mask):
            Vx_smooth[mask] = gaussian_filter(Vx, sigma=sigma)[mask]
            Vy_smooth[mask] = gaussian_filter(Vy, sigma=sigma)[mask]
    return Vx_smooth, Vy_smooth

def compute_strain_rate(Vx, Vy, dx, dy):
    """
    vx, vy から速度勾配を計算し、変形率（eps_xx, eps_yy, eps_xy）、発散、および eps_zz を返す
    """
    dVx_dx = np.gradient(Vx, dx, axis=1)
    dVx_dy = np.gradient(Vx, dy, axis=0)
    dVy_dx = np.gradient(Vy, dx, axis=1)
    dVy_dy = np.gradient(Vy, dy, axis=0)
    eps_xx = dVx_dx
    eps_yy = dVy_dy
    eps_xy = 0.5 * (dVx_dy + dVy_dx)
    divergence = dVx_dx + dVy_dy
    eps_zz = -(eps_xx + eps_yy)
    return eps_xx, eps_yy, eps_xy, divergence, eps_zz

def compute_effective_horizontal_strain_rate(eps_xx, eps_yy, eps_xy, max_value=2):
    """
    有効水平変形率 eps_h を計算する。eps_h > max_value の場合は nan とする。
    """
    eps_h = np.sqrt(0.5 * (eps_xx**2 + eps_yy**2) + eps_xy**2)
    eps_h[eps_h > max_value] = np.nan
    return eps_h

def load_shapefile(shapefile_path, crs):
    """
    shapefile を読み込み、必要に応じて指定CRSに変換して返す
    """
    shape = gpd.read_file(shapefile_path)
    if crs is not None and shape.crs != crs:
        shape = shape.to_crs(crs)
    return shape

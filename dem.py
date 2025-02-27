#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:41:13 2025

@author: junsaito
"""

# dem.py
import rasterio
import numpy as np
from scipy.ndimage import sobel, laplace

def load_dem(dem_path):
    """
    DEMファイルを読み込み、配列、変換情報、CRSを返す
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
    return dem_data, transform, crs

def compute_slope_and_aspect(dem, dx):
    """
    DEMから斜面角（度）と方位角（度）を計算する
    """
    dzdx = sobel(dem, axis=1) / (8.0 * dx)
    dzdy = sobel(dem, axis=0) / (8.0 * dx)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)
    aspect = np.arctan2(-dzdy, dzdx) * (180 / np.pi)
    aspect[aspect < 0] += 360
    return slope, aspect

def compute_curvature(dem, dx):
    """
    DEMからラプラシアンを用いて曲率を計算する
    """
    return laplace(dem) / (dx**2)



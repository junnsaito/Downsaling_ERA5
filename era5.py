#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:41:26 2025

@author: junsaito
"""

# era5.py
import xarray as xr
import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import reproject, Resampling

def load_era5_dataset(file_path):
    """
    ERA5データセットを読み込み、"time" 座標を "valid_time" にリネームして返す
    """
    ds = xr.open_dataset(file_path)
    if "time" in ds.coords:
        ds = ds.rename({"time": "valid_time"})
    return ds

def reproject_to_rema(data, src_crs, dst_crs, src_transform, dst_transform, width, height):
    """
    入力データをDEMグリッド（dst_transform, width, height）に再投影する
    """
    out = np.empty((height, width), dtype=np.float32)
    reproject(
        source=data.astype(np.float32),
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.cubic
    )
    return out

def adjust_temperature_with_station(era5_temp, grid_elev, station_elev, date, station_data, lapse_rate, z0=0.0):
    """
    ERA5温度補正
      T_corrected = ERA5_t2m + [(T_obs - ERA5_station) - |lapse_rate| * (grid_elev - station_elev)]
    """
    T_obs = get_station_temperature(date, station_data) + 273.15
    correct0 = T_obs - abs(lapse_rate) * (z0 - station_elev)
    correct1 = correct0 - abs(lapse_rate) * (grid_elev - z0)
    return correct1

def get_station_temperature(date, station_data):
    time_diff = (station_data["Date"] - date).abs()
    idx = time_diff.idxmin()
    return station_data.loc[idx, "AT"]

def adjust_dewpoint_with_station(era5_dew, grid_elev, station_elev, date, station_data, dew_lapse_rate, z0=0.0):
    """
    ERA5露点温度補正
      T_dew_corrected = ERA5_d2m + [(DPT_obs - ERA5_d2m_station) - |dew_lapse_rate| * (grid_elev - station_elev)]
    """
    station_dew = get_station_dewpoint(date, station_data) + 273.15
    correct0 = station_dew - abs(dew_lapse_rate) * (z0 - station_elev)
    correct1 = correct0 - abs(dew_lapse_rate) * (grid_elev - z0)
    return correct1

def get_station_dewpoint(date, station_data):
    time_diff = (station_data["Date"] - date).abs()
    idx = time_diff.idxmin()
    return station_data.loc[idx, "DT"]



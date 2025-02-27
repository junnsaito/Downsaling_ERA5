#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:44:09 2025

@author: junsaito
"""

# parallel_processing.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from tqdm import tqdm
from era5 import reproject_to_rema, adjust_temperature_with_station, adjust_dewpoint_with_station
from fluxes import (compute_specific_humidity, calc_esat, downscale_pressure,
                    downscale_longwave, calc_snow_ratio, compute_rel_humidity, calc_flux_bulk_iter)
import numpy as np
import pandas as pd

def process_time_step(i, ds_year, src_transform_era5, rema_transform,
                      W_dem, H_dem, i0, j0, lapse_value, lapse_value2,
                      lapse_value3, delta_z_point, slope2d, aspect2d,
                      curvature2d, solpos, modis_albedo_value, rema_dem_data,
                      time_interval_s, station_data, station_data2, station_elev):
    # 各ERA5変数の2次元配列を抽出
    t2m_2d = ds_year["t2m"].isel(valid_time=i).values
    skt_2d = ds_year["skt"].isel(valid_time=i).values
    tp_2d = ds_year["tp"].isel(valid_time=i).values
    ssrd_2d = ds_year["ssrd"].isel(valid_time=i).values
    strd_2d = ds_year["strd"].isel(valid_time=i).values
    sp_2d = ds_year["sp"].isel(valid_time=i).values
    u10_2d = ds_year["u10"].isel(valid_time=i).values
    v10_2d = ds_year["v10"].isel(valid_time=i).values
    d2m_2d = ds_year["d2m"].isel(valid_time=i).values
    fsr_2d = ds_year["fsr"].isel(valid_time=i).values

    src_crs = "EPSG:4326"
    dst_crs = "EPSG:4326"  # この例では再投影後はDEMのグリッドサイズ（500m）と同じ前提

    t2m_reproj = reproject_to_rema(t2m_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    skt_reproj = reproject_to_rema(skt_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    tp_reproj = reproject_to_rema(tp_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    ssrd_reproj = reproject_to_rema(ssrd_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    strd_reproj = reproject_to_rema(strd_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    sp_reproj = reproject_to_rema(sp_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    u10_reproj = reproject_to_rema(u10_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    v10_reproj = reproject_to_rema(v10_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    d2m_reproj = reproject_to_rema(d2m_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    fsr_reproj = reproject_to_rema(fsr_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)

    current_date = pd.Timestamp(ds_year.valid_time.values[i])
    grid_elev = rema_dem_data[i0, j0]
    T_adjusted = adjust_temperature_with_station(t2m_reproj[i0, j0], grid_elev, station_elev,
                                                  current_date, station_data, lapse_value)
    temp_corr = T_adjusted
    T_adjusted2 = adjust_dewpoint_with_station(d2m_reproj[i0, j0], grid_elev, station_elev,
                                                current_date, station_data2, lapse_value2)
    dt_corr = T_adjusted2

    tskin_corr = skt_reproj[i0, j0] + lapse_value3 * delta_z_point

    pressure_corr = sp_reproj[i0, j0]
    qv_corr = compute_specific_humidity(pressure_corr, dt_corr)
    wind_speed_raw = np.sqrt(u10_reproj[i0, j0]**2 + v10_reproj[i0, j0]**2)
    wind_speed_corr = wind_speed_raw

    RH_val = compute_rel_humidity(temp_corr, dt_corr) * 100
    s_rate = 0.5  # 固定値またはcalc_snow_ratio()で算出
    T0_val = temp_corr - 1.0
    ez_val = calc_esat(dt_corr)
    e0_val = calc_esat(T0_val)
    QH_val, QE_val = calc_flux_bulk_iter(temp_corr, ez_val, wind_speed_corr, T0_val, e0_val, pz=pressure_corr)

    tp_mm = tp_reproj[i0, j0] * 1000.0
    tp_mm_adj = tp_mm  # 降水補正が必要ならここで関数を呼ぶ
    rain_mm = tp_mm_adj * (1 - s_rate)
    snow_mm = tp_mm_adj * s_rate
    RAIN_val = rain_mm
    BDOT_val = rain_mm + snow_mm

    sw_val = ssrd_reproj[i0, j0] / time_interval_s
    lw_val = strd_reproj[i0, j0] / time_interval_s
    zen_val = solpos["zenith"].values[0]
    azi_val = solpos["azimuth"].values[0]
    sw_corr = sw_val
    lw_corr = lw_val
    ALBEDO_val = modis_albedo_value

    rho_new_val = 100.0
    rho_redeposit_val = 50.0

    time_val = pd.Timestamp(ds_year.valid_time.values[i])
    return (sw_corr, lw_corr, temp_corr, tskin_corr, QH_val, QE_val,
            RAIN_val, BDOT_val, time_val, ALBEDO_val, rho_new_val, rho_redeposit_val)

def process_time_steps_chunk(time_indices, ds_year, src_transform_era5, rema_transform,
                             W_dem, H_dem, i0, j0, lapse_value, lapse_value2,
                             lapse_value3, delta_z_point, slope2d, aspect2d,
                             curvature2d, solpos, modis_albedo_value, rema_dem_data,
                             time_interval_s, station_data, station_data2, station_elev):
    results = []
    for i in time_indices:
        results.append(process_time_step(i, ds_year, src_transform_era5, rema_transform,
                                         W_dem, H_dem, i0, j0, lapse_value, lapse_value2,
                                         lapse_value3, delta_z_point, slope2d, aspect2d,
                                         curvature2d, solpos, modis_albedo_value, rema_dem_data,
                                         time_interval_s, station_data, station_data2, station_elev))
    return results

def task(i, ds_year, src_transform_era5, rema_transform, W_dem, H_dem,
         i0, j0, lapse_value, lapse_value2, lapse_value3, delta_z_point,
         slope2d, aspect2d, curvature2d, solpos, modis_albedo_value,
         shm_name, shape, dtype, time_interval_s, station_data, station_data2, station_elev):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    rema_dem_data_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = process_time_step(i, ds_year, src_transform_era5, rema_transform,
                               W_dem, H_dem, i0, j0, lapse_value, lapse_value2, lapse_value3,
                               delta_z_point, slope2d, aspect2d, curvature2d, solpos,
                               modis_albedo_value, rema_dem_data_shared, time_interval_s,
                               station_data, station_data2, station_elev)
    existing_shm.close()
    return result

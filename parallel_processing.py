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
import numpy as np
import pandas as pd
# ERA5再投影＆補正関連
from era5 import reproject_to_rema, adjust_temperature_with_station, adjust_dewpoint_with_station
# fluxes関連
from fluxes import (
    compute_specific_humidity, calc_esat, downscale_pressure,
    downscale_longwave, calc_snow_ratio, compute_rel_humidity,
    calc_flux_bulk_iter,apply_topo_correction,calc_new_snow_density,calc_redeposit_density,adjust_wind_speed
)


def process_time_step(
    i, 
    ds_year,
    src_transform_era5,
    rema_transform,
    W_dem, H_dem,
    i0, j0,
    lapse_value,        # t2mラプスレート
    lapse_value2,       # d2mラプスレート
    lapse_value3,       # sktラプスレート
    delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array,
    modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev
):
    """
    1タイムステップ (i番目) のERA5データを再投影し、温度や降水などを補正計算して返す。
    
    戻り値: (sw_corr, lw_corr, temp_corr, tskin_corr, QH_val, QE_val,
             RAIN_val, BDOT_val, time_val, ALBEDO_val, rho_new_val, rho_redeposit_val)
    """

    # ---------------------------
    # 1) ds_year から i番目の2次元フィールドを取り出す
    # ---------------------------
    t2m_2d  = ds_year["t2m"].isel(valid_time=i).values
    skt_2d  = ds_year["skt"].isel(valid_time=i).values
    tp_2d   = ds_year["tp"].isel(valid_time=i).values
    ssrd_2d = ds_year["ssrd"].isel(valid_time=i).values
    strd_2d = ds_year["strd"].isel(valid_time=i).values
    sp_2d   = ds_year["sp"].isel(valid_time=i).values
    u10_2d  = ds_year["u10"].isel(valid_time=i).values
    v10_2d  = ds_year["v10"].isel(valid_time=i).values
    d2m_2d  = ds_year["d2m"].isel(valid_time=i).values
    fsr_2d  = ds_year["fsr"].isel(valid_time=i).values  # 例: fsr (optional変数)

    # ---------------------------
    # 2) ERA5をDEM座標系に再投影 (reproject_to_rema)
    # ---------------------------
    src_crs = "EPSG:4326"
    dst_crs = "EPSG:4326"  # この例では同じEPSG:4326にしているが、実際にはDEMに合わせる想定

    t2m_re  = reproject_to_rema(t2m_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    skt_re  = reproject_to_rema(skt_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    tp_re   = reproject_to_rema(tp_2d,   src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    ssrd_re = reproject_to_rema(ssrd_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    strd_re = reproject_to_rema(strd_2d, src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    sp_re   = reproject_to_rema(sp_2d,   src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    u10_re  = reproject_to_rema(u10_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    v10_re  = reproject_to_rema(v10_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    d2m_re  = reproject_to_rema(d2m_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)
    fsr_re  = reproject_to_rema(fsr_2d,  src_crs, dst_crs, src_transform_era5, rema_transform, W_dem, H_dem)

    # ---------------------------
    # 3) DEMインデックス (i0, j0) の値を取り出す & 観測所補正など
    # ---------------------------
    current_date = pd.Timestamp(ds_year.valid_time.values[i])
    grid_elev = rema_dem_data[i0, j0]

    # 例: adjust_temperature_with_station(...) で実測補正＋ラプスレート補正
    t2m_val  = t2m_re[i0, j0]
    T_adjusted = adjust_temperature_with_station(
        t2m_val,
        grid_elev,
        station_elev,
        current_date,
        station_data,
        lapse_value
    )
    temp_corr = T_adjusted

    # 例: 露点 (d2m)
    d2m_val = d2m_re[i0, j0]
    T_adjusted2 = adjust_dewpoint_with_station(
        d2m_val,
        grid_elev,
        station_elev,
        current_date,
        station_data2,
        lapse_value2
    )
    dt_corr = T_adjusted2

    # skt
    skt_val = skt_re[i0, j0]
    tskin_corr = skt_val + lapse_value3 * delta_z_point

    # ---------------------------
    # 4) 風速, 気圧, 湿度, etc.
    # ---------------------------
    sp_val = sp_re[i0, j0]
    pressure_corr = downscale_pressure(sp_re[i0, j0], t2m_re[i0, j0], temp_corr, delta_z_point)
    
    pressure_corr = downscale_pressure(sp_re[i0, j0], t2m_re[i0, j0], temp_corr, delta_z_point)
    qv_corr = compute_specific_humidity(pressure_corr, dt_corr)

    u10_val = u10_re[i0, j0]
    v10_val = v10_re[i0, j0]
    wind_speed_raw = np.sqrt(u10_val**2 + v10_val**2)
    wind_speed_corr = adjust_wind_speed(wind_speed_raw, u10_re[i0, j0], v10_re[i0, j0],
                                          slope2d[i0, j0], curvature2d[i0, j0],
                                          delta_z_point, fsr_re[i0, j0], d=0.1, measurement_height=10.0)

    # 相対湿度など
    RH_val = compute_rel_humidity(temp_corr, dt_corr) * 100

    # ---------------------------
    # 5) フラックス (QH, QE) 計算 (例: calc_flux_bulk_iter)
    # ---------------------------
    T0_val = temp_corr - 1.0
    ez_val = calc_esat(dt_corr)
    e0_val = calc_esat(T0_val)
    QH_val, QE_val = calc_flux_bulk_iter(
        temp_corr, ez_val,
        wind_speed_corr,
        T0_val, e0_val,
        pz=pressure_corr
    )

    # ---------------------------
    # 6) 降水, RAIN, BDOT (例: 雨 + 雪)
    # ---------------------------
    tp_val = tp_re[i0, j0] * 1000.0  # mm
    s_rate = calc_snow_ratio(pressure_corr, qv_corr, temp_corr)
    rain_mm = tp_val * (1 - s_rate)
    snow_mm = tp_val * s_rate
    RAIN_val = rain_mm
    BDOT_val = rain_mm + snow_mm

    # ---------------------------
    # 7) 短波・長波 (ssrd, strd)
    # ---------------------------
    ssrd_val = ssrd_re[i0, j0]
    strd_val = strd_re[i0, j0]

    sw_val = ssrd_val / time_interval_s
    lw_val = strd_val / time_interval_s
    
    
    
    sw_val = ssrd_re[i0, j0] / time_interval_s
    lw_val = strd_re[i0, j0] / time_interval_s
    zen_val = zenith_array[i]
    azi_val = azimuth_array[i]

    # zen_val = solpos["zenith"].values[0]
    # azi_val = solpos["azimuth"].values[0]
    sw_corr = apply_topo_correction(sw_val, zen_val, azi_val, slope2d[i0, j0],
                                    aspect2d[i0, j0], rema_dem_data, dx=500, i0=i0, j0=j0)
    lw_corr = downscale_longwave(strd_re[i0, j0] / time_interval_s, t2m_re[i0, j0], temp_corr)
    
    
    # アルベド
    ALBEDO_val = modis_albedo_value

    # ---------------------------
    # 8) 雪密度など (rho_new, 再堆積など)
    # ---------------------------
    # rho_new_val = 100.0       # 仮の雪密度
    # rho_redeposit_val = 50.0  # 仮の再堆積密度
    rho_new_val = calc_new_snow_density(temp_corr, tskin_corr, RH_val, wind_speed_corr)
    rho_redeposit_val = calc_redeposit_density(wind_speed_corr)

    # 時刻
    time_val = current_date

    # ---------------------------
    # 9) 戻り値 (12要素)
    # ---------------------------
    return (
        sw_corr,
        lw_corr,
        temp_corr,
        tskin_corr,
        QH_val,
        QE_val,
        RAIN_val,
        BDOT_val,
        time_val,
        ALBEDO_val,
        rho_new_val,
        rho_redeposit_val,
        
    )


def process_time_steps_chunk(
    time_indices, ds_year,
    src_transform_era5, rema_transform,
    W_dem, H_dem, i0, j0,
    lapse_value, lapse_value2, lapse_value3, delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array, modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev
):
    """
    time_indices (例: 50件) をまとめて処理。
    返り値: [process_time_step()の結果, ...] のリスト
    """
    results = []
    for i in time_indices:
        row_result = process_time_step(
    i, ds_year,
    src_transform_era5, rema_transform,
    W_dem, H_dem, i0, j0,
    lapse_value, lapse_value2, lapse_value3, delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array, modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev
        )
        results.append(row_result )
    return results


def task(
    i, ds_year,
    src_transform_era5, rema_transform,
    W_dem, H_dem, i0, j0,
    lapse_value, lapse_value2, lapse_value3, delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array, modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev
):
    """
    共有メモリを使って、1つのタイムステップ (i) の計算をする例 (より細分化した実装が必要な場合)
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    rema_dem_data_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    result = process_time_step(
    i, ds_year,
    src_transform_era5, rema_transform,
    W_dem, H_dem, i0, j0,
    lapse_value, lapse_value2, lapse_value3, delta_z_point,
    slope2d, aspect2d, curvature2d,
    zenith_array, azimuth_array, modis_albedo_value,
    rema_dem_data,
    time_interval_s,
    station_data,
    station_data2,
    station_elev
    )

    existing_shm.close()
    return result

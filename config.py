#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:31:58 2025

@author: junsaito
"""

# config.py
# グローバル設定

# 複数のターゲット座標が記載されたCSVファイルのパス
target_coords_csv = "/media/junsaito/E4308E2C308E05B0/JS_model/target_coords.csv"  # CSV例：列 "lon" と "lat"

rema_dem_file = "/media/junsaito/E4308E2C308E05B0/Antarctica/REMA/rema_mosaic_500m_v2.0_dem_station_wgs84.tif"
era5_base_dir = "/media/junsaito/E4308E2C308E05B0/Antarctica/ERA5"
modis_wgs84_file = "/media/junsaito/E4308E2C308E05B0/Antarctica/MODIS_QA_latest/MODIS_Albedo_2000-2024_avg_wgs84.tif"
output_dir = "/media/junsaito/E4308E2C308E05B0/JS_model/Output"

z0_station = 0
kappa_value = 0.35
time_interval_s = 3600  # 1時間 = 3600秒

# 月別ラプスレート（単位: °C/km）
monthly_beta = {
    1: 7.44, 2: 8.70, 3: 9.59, 4: 9.68, 5: 10.00, 6: 9.12,
    7: 8.23, 8: 8.83, 9: 8.06, 10: 9.83, 11: 9.69, 12: 9.04
}

# 観測所データのファイルパス
station_data_file = "/media/junsaito/E4308E2C308E05B0/Antarctica/Showa/AT/Showa_AT_hourly_1980_2024.csv"
station_data2_file = "/media/junsaito/E4308E2C308E05B0/Antarctica/Showa/DT/Showa_DT_hourly_1980_2024.csv"
station_elev = 29.0


# 氷流速度（ベクトル）データのファイルパス
vx_tif_path = '/media/junsaito/E4308E2C308E05B0/Antarctica/ice_flow/measure_velocity_450m_vx_shirase_lang_new.tif'
vy_tif_path = '/media/junsaito/E4308E2C308E05B0/Antarctica/ice_flow/measure_velocity_450m_vy_shirase_lang_new.tif'
vx_error_tif_path = '/media/junsaito/E4308E2C308E05B0/Antarctica/ice_flow/measure_velocity_450m_errvx_shirase_lang_new.tif'
vy_error_tif_path = '/media/junsaito/E4308E2C308E05B0/Antarctica/ice_flow/measure_velocity_450m_errvy_shirase_lang_new.tif'
shapefile_path = '/media/junsaito/E4308E2C308E05B0/Antarctica/iceshelved_wgs84.shp'
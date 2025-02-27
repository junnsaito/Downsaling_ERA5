#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:42:59 2025

@author: junsaito
"""
# main.py
import os
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import rowcol
from tqdm import tqdm
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import pvlib
from pyproj import Transformer

from config import (target_coords_csv, rema_dem_file, era5_base_dir, modis_wgs84_file,
                    output_dir, time_interval_s, z0_station, kappa_value,
                    station_data_file, station_data2_file, station_elev,
                    vx_tif_path, vy_tif_path, vx_error_tif_path, vy_error_tif_path)
from dem import load_dem, compute_slope_and_aspect, compute_curvature
from era5 import reproject_to_rema, adjust_temperature_with_station, adjust_dewpoint_with_station
from lapse_rate import calc_lapse_rate_parallel
from fluxes import (downscale_pressure, compute_specific_humidity, downscale_longwave,
                    calc_snow_ratio, calc_esat, compute_rel_humidity)
from velocity import load_velocity_components, load_uncertainty_maps, apply_variable_gaussian_filter, compute_strain_rate, compute_effective_horizontal_strain_rate
from parallel_processing import process_time_steps_chunk

def main():
    # 出力フォルダーが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")
    
    # --- 1) DEM, MODIS, 観測所データの読み込み ---
    rema_dem_data, rema_transform, rema_crs = load_dem(rema_dem_file)
    H_dem, W_dem = rema_dem_data.shape

    # MODISは静的データ：指定座標 (39.820, -69.204) からアルベド値を抽出
    with rasterio.open(modis_wgs84_file) as src:
        modis_albedo_data = src.read(1)
        modis_transform = src.transform
        i_mod, j_mod = rowcol(modis_transform, 39.820, -69.204, op=round)
        if 0 <= i_mod < src.height and 0 <= j_mod < src.width:
            modis_albedo_value = modis_albedo_data[i_mod, j_mod]
        else:
            modis_albedo_value = 0.8
    print(f"MODIS Albedo at (39.820, -69.204): {modis_albedo_value}")

    # 観測所データ
    station_data = pd.read_csv(station_data_file, parse_dates=["Date"])
    station_data2 = pd.read_csv(station_data2_file, parse_dates=["Date"])

    # --- 2) 速度データ（氷流）の読み込み（450m, CRS=EPSG:3031） ---
    Vx, Vy, dx, dy, vel_transform, vx_crs = load_velocity_components(vx_tif_path, vy_tif_path)
    uncertainty_map = load_uncertainty_maps(vx_error_tif_path, vy_error_tif_path)
    # 速度データは静的：指定座標 (39.820, -69.204) を用いて抽出
    transformer_vel = Transformer.from_crs("EPSG:4326", vx_crs, always_xy=True)
    target_x_vel, target_y_vel = transformer_vel.transform(39.820, -69.204)
    i_vel, j_vel = rowcol(vel_transform, target_x_vel, target_y_vel, op=round)
    Vx_smooth, Vy_smooth = apply_variable_gaussian_filter(Vx, Vy, uncertainty_map)
    eps_xx, eps_yy, eps_xy, divergence, eps_zz = compute_strain_rate(Vx_smooth, Vy_smooth, dx, dy)
    eps_h = compute_effective_horizontal_strain_rate(eps_xx, eps_yy, eps_xy)
    target_Vx = Vx_smooth[i_vel, j_vel]
    target_Vy = Vy_smooth[i_vel, j_vel]
    flow_speed = np.sqrt(target_Vx**2 + target_Vy**2)
    flow_dir = np.degrees(np.arctan2(target_Vy, target_Vx))
    print(f"氷流速度: {flow_speed:.2f} m/yr, 方向: {flow_dir:.1f}°")

    # --- 3) 複数座標の読み込み（CSV）
    coords_df = pd.read_csv(target_coords_csv)  # CSVには "lon", "lat" 列がある
    all_results = []
    transformer_dem = Transformer.from_crs("EPSG:4326", rema_crs, always_xy=True)

    for idx, row in coords_df.iterrows():
        lon_target = row["lon"]
        lat_target = row["lat"]
        target_x_dem, target_y_dem = transformer_dem.transform(lon_target, lat_target)
        i0_target, j0_target = rowcol(rema_transform, target_x_dem, target_y_dem, op=round)
        i0_target = min(max(i0_target, 0), H_dem - 1)
        j0_target = min(max(j0_target, 0), W_dem - 1)
        print(f"座標 ({lon_target}, {lat_target}) -> DEM index ({i0_target}, {j0_target})")

        # --- 4) ERA5 各年の処理（並列処理）
        all_year_forcing = []
        all_year_rho = []
        all_year_time = []
        years = list(range(2021, 2023))  # 例：2021～2022年
        for year in years:
            print(f"Processing {year} for ({lon_target}, {lat_target}) ...")
            accum_file = os.path.join(era5_base_dir, str(year), "data_stream-oper_stepType-accum.nc")
            instant_file = os.path.join(era5_base_dir, str(year), "data_stream-oper_stepType-instant.nc")
            ds_accum = xr.open_dataset(accum_file)
            ds_instant = xr.open_dataset(instant_file)
            ds_merged = xr.merge([ds_accum, ds_instant], compat="override")
            if "time" in ds_merged.coords:
                ds_merged = ds_merged.rename({"time": "valid_time"})
            ds_year = ds_merged.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31")).load()
            nT_year = ds_year.valid_time.size
            print(f"Year {year}: {nT_year} time steps.")

            src_transform_era5 = rasterio.transform.from_bounds(
                37.9, -70.5, 40.5, -68.5,
                ds_year.longitude.size, ds_year.latitude.size
            )
            # ERA5 平均値の再投影
            t2m_mean_year = ds_year["t2m"].mean(dim="valid_time")
            t2m_first = reproject_to_rema(t2m_mean_year, "EPSG:4326", "EPSG:4326",
                                          src_transform_era5, rema_transform, H_dem, W_dem)
            if t2m_first.shape != rema_dem_data.shape:
                t2m_first = t2m_first.T
            d2m_mean_year = ds_year["d2m"].mean(dim="valid_time")
            d2m_first = reproject_to_rema(d2m_mean_year, "EPSG:4326", "EPSG:4326",
                                          src_transform_era5, rema_transform, H_dem, W_dem)
            if d2m_first.shape != rema_dem_data.shape:
                d2m_first = d2m_first.T      
            skt_mean_year = ds_year["skt"].mean(dim="valid_time")
            skt_first = reproject_to_rema(skt_mean_year, "EPSG:4326", "EPSG:4326",
                                          src_transform_era5, rema_transform, H_dem, W_dem)
            if skt_first.shape != rema_dem_data.shape:
                skt_first = skt_first.T

            lapse_rate_full = calc_lapse_rate_parallel(t2m_first, rema_dem_data)
            lapse_rate_full2 = calc_lapse_rate_parallel(d2m_first, rema_dem_data)
            lapse_rate_full3 = calc_lapse_rate_parallel(skt_first, rema_dem_data)
            lapse_value = lapse_rate_full[i0_target, j0_target]
            lapse_value2 = lapse_rate_full2[i0_target, j0_target]
            lapse_value3 = lapse_rate_full3[i0_target, j0_target]
            delta_z_point = rema_dem_data[i0_target, j0_target]
            curvature2d = compute_curvature(rema_dem_data, dx=500)
            slope2d, aspect2d = compute_slope_and_aspect(rema_dem_data, dx=500)

            time_val_year0 = pd.to_datetime(ds_year.valid_time.values[0])
            lat_mean_year = float(ds_year.latitude.mean().values)
            lon_mean_year = float(ds_year.longitude.mean().values)
            solpos = pvlib.solarposition.get_solarposition(time_val_year0, lat_mean_year, lon_mean_year)

            # 共有メモリの作成（DEMデータの共有）
            shm = shared_memory.SharedMemory(create=True, size=rema_dem_data.nbytes)
            shared_array = np.ndarray(rema_dem_data.shape, dtype=rema_dem_data.dtype, buffer=shm.buf)
            np.copyto(shared_array, rema_dem_data)

            chunk_size = 50
            time_indices = list(range(nT_year))
            year_results = []
            with ProcessPoolExecutor(max_workers=14) as executor:
                futures = [executor.submit(
                    process_time_steps_chunk, time_indices[i:i+chunk_size], ds_year, src_transform_era5, rema_transform,
                    H_dem, W_dem, i0_target, j0_target, lapse_value, lapse_value2, lapse_value3, delta_z_point,
                    slope2d, aspect2d, curvature2d, solpos, modis_albedo_value, rema_dem_data,
                    time_interval_s, station_data, station_data2, station_elev
                ) for i in range(0, nT_year, chunk_size)]
                for fut in tqdm(futures, total=len(futures), desc=f"Processing {year} chunks"):
                    year_results.extend(fut.result())

            shm.close()
            shm.unlink()

            year_forcing = [res[:-2] for res in year_results]
            year_rho = [res[10] + res[11] for res in year_results]
            year_time = [res[8] for res in year_results]

            all_year_forcing.extend(year_forcing)
            all_year_rho.extend(year_rho)
            all_year_time.extend(year_time)
            print(f"Year {year} processed for ({lon_target}, {lat_target}).")

        # 全期間の結果を時系列でソート
        combined = list(zip(all_year_time, all_year_forcing, all_year_rho))
        combined.sort(key=lambda x: x[0])
        all_year_time_sorted, all_year_forcing_sorted, all_year_rho_sorted = zip(*combined)

        lon_str = str(lon_target).replace(".", "_")
        lat_str = str(lat_target).replace(".", "_")
        forcing_csv = os.path.join(output_dir, f"cfm_forcing_{lon_str}_{lat_str}.csv")
        df_forcing = pd.DataFrame(list(all_year_forcing_sorted),
                                  columns=["SW_d", "LW_d", "T2m", "TSKIN", "QH", "QL", "RAIN", "BDOT", "time", "ALBEDO"])
        df_forcing["time"] = pd.to_datetime(list(all_year_time_sorted))
        df_forcing.set_index("time", inplace=True)
        df_forcing.to_csv(forcing_csv, float_format="%.3f")
        print("Forcing CSV:", forcing_csv)

        # MODISと速度は静的データなので、各座標ごとに抽出した値を全時刻に流用
        strain_csv = os.path.join(output_dir, f"cfm_strain_inputs_{lon_str}_{lat_str}.csv")
        time_strain = pd.date_range("2021-01-01", "2022-12-31 23:00:00", freq="H")
        N_strain = len(time_strain)
        eps_xx_arr = np.full(N_strain, flow_speed)
        eps_yy_arr = np.full(N_strain, flow_dir)
        eps_xy_arr = np.full(N_strain, 0.0)
        def to_year_fraction_strain(dt):
            year = dt.year
            start = pd.Timestamp(year, 1, 1)
            end = pd.Timestamp(year+1, 1, 1)
            return year + (dt - start).total_seconds() / ((end - start).total_seconds())
        dec_dates_strain = [to_year_fraction_strain(t) for t in time_strain]
        data_strain = np.column_stack([dec_dates_strain, eps_xx_arr, eps_yy_arr, eps_xy_arr])
        df_strain = pd.DataFrame(data_strain, columns=["DecimalDate", "eps_xx", "eps_yy", "eps_xy"])
        df_strain_T = df_strain.T
        df_strain_T.to_csv(strain_csv, header=False, index=False, float_format="%.8f")
        print("Strain CSV:", strain_csv)

        rho_csv = os.path.join(output_dir, f"cfm_rho_inputs_{lon_str}_{lat_str}.csv")
        def to_year_fraction_rho(dt):
            year = dt.year
            start = pd.Timestamp(year, 1, 1)
            end = pd.Timestamp(year+1, 1, 1)
            return year + (dt - start).total_seconds() / ((end - start).total_seconds())
        dec_dates_rho = [to_year_fraction_rho(t) for t in all_year_time_sorted]
        df_rho = pd.DataFrame([dec_dates_rho, all_year_rho_sorted]).T
        df_rho.to_csv(rho_csv, header=False, index=False, float_format="%.8f")
        print("Density CSV:", rho_csv)

        all_results.append({
            "lon": lon_target,
            "lat": lat_target,
            "forcing_csv": forcing_csv,
            "strain_csv": strain_csv,
            "rho_csv": rho_csv,
            "flow_speed": flow_speed,
            "flow_dir": flow_dir
        })

    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(output_dir, "summary_target_outputs.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("Summary CSV:", summary_csv)

if __name__ == "__main__":
    main()

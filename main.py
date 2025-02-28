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

###########################
# ここから各種関数をインポート
###########################
from config import (
    target_coords_csv,            # 複数座標のCSV, "lon", "lat"
    rema_dem_file, era5_base_dir, modis_wgs84_file,
    output_dir, time_interval_s,
    z0_station, kappa_value,
    station_data_file, station_data2_file, station_elev,
    vx_tif_path, vy_tif_path, vx_error_tif_path, vy_error_tif_path
)
from dem import load_dem
from era5 import (
    reproject_to_rema, adjust_temperature_with_station,
    adjust_dewpoint_with_station
)
from lapse_rate import calc_lapse_rate_parallel
from fluxes import (
    downscale_pressure, compute_specific_humidity, downscale_longwave,
    calc_snow_ratio, calc_esat, compute_rel_humidity
)
from velocity import (
    load_velocity_components, load_uncertainty_maps, apply_variable_gaussian_filter,
    compute_strain_rate, compute_effective_horizontal_strain_rate
)
from parallel_processing import process_time_steps_chunk   # → 下記で定義例
###########################################################


def main():
    # ---------------------------
    # 出力フォルダー作成
    # ---------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output folder: {output_dir}")

    # ---------------------------
    # DEM 読み込み
    # ---------------------------
    rema_dem_data, rema_transform, rema_crs = load_dem(rema_dem_file)
    H_dem, W_dem = rema_dem_data.shape
    print("[INFO] DEM loaded:", rema_dem_file, (H_dem, W_dem))

    # ---------------------------
    # MODIS (静的) 読み込み
    # ---------------------------
    with rasterio.open(modis_wgs84_file) as modis_src:
        modis_albedo_data = modis_src.read(1)
        modis_transform = modis_src.transform
        print("[INFO] MODIS loaded:", modis_wgs84_file, modis_albedo_data.shape)

    # ---------------------------
    # 観測所データ（2種類）
    # ---------------------------
    station_data = pd.read_csv(station_data_file, parse_dates=["Date"])
    station_data2 = pd.read_csv(station_data2_file, parse_dates=["Date"])

    # ---------------------------
    # Velocity (450m, EPSG:3031) + 不確実性マップ
    # ---------------------------
    Vx, Vy, dx, dy, vel_transform, vx_crs = load_velocity_components(vx_tif_path, vy_tif_path)
    uncertainty_map = load_uncertainty_maps(vx_error_tif_path, vy_error_tif_path)
    # ガウシアンでスムージング
    Vx_smooth, Vy_smooth = apply_variable_gaussian_filter(Vx, Vy, uncertainty_map)

    # ---------------------------
    # 複数座標を CSV から読み込み
    #   (例: "lon", "lat" の列)
    # ---------------------------
    coords_df = pd.read_csv(target_coords_csv)
    print("[INFO] target coords loaded:", len(coords_df))

    all_results = []

    # EPSG:4326 → REMA(=EPSG:4326 500m) じゃなく...?
    #   DEMは WGS84(=EPSG:4326) そのままなので、変換不要かもしれない
    #   ただし rowcol() でインデックス取るだけならOK
    #   Velocity だけEPSG:3031に変換
    transformer_vel = Transformer.from_crs("EPSG:4326", vx_crs, always_xy=True)

    # --------------------------------------------
    # 複数座標をループし、それぞれに対してERA5計算
    # --------------------------------------------
    for idx, row in coords_df.iterrows():
        lon_target = row["lon"]
        lat_target = row["lat"]
        print("\n==============================")
        print(f"[INFO] Now processing lon={lon_target}, lat={lat_target}")

        # 1) DEM上でのインデックス
        i0_target, j0_target = rowcol(rema_transform, lon_target, lat_target, op=round)
        i0_target = np.clip(i0_target, 0, H_dem - 1)
        j0_target = np.clip(j0_target, 0, W_dem - 1)
        print(f"  DEM index=({i0_target}, {j0_target})")

        # 2) MODISアルベド抽出
        i_mod, j_mod = rowcol(modis_transform, lon_target, lat_target, op=round)
        if 0 <= i_mod < modis_albedo_data.shape[0] and 0 <= j_mod < modis_albedo_data.shape[1]:
            modis_albedo_value = float(modis_albedo_data[i_mod, j_mod])
        else:
            modis_albedo_value = 0.8
        print(f"  MODIS Albedo= {modis_albedo_value:.3f}")

        # 3) Velocity抽出(氷流速)
        tx_vel, ty_vel = transformer_vel.transform(lon_target, lat_target)
        i_vel, j_vel = rowcol(vel_transform, tx_vel, ty_vel, op=round)
        # インデックスが範囲外の場合のクリップ
        i_vel = np.clip(i_vel, 0, Vx_smooth.shape[0] - 1)
        j_vel = np.clip(j_vel, 0, Vx_smooth.shape[1] - 1)

        vx_val = Vx_smooth[i_vel, j_vel]
        vy_val = Vy_smooth[i_vel, j_vel]
        flow_speed = float(np.sqrt(vx_val**2 + vy_val**2))
        flow_dir = float(np.degrees(np.arctan2(vy_val, vx_val)))
        print(f"  Ice Flow speed= {flow_speed:.2f} m/yr, dir= {flow_dir:.1f} deg")

        # ---------------------------
        # ERA5を年ごとに読み込んで並列実行
        # ---------------------------
        all_year_forcing = []
        all_year_rho = []
        all_year_time = []

        # ここでは 2021-2022 例として
        years = [2021, 2022]
        for year in years:
            print(f"  >> Year={year}")
            accum_file = os.path.join(era5_base_dir, str(year), "data_stream-oper_stepType-accum.nc")
            instant_file = os.path.join(era5_base_dir, str(year), "data_stream-oper_stepType-instant.nc")

            ds_accum = xr.open_dataset(accum_file)
            ds_instant = xr.open_dataset(instant_file)
            ds_merged = xr.merge([ds_accum, ds_instant], compat="override")
            if "time" in ds_merged.coords:
                ds_merged = ds_merged.rename({"time": "valid_time"})
            ds_year = ds_merged.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31")).load()
            nT_year = ds_year.valid_time.size
            print(f"     nT_year= {nT_year}")

            # ERA5 -> DEM再投影
            #   まず年平均でラプスレートを計算
            src_transform_era5 = rasterio.transform.from_bounds(
                37.9, -70.5, 40.5, -68.5,
                ds_year.longitude.size, ds_year.latitude.size
            )
            t2m_aver = ds_year["t2m"].mean(dim="valid_time")
            t2m_first = reproject_to_rema(
                t2m_aver, "EPSG:4326", "EPSG:4326",
                src_transform_era5, rema_transform,
                W_dem, H_dem
            )
            if t2m_first.shape != rema_dem_data.shape:
                # 必要なら転置など
                t2m_first = t2m_first.T

            d2m_aver = ds_year["d2m"].mean(dim="valid_time")
            d2m_first = reproject_to_rema(
                d2m_aver, "EPSG:4326", "EPSG:4326",
                src_transform_era5, rema_transform,
                W_dem, H_dem
            )
            if d2m_first.shape != rema_dem_data.shape:
                d2m_first = d2m_first.T

            skt_aver = ds_year["skt"].mean(dim="valid_time")
            skt_first = reproject_to_rema(
                skt_aver, "EPSG:4326", "EPSG:4326",
                src_transform_era5, rema_transform,
                W_dem, H_dem
            )
            if skt_first.shape != rema_dem_data.shape:
                skt_first = skt_first.T

            # ラプスレート計算
            lapse_1 = calc_lapse_rate_parallel(t2m_first, rema_dem_data)
            lapse_2 = calc_lapse_rate_parallel(d2m_first, rema_dem_data)
            lapse_3 = calc_lapse_rate_parallel(skt_first, rema_dem_data)

            # この座標におけるラプス値
            lapse_value  = lapse_1[i0_target, j0_target]
            lapse_value2 = lapse_2[i0_target, j0_target]
            lapse_value3 = lapse_3[i0_target, j0_target]
            delta_z_point = rema_dem_data[i0_target, j0_target]

            # 地形量
            from scipy.ndimage import laplace
            curvature2d = laplace(rema_dem_data) / (500**2)
            from dem import compute_slope_and_aspect
            slope2d, aspect2d = compute_slope_and_aspect(rema_dem_data, dx=500)

            # # 時刻0 の値で太陽天頂角など
            # time_val_0 = pd.to_datetime(ds_year.valid_time.values[0])
            # lat_mean_year = float(ds_year.latitude.mean().values)
            # lon_mean_year = float(ds_year.longitude.mean().values)
            # solpos = pvlib.solarposition.get_solarposition(time_val_0, lat_mean_year, lon_mean_year)
            
            
        # **各時間ステップごとに太陽位置を計算**
            print("  Calculating solar position for each timestamp...")
            solpos_list = []
            for t in tqdm(ds_year.valid_time.values, desc="  Solar position calc"):
                time_val = pd.to_datetime(t)  # **ERA5 の valid_time を使う**
                solpos = pvlib.solarposition.get_solarposition(time_val, lon_target, lat_target)
                solpos_list.append(solpos)
        
            # `solpos_list` を numpy 配列に変換（高速化）
            solpos_array = np.array([[sp.zenith.values[0], sp.azimuth.values[0]] for sp in solpos_list])
        
            # `zenith` と `azimuth` を `process_time_steps_chunk` に渡せる形にする
            zenith_array = solpos_array[:, 0]  # 太陽天頂角（zenith）
            azimuth_array = solpos_array[:, 1]  # 太陽方位角（azimuth）            
            
            
            
            

            # 共有メモリ
            shm = shared_memory.SharedMemory(create=True, size=rema_dem_data.nbytes)
            shared_array = np.ndarray(rema_dem_data.shape, dtype=rema_dem_data.dtype, buffer=shm.buf)
            np.copyto(shared_array, rema_dem_data)

            # タスク実行
            chunk_size = 50
            time_indices = list(range(nT_year))
            year_results = []
            with ProcessPoolExecutor(max_workers=14) as exe:
                futures = []
                for start_idx in range(0, nT_year, chunk_size):
                    chunk = time_indices[start_idx : start_idx+chunk_size]
                    fut = exe.submit(
                        process_time_steps_chunk,
                        chunk,
                        ds_year,                     # ds_year
                        src_transform_era5,          # src_transform_era5
                        rema_transform,              # rema_transform
                        W_dem, H_dem,                # W_dem, H_dem
                        i0_target, j0_target,        # i0, j0
                        lapse_value, lapse_value2,   # lapse
                        lapse_value3, delta_z_point,
                        slope2d, aspect2d, curvature2d,
                        zenith_array, azimuth_array, modis_albedo_value,
                        rema_dem_data,
                        time_interval_s,
                        station_data,
                        station_data2,
                        station_elev
                    )
                    futures.append(fut)
                for ft in tqdm(futures, desc=f"Processing {year} chunks"):
                    year_results.extend(ft.result())

            shm.close()
            shm.unlink()

            # year結果をまとめ
            #  => res=(sw_corr, lw_corr, temp_corr, tskin_corr, QH_val, QE_val, RAIN_val, BDOT_val, time_val, ALBEDO_val, rho_new_val, rho_redeposit_val)
            year_forcing = [r[:-2] for r in year_results]    # 最後2つ(rho_new_val, rho_redeposit_val)を除く
            year_rho     = [r[10] + r[11] for r in year_results]  # r[10] + r[11] = rho_new_val + rho_redeposit_val
            year_time    = [r[8] for r in year_results]      # r[8] = time_val

            # accumulate
            all_year_forcing.extend(year_forcing)
            all_year_rho.extend(year_rho)
            all_year_time.extend(year_time)

        # ひとつの座標に対する全timeを時系列ソート
        combined = list(zip(all_year_time, all_year_forcing, all_year_rho))
        combined.sort(key=lambda x: x[0])
        time_sorted, forcing_sorted, rho_sorted = zip(*combined)

        # CSV出力
        # ========== Forcing ==========
        forcing_csv = os.path.join(
            output_dir,
            f"cfm_forcing_{lon_target:.3f}_{lat_target:.3f}.csv"
        )
        df_forcing = pd.DataFrame(list(forcing_sorted),
            columns=["SW_d", "LW_d", "T2m", "TSKIN", "QH", "QL", "RAIN", "BDOT", "time", "ALBEDO"]
        )
        df_forcing["time"] = pd.to_datetime(time_sorted)
        df_forcing.set_index("time", inplace=True)
        df_forcing.to_csv(forcing_csv, float_format="%.3f")
        print("[DONE] Forcing CSV:", forcing_csv)

        # ========== Strain (Velocity) ==========
        # Velocityは静的: flow_speed, flow_dir
        #  → 全時刻同じでもいいなら以下でOK
        strain_csv = os.path.join(
            output_dir,
            f"cfm_strain_inputs_{lon_target:.3f}_{lat_target:.3f}.csv"
        )
        # 例えば 2021-2022 全期間(2年x8760=17520時間)
        # ここをどうするかはお好み
        t_start = pd.Timestamp("2021-01-01")
        t_end   = pd.Timestamp("2022-12-31 23:00:00")
        time_strain = pd.date_range(t_start, t_end, freq="H")
        N_strain = len(time_strain)
        eps_xx, eps_yy, eps_xy, divergence, eps_zz = compute_strain_rate(Vx_smooth, Vy_smooth, dx, dy)
        target_eps_xx = eps_xx[i_vel, j_vel]
        target_eps_yy = eps_yy[i_vel, j_vel]
        target_eps_xy = eps_xy[i_vel, j_vel]
        
        eps_xx_arr = np.full(N_strain, target_eps_xx)  
        eps_yy_arr = np.full(N_strain, target_eps_yy)
        eps_xy_arr = np.full(N_strain, target_eps_xy)
        # arr_xx = np.full(N_strain, flow_speed)
        # arr_yy = np.full(N_strain, flow_dir)
        # arr_xy = np.zeros(N_strain)
        # decimal date
        def to_year_frac(dt):
            y = dt.year
            start = pd.Timestamp(y,1,1)
            end   = pd.Timestamp(y+1,1,1)
            return y + (dt - start).total_seconds()/(end-start).total_seconds()
        dec_dates = [to_year_frac(t) for t in time_strain]
        data_strain = np.column_stack([dec_dates,eps_xx_arr, eps_yy_arr,eps_xy_arr])
        df_strain = pd.DataFrame(data_strain, columns=["DecimalDate","eps_xx","eps_yy","eps_xy"])
        df_strain_T = df_strain.T
        df_strain_T.to_csv(strain_csv, header=False, index=False, float_format="%.8f")
        print("[DONE] Strain CSV:", strain_csv)

        # ========== Density (rho) ==========
        # ここでは "cfm_rho_inputs_*.csv" として year_rho から書き出し
        rho_csv = os.path.join(
            output_dir,
            f"cfm_rho_inputs_{lon_target:.3f}_{lat_target:.3f}.csv"
        )
        # decimaldate
        dec_dates_rho = []
        def to_year_frac_rho(dt):
            y = dt.year
            st = pd.Timestamp(y,1,1)
            ed = pd.Timestamp(y+1,1,1)
            return y + (dt - st).total_seconds()/(ed-st).total_seconds()

        for t in time_sorted:
            dec_dates_rho.append(to_year_frac_rho(t))

        df_rho = pd.DataFrame(list(zip(dec_dates_rho, rho_sorted)))
        df_rho_T = df_rho.T
        # rho_csv = os.path.join(output_dir, f"cfm_rho_inputs_{lon_str}_{lat_str}.csv")
        df_rho_T.to_csv(rho_csv, header=False, index=False, float_format="%.8f")
        print("[DONE] Rho CSV:", rho_csv)

        # summarize
        all_results.append({
            "lon": lon_target,
            "lat": lat_target,
            "forcing_csv": forcing_csv,
            "strain_csv": strain_csv,
            "rho_csv": rho_csv,
            "flow_speed": flow_speed,
            "flow_dir": flow_dir
        })

    # 全座標まとめ
    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(output_dir, "summary_target_outputs.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[DONE] Summary CSV:", summary_csv)


if __name__ == "__main__":
    main()

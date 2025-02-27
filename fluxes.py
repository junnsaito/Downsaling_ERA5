#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:42:20 2025

@author: junsaito
"""
# fluxes.py
import numpy as np
import pandas as pd
from scipy.ndimage import sobel, laplace, gaussian_filter

def downscale_pressure(press, t_orig, t_corr, dz):
    R = 287.0
    g = 9.81
    T_mean = (t_orig + t_corr) / 2.0
    T_mean = max(T_mean, 200)
    return press * np.exp(-g * dz / (R * T_mean))

def compute_specific_humidity(press, dew):
    C1, C2, C3 = 611.21, 17.67, 243.5
    tmp_diff = max(dew - C3, 1e-6)
    e_s = C1 * np.exp((C2 * (dew - 273.15)) / tmp_diff)
    e_s = max(e_s, 1e-9)
    return 0.622 * e_s / max(press - 0.378 * e_s, 1e-9)

def downscale_longwave(lw, t_orig, t_corr):
    t_o = max(t_orig, 200)
    t_c = max(t_corr, 200)
    emi_orig = 0.85 + 0.005 * (t_o - 273.15)
    emi_corr = 0.85 + 0.005 * (t_c - 273.15)
    return lw * (emi_corr / emi_orig) * ((t_corr / t_o) ** 4)

def calc_snow_ratio(pressure2d, qv2d, temp_corr2d):
    pressure2d_hpa = pressure2d / 100.0
    vap = (pressure2d_hpa * qv2d) / (0.622 + 0.378 * qv2d)
    Tw = 0.584 * (temp_corr2d - 273.15) + 0.875 * vap - 5.32
    Tw_low = 1 - 0.5 * np.exp(-2.2 * np.abs(1.1 - Tw)**1.3)
    Tw_high = 0.5 * np.exp(-2.2 * np.abs(Tw - 1.1)**1.3)
    Tw_low = np.where(Tw >= 1.1, 0, Tw_low)
    Tw_high = np.where(Tw < 1.1, 0, Tw_high)
    s_rate = Tw_low + Tw_high
    return np.clip(s_rate, 0, 1)

def calc_esat(T):
    return 611.2 * np.exp(17.62*(T - 273.15)/(T - 30.03))

def compute_solar_incidence_angle(zen_deg, azi_deg, slope_deg, aspect_deg):
    z_rad = np.radians(zen_deg)
    a_rad = np.radians(azi_deg)
    sp_rad = np.radians(slope_deg)
    as_rad = np.radians(aspect_deg)
    return np.cos(z_rad)*np.cos(sp_rad) + np.sin(z_rad)*np.sin(sp_rad)*np.cos(a_rad - as_rad)

def apply_topo_correction(sw_wm2, zen_deg, azi_deg, slope_deg, aspect_deg, dem2d, dx, i0, j0):
    cos_incidence = compute_solar_incidence_angle(zen_deg, azi_deg, slope_deg, aspect_deg)
    if zen_deg > 90:
        return 0.0
    sun_rad = np.radians(azi_deg)
    H, W = dem2d.shape
    shadow_mask = np.zeros_like(dem2d, dtype=bool)
    for k in range(1, 50):
        shift_x = int(np.round(np.cos(sun_rad) * k))
        shift_y = int(np.round(np.sin(sun_rad) * k))
        shift_x = np.clip(shift_x, -W+1, W-1)
        shift_y = np.clip(shift_y, -H+1, H-1)
        dem_shifted = np.roll(dem2d, shift_x, axis=1)
        dem_shifted = np.roll(dem_shifted, shift_y, axis=0)
        shadow_angle = np.degrees(np.arctan((dem_shifted - dem2d) / (dx * k + 1e-6)))
        shadow_mask |= (shadow_angle > (90 - zen_deg))
    if shadow_mask[i0, j0]:
        return 0.0
    cos_grid = max(np.cos(np.radians(zen_deg)), 0.1)
    ratio = np.clip(cos_incidence / cos_grid, 0, 3.0)
    return sw_wm2 * ratio

def compute_rel_humidity(t2d, dew2d):
    C1, C2, C3 = 611.21, 17.67, 243.5
    e_s = C1 * np.exp((C2 * (t2d - 273.15)) / (t2d - 273.15 + C3))
    e = C1 * np.exp((C2 * (dew2d - 273.15)) / (dew2d - 273.15 + C3))
    return np.clip(e / e_s, 0, 1)

def downscale_dew_point(dew2d, lapse2d, dz2d):
    return dew2d + (lapse2d * dz2d)

def to_year_fraction(dt):
    dt = pd.to_datetime(dt)
    year = dt.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year+1, month=1, day=1)
    return year + (dt - start).total_seconds() / ((end - start).total_seconds())

def calc_lapse_rate_block(block_indices, temp2d, elev2d):
    results = []
    for (i, j) in block_indices:
        if i < 2 or j < 2 or i >= temp2d.shape[0]-2 or j >= temp2d.shape[1]-2:
            results.append((i, j, -6.5e-3))
            continue
        t_win = temp2d[i-2:i+3, j-2:j+3].flatten()
        e_win = elev2d[i-2:i+3, j-2:j+3].flatten()
        if np.std(e_win) == 0:
            slope = -6.5e-3
        else:
            slope_, _, _, _, _ = linregress(e_win, t_win)
            slope = np.clip(slope_, -9e-3, -4e-3)
        results.append((i, j, slope))
    return results

def calc_lapse_rate_parallel(temp2d, elev2d):
    H, W = temp2d.shape
    indices = [(i, j) for i in range(H) for j in range(W)]
    block_size = 1000
    lapse = np.full((H, W), -6.5e-3, dtype=np.float32)
    blocks = [indices[k:k+block_size] for k in range(0, len(indices), block_size)]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(calc_lapse_rate_block, blk, temp2d, elev2d) for blk in blocks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Calculating Lapse Rate"):
            for i, j, slope in fut.result():
                lapse[i, j] = slope
    return lapse

def psi_m_stable(xi):
    a, b, c, d = 6.1, 2.5, 5.3, 1.1
    return a*xi + b*(xi - c/d)*np.exp(-d*xi) + (b*c/d)

def psi_h_stable(xi):
    a, b, c, d = 6.1, 2.5, 5.3, 1.1
    return a*xi + b*(xi - c/d)*np.exp(-d*xi) + (b*c/d)

def psi_m_unstable(xi):
    return 2.0 * np.log((1.0 + (1.0 - 16.0*xi)**0.25)/2.0) + np.log((1.0 + (1.0 - 16.0*xi)**0.5)/2.0) - 2.0*np.arctan((1.0 - 16.0*xi)**0.25) + np.pi/2.0

def psi_h_unstable(xi):
    return 2.0 * np.log((1.0 + (1.0 - 16.0*xi)**0.25)/2.0)

def calc_flux_bulk_iter(Tz, ez, Uz, T0, e0, pz=101325.0, z=2.0,
                        z0w=1e-3, z0T=1e-4, z0E=1e-4,
                        rho=1.3, cp=1005.0, Lv=2.5e6,
                        g=9.81, k=0.4, max_iter=20):
    if Uz < 0.01:
        return 0.0, 0.0
    L_current = 1e6
    T_mean = (Tz + T0) / 2.0
    R_d = 287.05
    rho_air = pz / (R_d * T_mean)
    for _ in range(max_iter):
        xi = z / L_current
        if L_current > 0:
            psi_m_val = psi_m_stable(xi)
            psi_h_val = psi_h_stable(xi)
        else:
            psi_m_val = psi_m_unstable(xi)
            psi_h_val = psi_h_unstable(xi)
        denom_m = np.log(z / z0w) - psi_m_val
        denom_h = np.log(z / z0T) - psi_h_val
        denom_m = max(denom_m, 1e-6)
        denom_h = max(denom_h, 1e-6)
        u_star = k * Uz / denom_m
        CH = (k**2) / (denom_m * denom_h)
        QH = rho_air * cp * CH * Uz * (Tz - T0)
        QE = rho_air * Lv * CH * Uz * ((ez - e0) / pz)
        if abs(QH) < 1e-9:
            L_new = 1e6
        else:
            L_new = -(rho_air * cp * (u_star**3) * Tz) / (k * g * QH)
        if np.isnan(L_new) or np.isinf(L_new):
            L_new = 1e6
        if abs(L_new - L_current) < 1.0:
            L_current = L_new
            break
        L_current = L_new
    return QH, QE

def adjust_wind_speed(W_coarse, u10, v10, slope, curvature, delta_z, roughness, d=0.1, measurement_height=10.0):
    kappa = 0.41
    z0 = np.maximum(roughness, 1e-3)
    u_star = kappa * W_coarse / np.log((measurement_height - d) / z0)
    U_log = u_star / kappa * np.log((measurement_height - d) / z0)
    wind_dir_rad = np.arctan2(v10, u10)
    slope_rad = np.radians(slope)
    Vs = slope_rad * np.cos(wind_dir_rad)
    terrain_factor = 1 + 0.5 * Vs + 0.5 * curvature
    altitude_factor = np.exp(0.0015 * np.clip(delta_z, -500, 500))
    U_fine = U_log * terrain_factor * altitude_factor
    adjusted_speed = gaussian_filter(U_fine, sigma=2)
    return np.clip(adjusted_speed, 0.1, 50)

def compute_slope_and_aspect(dem, dx):
    dzdx = sobel(dem, axis=1) / (8.0 * dx)
    dzdy = sobel(dem, axis=0) / (8.0 * dx)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)
    aspect = np.arctan2(-dzdy, dzdx) * (180 / np.pi)
    aspect[aspect < 0] += 360
    return slope, aspect

def adjust_precipitation(P0, z, z0, kappa):
    dz_km = (z - z0) / 1000.0
    correction = 1 + dz_km / (1 + dz_km / kappa)
    return P0 * correction

def calc_new_snow_density(T2m, Ts, RH, U10m):
    T2m_C = T2m - 273.15
    Ts_C = Ts - 273.15
    rho_new = (70 + 6.5 * T2m_C + 7.5 * Ts_C + 0.26 * RH + 13 * U10m
               - 4.5 * T2m_C * Ts_C - 0.65 * T2m_C * U10m - 0.17 * RH * U10m
               + 0.06 * T2m_C * Ts_C * RH)
    return np.clip(rho_new, 30, 150)

def calc_threshold_friction_velocity(SP, rg, rb, N3, rho_air=1.1, rho_ice=917, g=9.8, sigma=300):
    A, B = 0.02, 0.0015
    return np.sqrt((A * rho_ice * rg * (SP + 1) + B * sigma * N3 * (rb**2 / rg**2)) / rho_air)

def calc_redeposit_density(U10m):
    if U10m > 1:
        return 361 * np.log10(U10m) + 33
    else:
        return 33

def compute_solar_incidence_angle(zen_deg, azi_deg, slope_deg, aspect_deg):
    z_rad = np.radians(zen_deg)
    a_rad = np.radians(azi_deg)
    sp_rad = np.radians(slope_deg)
    as_rad = np.radians(aspect_deg)
    return np.cos(z_rad)*np.cos(sp_rad) + np.sin(z_rad)*np.sin(sp_rad)*np.cos(a_rad - as_rad)

def apply_topo_correction(sw_wm2, zen_deg, azi_deg, slope_deg, aspect_deg, dem2d, dx, i0, j0):
    cos_incidence = compute_solar_incidence_angle(zen_deg, azi_deg, slope_deg, aspect_deg)
    if zen_deg > 90:
        return 0.0
    sun_rad = np.radians(azi_deg)
    H, W = dem2d.shape
    shadow_mask = np.zeros_like(dem2d, dtype=bool)
    for k in range(1, 50):
        shift_x = int(np.round(np.cos(sun_rad) * k))
        shift_y = int(np.round(np.sin(sun_rad) * k))
        shift_x = np.clip(shift_x, -W+1, W-1)
        shift_y = np.clip(shift_y, -H+1, H-1)
        dem_shifted = np.roll(dem2d, shift_x, axis=1)
        dem_shifted = np.roll(dem_shifted, shift_y, axis=0)
        shadow_angle = np.degrees(np.arctan((dem_shifted - dem2d) / (dx * k + 1e-6)))
        shadow_mask |= (shadow_angle > (90 - zen_deg))
    if shadow_mask[i0, j0]:
        return 0.0
    cos_grid = max(np.cos(np.radians(zen_deg)), 0.1)
    ratio = np.clip(cos_incidence / cos_grid, 0, 3.0)
    return sw_wm2 * ratio

def compute_rel_humidity(t2d, dew2d):
    C1, C2, C3 = 611.21, 17.67, 243.5
    e_s = C1 * np.exp((C2 * (t2d - 273.15)) / (t2d - 273.15 + C3))
    e = C1 * np.exp((C2 * (dew2d - 273.15)) / (dew2d - 273.15 + C3))
    return np.clip(e / e_s, 0, 1)

def downscale_dew_point(dew2d, lapse2d, dz2d):
    return dew2d + (lapse2d * dz2d)

def to_year_fraction(dt):
    dt = pd.to_datetime(dt)
    year = dt.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year+1, month=1, day=1)
    return year + (dt - start).total_seconds() / ((end - start).total_seconds())

def calc_lapse_rate_block(block_indices, temp2d, elev2d):
    results = []
    for (i, j) in block_indices:
        if i < 2 or j < 2 or i >= temp2d.shape[0]-2 or j >= temp2d.shape[1]-2:
            results.append((i, j, -6.5e-3))
            continue
        t_win = temp2d[i-2:i+3, j-2:j+3].flatten()
        e_win = elev2d[i-2:i+3, j-2:j+3].flatten()
        if np.std(e_win) == 0:
            slope = -6.5e-3
        else:
            slope_, _, _, _, _ = linregress(e_win, t_win)
            slope = np.clip(slope_, -9e-3, -4e-3)
        results.append((i, j, slope))
    return results

def calc_lapse_rate_parallel(temp2d, elev2d):
    H, W = temp2d.shape
    indices = [(i, j) for i in range(H) for j in range(W)]
    block_size = 1000
    lapse = np.full((H, W), -6.5e-3, dtype=np.float32)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    blocks = [indices[k:k+block_size] for k in range(0, len(indices), block_size)]
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(calc_lapse_rate_block, np.array(blk), temp2d, elev2d) for blk in blocks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Calculating Lapse Rate"):
            for i, j, slope in fut.result():
                lapse[i, j] = slope
    return lapse



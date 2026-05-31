import requests
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

def get_automatic_wind_rose(lat, lon, hub_height, turb_ci=4.0, turb_co=25.0, rated_ws=9.8, n_bins=16, alpha=0.143, years=2):
    """
    Fetch historical wind data from Open-Meteo ERA5 API and generate a wind rose.
    
    Parameters
    ----------
    lat : float
        Latitude in WGS84.
    lon : float
        Longitude in WGS84.
    hub_height : float
        Turbine hub height to extrapolate wind speed via Power Law.
    n_bins : int
        Number of directional bins (default 36, meaning 10 degrees each).
    alpha : float
        Hellmann exponent for Power Law (0.143 is standard for offshore/coastal).
    years : int
        Number of recent years to fetch for averaging (default 2 years).
        
    Returns
    -------
    wind_dir : ndarray
        Angles of each bin center in degrees.
    wind_freq : ndarray
        Probability (0 to 1) of wind coming from each bin.
    wind_speed : ndarray
        Average wind speed (m/s) at hub_height for each bin.
    """
    print(f"[Wind API] Fetching {years} years of ERA5 data for Lat: {lat:.4f}, Lon: {lon:.4f}...")
    
    # Calculate dates: from 14 days ago (to ensure data availability) back by 'years' years
    end_date_dt = datetime.now() - timedelta(days=14)
    start_date_dt = end_date_dt - timedelta(days=365 * years)
    
    end_date = end_date_dt.strftime("%Y-%m-%d")
    start_date = start_date_dt.strftime("%Y-%m-%d")
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=windspeed_100m,winddirection_100m&"
        f"wind_speed_unit=ms"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch wind data: HTTP {response.status_code}")
        
    data = response.json()
    if "hourly" not in data:
        raise ValueError("Invalid API response format (missing 'hourly' data).")
        
    ws_100m = np.array(data["hourly"]["windspeed_100m"], dtype=float)
    wd_100m = np.array(data["hourly"]["winddirection_100m"], dtype=float)
    
    # Remove NaN values if any
    valid = ~np.isnan(ws_100m) & ~np.isnan(wd_100m)
    ws_100m = ws_100m[valid]
    wd_100m = wd_100m[valid]
    
    # Apply Power Law for height correction
    # V_hub = V_100 * (Z_hub / 100)^alpha
    ws_hub = ws_100m * ((hub_height / 100.0) ** alpha)
    
    # Setup Bins
    bin_width = 360.0 / n_bins
    wind_dir = np.arange(0, 360, bin_width)
    wind_freq = np.zeros(n_bins)
    wind_speed = np.zeros(n_bins)
    
    # To correctly bin, we shift directions by half a bin so 0 is centered
    wd_shifted = (wd_100m + (bin_width / 2.0)) % 360.0
    bin_indices = (wd_shifted // bin_width).astype(int)
    
    total_valid_hours = len(ws_hub)
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        count = np.sum(mask)
        wind_freq[i] = count / total_valid_hours
        if count > 0:
            wind_speed[i] = np.mean(ws_hub[mask])
            
    print(f"[Wind API] Successfully processed {total_valid_hours} hourly records (from {start_date} to {end_date}).")
    
    # --- Diagnostics ---
    below_cut_in = float(np.sum(ws_hub < turb_ci) / total_valid_hours * 100)
    above_cut_out = float(np.sum(ws_hub > turb_co) / total_valid_hours * 100)
    rated_to_cut_out = float(np.sum((ws_hub >= rated_ws) & (ws_hub <= turb_co)) / total_valid_hours * 100)
    cubic_region = float(np.sum((ws_hub >= turb_ci) & (ws_hub < rated_ws)) / total_valid_hours * 100)
    mean_speed = float(np.mean(ws_hub))
    
    diagnostic = {
        "site_latitude": lat,
        "site_longitude": lon,
        "hub_height_m": hub_height,
        "mean_wind_speed_ms": mean_speed,
        "hours_evaluated": total_valid_hours,
        "time_below_cut_in_percent": below_cut_in,
        "time_in_cubic_region_percent": cubic_region,
        "time_at_rated_power_percent": rated_to_cut_out,
        "time_above_cut_out_percent": above_cut_out,
        "recommendation": ""
    }
    
    if below_cut_in > 25.0:
        diagnostic["recommendation"] = "Wind is frequently below cut-in. Consider a Low Wind Speed Turbine with lower cut-in speed."
    elif rated_to_cut_out < 10.0:
        diagnostic["recommendation"] = "Wind rarely reaches rated power. Generator is likely oversized. Consider larger rotor or smaller generator."
    elif above_cut_out > 5.0:
        diagnostic["recommendation"] = "Wind frequently exceeds cut-out. Extreme wind conditions. Consider High Wind Speed Turbine."
    else:
        diagnostic["recommendation"] = "Turbine is reasonably well-matched for this site."
        
    return wind_dir, wind_freq, wind_speed, diagnostic

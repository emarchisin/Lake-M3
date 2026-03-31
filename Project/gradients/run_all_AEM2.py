import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from processBased_lakeModel_functions import run_wq_model
from model_setup import get_num_data_columns, get_hypsography, get_lake_config, get_model_params, get_run_config, get_ice_and_snow, provide_meteorology, initial_profile, wq_initial_profile, provide_phosphorus, provide_carbon

def melt_var(arr_2d, datetimes, depth, varname):
    arr_2d = np.asarray(arr_2d)

    # force shape (depth , time)
    if arr_2d.shape == (len(datetimes), len(depth)):
        arr_2d = arr_2d.T

    assert arr_2d.shape == (len(depth), len(datetimes)), \
        f"{varname} shape mismatch {arr_2d.shape}"

    df = pd.DataFrame({
        "datetime": np.repeat(datetimes, len(depth)),
        "depth": np.tile(depth, len(datetimes)),
        varname: arr_2d.flatten(order="F")
    })

    return df

lake_dir = Path('Project/gradients')

config_dir = lake_dir / "config"
driver_dir = lake_dir / "drivers"
output_dir = lake_dir / "output"

# --- 1. Define the worker function ---
def process_lake(lake_num):
    lake_config = get_lake_config(config_dir / "lake_config.csv", lake_num)
    model_params = get_model_params(config_dir / "model_params.csv", lake_num)
    run_config = get_run_config(config_dir / "run_config.csv", lake_num)
    ice_and_snow = get_ice_and_snow(config_dir / "ice_and_snow.csv", lake_num)
    
    lake_key = f"{run_config.name}"
    print(f"======= Starting {lake_key} =======")
    
    windfactor = float(lake_config["wind_factor"])
    nx = int(run_config["nx"])
    # dt = float(run_config["dt"])
    dx = float(run_config["dx"])

    area, depth, volume, hypso_weight = get_hypsography(
        hypsofile=driver_dir / run_config['hypso_ini_file'],
        dx=dx, nx=nx, outflow_depth=float(lake_config["outflow_depth"])
    )
  
    desired_start = pd.Timestamp(run_config["start_time"])  
    desired_end = pd.Timestamp(run_config["end_time"])  
    
    startTime = 1 
    startingDate = desired_start
    
    # n_days = (desired_end - desired_start).days + (desired_end - desired_start).seconds / 86400 
    
    # hydrodynamic_timestep = 24 * dt
    # total_runtime = (n_days) * hydrodynamic_timestep / dt  
    
    # endTime = (startTime + total_runtime) 
  
    endingDate = desired_end

    times = pd.date_range(startingDate, endingDate, freq='H')

    # nTotalSteps = int(total_runtime)

    meteo_all = provide_meteorology(
        meteofile=driver_dir / run_config["meteo_ini_file"], 
        windfactor=windfactor, lat=lake_config["Latitude"], lon=lake_config["Longitude"], elev=lake_config["Elevation"],
        startDate=startingDate, endDate=endingDate
    )
                     
    # atm_flux_output = np.zeros(nTotalSteps,) 
    u_ini = initial_profile(
        initfile=driver_dir / run_config["u_ini_file"], nx=nx, dx=dx,
        depth=depth, startDate=startingDate
    ) 
    wq_ini = wq_initial_profile(
        initfile=driver_dir / run_config["wq_ini_file"], nx=nx, dx=dx,
        depth=depth, volume=volume, startDate=startingDate
    )
    tp_boundary = provide_phosphorus(
        tpfile=driver_dir / run_config["tp_ini_file"], 
        startingDate=startingDate, startTime=startTime
    )
    carbon = provide_carbon(
        ocloadfile=driver_dir / run_config["oc_load_file"], 
        startingDate=startingDate, startTime=startTime
    ).dropna(subset=['oc'])

    res = run_wq_model(
        lake_num=lake_num,
        startTime=startingDate,
        endTime=endingDate,
        nx=run_config["nx"],
        dt=run_config["dt"],
        dx=run_config["dx"],
        timelabels=times,  
        pgdl_mode=run_config["pgdl_mode"],
        training_data_path=run_config["training_data_path"],
        diffusion_method=run_config["diffusion_method"],
        scheme=run_config["scheme"],
        area=area,  
        volume=volume,  
        depth=depth,  
        zmax=lake_config['Zmax'],
        outflow_depth=lake_config['outflow_depth'],
        mean_depth=sum(volume) / max(area),
        hypso_weight=hypso_weight,
        altitude=lake_config['Elevation'],
        lat=lake_config['Latitude'],
        long=lake_config['Longitude'],
        u=deepcopy(u_ini),  
        o2=deepcopy(wq_ini[0]),  
        docr=deepcopy(wq_ini[1]) * .75, 
        docl=deepcopy(wq_ini[1]) * .25,
        pocr=0.5 * volume, 
        pocl=0.5 * volume, 
        daily_meteo=meteo_all,
        secview=None,
        phosphorus_data=tp_boundary,
        oc_load_input=carbon,
        ice=ice_and_snow["ice"],
        Hi=ice_and_snow["Hi"],
        Hs=ice_and_snow["Hs"],
        Hsi=ice_and_snow["Hsi"],
        iceT=ice_and_snow["iceT"],
        supercooled=ice_and_snow["supercooled"],
        dt_iceon_avg=ice_and_snow["dt_iceon_avg"],
        Ice_min=ice_and_snow["Ice_min"],
        KEice=ice_and_snow["KEice"],
        rho_snow=ice_and_snow["rho_snow"],
        km=model_params["km"],
        k0=model_params["k0"],
        weight_kz=model_params["weight_kz"],
        piston_velocity=model_params["piston_velocity"] / 86400, 
        Cd=model_params["Cd"],
        hydro_res_time_hr=lake_config["hydro_res_time"] * 8760, 
        W_str=(None if pd.isna(model_params["W_str"]) else model_params["W_str"]),
        denThresh=model_params["denThresh"],
        kd_light=model_params["kd_light"],
        light_water=model_params["light_water"],
        light_doc=model_params["light_doc"],
        light_poc=model_params["light_poc"],
        albedo=lake_config["Albedo"],
        eps=model_params["eps"],
        emissivity=model_params["emissivity"],
        sigma=model_params["sigma"],
        sw_factor=lake_config["sw_factor"],
        wind_factor=lake_config["wind_factor"],
        at_factor=lake_config["at_factor"],
        turb_factor=lake_config["turb_factor"],
        Hgeo=model_params["Hgeo"],
        resp_docr=model_params["resp_docr"] / 86400,
        resp_docl=model_params["resp_docl"] / 86400,
        resp_pocr=model_params["resp_pocr"] / 86400,
        resp_pocl=model_params["resp_pocl"] / 86400,
        resp_poc=model_params["resp_pocl"] / 86400,
        sed_sink=model_params["sed_sink"] / 86400,
        settling_rate_labile=model_params["settling_rate_labile"] / 86400,
        settling_rate_refractory=model_params['settling_rate_refractory'] / 86400,
        sediment_rate=model_params["sediment_rate"] / 86400,
        theta_npp=model_params["theta_npp"],
        theta_r=model_params["theta_r"],
        conversion_constant=model_params["conversion_constant"],
        k_half=model_params["k_half"],
        p_max=model_params["p_max"] / 86400,
        prop_I_npp=model_params['prop_I_npp'],
        k_TP=model_params['k_TP'],
        f_sod=lake_config["f_sod"],
        d_thick=model_params["d_thick"],
        prop_oc_docr=lake_config["prop_oc_docr"],
        prop_oc_docl=lake_config["prop_oc_docl"],
        prop_oc_pocr=lake_config["prop_oc_pocr"],
        prop_oc_pocl=lake_config["prop_oc_pocl"],
        p2=model_params["p2"],
        B=model_params["B"],
        g=model_params["g"],
        meltP=model_params["meltP"],
    )

    res['starttime'] = startingDate
    res['times'] = times
    res['dx'] = dx
    res['nx'] = nx
    res['volume'] = volume
    res['area'] = area
    res['depth'] = depth

    lake_output_dir = output_dir / lake_key
    lake_output_dir.mkdir(exist_ok=True)

    # Model Output
    temp = res["temp"]
    o2 = res["o2"] / volume[:, None]
    docl = res["docl"]
    docr = res["docr"]
    pocl = res["pocl"]
    pocr = res["pocr"]
    npp = res["npp"]
    # atm_flux = res["atm_flux_output"]
    docl_resp = res["docl_respiration"]
    docr_resp = res["docr_respiration"]
    poc_resp = res["poc_respiration"]
    secchi = res["secchi"]
    doc = (res["docl"] + res["docr"]) / volume[:, None]
    poc = (res["pocl"] + res["pocr"]) / volume[:, None]
        
    r_layer = (
            (docl * docl_resp) +
            (docr * docr_resp) +
            (pocl * poc_resp) +
            (pocr * poc_resp))  # g/d per layer
        
    r_layer_m2 = r_layer / area[:, None] #g/m2/d
        
    gpp_layer = npp  # g/d per layer
    gpp_layer_m2 = gpp_layer / area[:, None] #g/m2/d
        
    nep_layer = gpp_layer - r_layer #g/d per layer
    nep_layer_m2 = nep_layer / area[:, None] #g/m2/d
        
    dfs = [
            melt_var(temp, times, depth, "WaterTemp_C"),
            melt_var(o2, times, depth, "Water_DO_mg_per_L"),
            melt_var(doc, times, depth, "Water_DOC_mg_per_L"),
            melt_var(poc, times, depth, "Water_POC_mg_per_L"),
            melt_var(r_layer, times, depth, "Resp_g_per_day"),
            melt_var(r_layer_m2, times, depth, "Resp_g_per_m2_day"),
            melt_var(gpp_layer, times, depth, "GPP_g_per_day"),
            melt_var(gpp_layer_m2, times, depth, "GPP_g_per_m2_day"),  
            melt_var(nep_layer, times, depth, "NEP_g_per_day"),
            melt_var(nep_layer_m2, times, depth, "NEP_g_per_m2_day"),
  ]
    
    fm_lake = dfs[0]
    for df in dfs[1:]:
            fm_lake = fm_lake.merge(df, on=["datetime", "depth"], how="left")
            
    fm_lake["depth"] = fm_lake["depth"] - 0.25
        
    fm_lake.to_parquet(lake_output_dir / f"{lake_key}_model.parquet", index=False, compression='zstd')

    # Driver Output
    meteo = res["meteo_input"]
    secchi = res["secchi"]
    TP = res.get("TP", np.zeros_like(secchi))
    
    fm_driver = pd.DataFrame({
            "datetime": times,
            "Shortwave_Radiation_Downwelling_wattPerMeterSquared": meteo_all["Shortwave_Radiation_Downwelling_wattPerMeterSquared"].values, #input file
            "Longwave_Flux_wattPerMeterSquared": meteo[1, :], #flux calculated in heating res 
            "Air_Temperature_celsius": meteo_all["Air_Temperature_celsius"].values, #input file
            "Ten_Meter_Elevation_Wind_Speed_meterPerSecond": meteo_all["Ten_Meter_Elevation_Wind_Speed_meterPerSecond"].values, #added windfactor
            "Precipitation_millimeterPerDay": meteo_all["Precipitation_millimeterPerDay"].values,#input file
            "Water_Secchi_m": secchi.flatten(),
            "TP_load_ug_per_L": TP.flatten(),})
    

    fm_driver.to_parquet(lake_output_dir / f"{lake_key}_driver.parquet", index=False, compression='zstd')

    return f"======= Completed {lake_key} ======="



# --- 2. Main Execution Block ---
if __name__ == '__main__':
    # Determine the total number of lakes directly from the configs
    num_lakes = get_num_data_columns(config_dir / "run_config.csv", "nx")
    
    # Use CPU count to determine number of workers
    max_workers = multiprocessing.cpu_count()
    
    print(f"Detected {num_lakes} lakes in config.")
    print(f"Starting parallel run across {max_workers} worker processes...")
    
    # Run the pool using indices 1 to num_lakes inclusive
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all lakes to the pool
        futures = {executor.submit(process_lake, i): i for i in range(1, num_lakes + 1)}
        
        for future in as_completed(futures):
            lake_idx = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"❌ Lake index {lake_idx} generated an exception: {exc}")
                
    print("\nAll simulations completed.")
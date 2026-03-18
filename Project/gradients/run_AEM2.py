import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import h5py

from processBased_lakeModel_functions import  run_wq_model
from model_setup import get_hypsography, get_lake_config, get_model_params, get_run_config, get_ice_and_snow , get_num_data_columns, provide_meteorology, initial_profile,  wq_initial_profile, provide_phosphorus, provide_carbon
from hdf5_functions import save_dict_to_hdf5

lake_dir=Path('Project/gradients')

config_dir=lake_dir/"config"
driver_dir=lake_dir/"drivers"
output_dir=lake_dir/"output"


num_lakes = get_num_data_columns(config_dir/"lake_config.csv", "Zmax")
num_lakes = 1
for lake_num in range(1, num_lakes + 1):   
    lake_config = get_lake_config(config_dir/"lake_config.csv", lake_num)
    model_params = get_model_params(config_dir/"model_params.csv", lake_num)
    run_config = get_run_config(config_dir/"run_config.csv", lake_num)
    ice_and_snow = get_ice_and_snow(config_dir/"ice_and_snow.csv", lake_num)
    
    print(f"=======Running {run_config.name}=======")
    
    windfactor = float(model_params["wind_factor"])
    nx = int(run_config["nx"])# number of layers we will have
    dt = float(run_config["dt"])# 24 hours times 60 min/hour times 60 seconds/min to convert s to day
    dx = float(run_config["dx"]) # spatial step

    ## area and depth values of our lake 
    area, depth, volume, hypso_weight = get_hypsography(hypsofile = driver_dir/run_config['hypso_ini_file'],
                            dx = dx, nx = nx, outflow_depth=float(lake_config["outflow_depth"]))
  
                
    ## time step discretization 

    #get start time from input file
    desired_start = pd.Timestamp(run_config["start_time"])  
    desired_end = pd.Timestamp(run_config["end_time"])  
    
    startTime = 1 # RL: SOMEONE SHOULD THINK ABOUT THIS MORE DEEPLY!
    startingDate = desired_start
    
    n_days = (desired_end - desired_start).days + (desired_end - desired_start).seconds/86400 
    
    hydrodynamic_timestep = 24 * dt
    total_runtime =  (n_days) * hydrodynamic_timestep/dt  
    
    endTime =  (startTime + total_runtime) 
  
    endingDate = desired_end

    times = pd.date_range(startingDate, endingDate, freq='H')

    nTotalSteps = int(total_runtime)

    meteo_all = provide_meteorology(meteofile = driver_dir/run_config["meteo_ini_file"], 
                    windfactor = windfactor, lat = lake_config["Latitude"], lon = lake_config["Longitude"], elev = lake_config["Elevation"],
                    startDate = startingDate, endDate=endingDate)
                     
    atm_flux_output = np.zeros(nTotalSteps,) 
    u_ini = initial_profile(initfile = driver_dir/run_config["u_ini_file"], nx = nx, dx = dx,
                     depth = depth,
                     startDate = startingDate) 
    wq_ini = wq_initial_profile(initfile = driver_dir/run_config["wq_ini_file"], nx = nx, dx = dx,
                     depth = depth, 
                     volume = volume,
                     startDate = startingDate)
    tp_boundary = provide_phosphorus(tpfile =  driver_dir/run_config["tp_ini_file"], 
                                 startingDate = startingDate,
                                 startTime = startTime)
    carbon = provide_carbon(ocloadfile =  driver_dir/run_config["oc_load_file"], # RL: carbon driver?
                                 startingDate=startingDate,
                                 startTime = startTime).dropna(subset=['oc'])


    res = run_wq_model(
        # RUNTIME CONFIG
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

        # LAKE CONFIG
        area=area,  # already read
        volume=volume,  # already read
        depth=depth,  # already read
        zmax=lake_config['Zmax'],
        outflow_depth=lake_config['outflow_depth'],
        mean_depth=sum(volume) / max(area),
        hypso_weight=hypso_weight,
        altitude=lake_config['Elevation'],
        lat=lake_config['Latitude'],
        long=lake_config['Longitude'],

        # MODEL PARAMS - initial conditions
        u=deepcopy(u_ini),  # already read
        o2=deepcopy(wq_ini[0]),  # already read
        docr=deepcopy(wq_ini[1])*.75, # split apart doc into docr and docl
        docl=deepcopy(wq_ini[1])*.25,
        pocr=0.5 * volume, #0.5 # set starting pocr and pocl
        pocl=0.5 * volume, #0.5

        # meteorology & boundary forcing
        daily_meteo=meteo_all,
        secview=None,
        phosphorus_data=tp_boundary,
        oc_load_input=carbon,

        # ice & snow dynamics
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
    

        # mixing and physical transport
        km=model_params["km"],
        k0=model_params["k0"],
        weight_kz=model_params["weight_kz"],
        piston_velocity=model_params["piston_velocity"]/86400, # seconds in day
        Cd=model_params["Cd"],
        hydro_res_time_hr=model_params["hydro_res_time"]*8760, # hours in year
        W_str=(
            None if pd.isna(model_params["W_str"])
            else model_params["W_str"]
        ),
        denThresh=model_params["denThresh"],

        # light & heat fluxes
        kd_light=model_params["kd_light"],
        light_water=model_params["light_water"],
        light_doc=model_params["light_doc"],
        light_poc=model_params["light_poc"],
        albedo=lake_config["Albedo"],
        eps=model_params["eps"],
        emissivity=model_params["emissivity"],
        sigma=model_params["sigma"],
        sw_factor=model_params["sw_factor"],
        wind_factor=model_params["wind_factor"],
        at_factor=model_params["at_factor"],
        turb_factor=model_params["turb_factor"],
        Hgeo=model_params["Hgeo"],

        # biogeochemical params 
        ###PCH changed POCr because high mineralization rate of POCr was interferring
        # with other aspects of OC and DO in Peter Lake. We should probably just
        # add another parameter for this.
        resp_docr=model_params["resp_docr"]/86400,
        resp_docl=model_params["resp_docl"]/86400,
        resp_pocr=model_params["resp_pocr"]/86400,
        resp_pocl=model_params["resp_pocl"]/86400,
        resp_poc=model_params["resp_pocl"]/86400,
        sed_sink=model_params["sed_sink"]/86400,
        settling_rate_labile=model_params["settling_rate_labile"]/86400,
        settling_rate_refractory=model_params['settling_rate_refractory']/86400,
        sediment_rate=model_params["sediment_rate"]/86400,
        theta_npp=model_params["theta_npp"],
        theta_r=model_params["theta_r"],
        conversion_constant=model_params["conversion_constant"],
        k_half=model_params["k_half"],
        p_max=model_params["p_max"]/86400,
        prop_I_npp=model_params['prop_I_npp'],
        k_TP=model_params['k_TP'],
        f_sod=model_params["f_sod"],
        d_thick=model_params["d_thick"],
        
       
        # carbon pool partitioning
        prop_oc_docr=model_params["prop_oc_docr"],
        prop_oc_docl=model_params["prop_oc_docl"],
        prop_oc_pocr=model_params["prop_oc_pocr"],
        prop_oc_pocl=model_params["prop_oc_pocl"],

        # general physical constants
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

    # write out output
    lake_key=f"{run_config.name}"
    lake_output_dir=output_dir/lake_key
    lake_output_dir.mkdir(exist_ok=True)
    with h5py.File(lake_output_dir/f"{run_config.name}.h5", "w") as h5f:
      save_dict_to_hdf5(h5f, "/", res)




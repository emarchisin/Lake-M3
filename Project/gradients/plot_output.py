import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from hdf5_functions import load_hdf5_to_dict
from postprocess import save_fig, calc_dens, depth_to_index

def heatmap_plot(data, temp, datetimes, dx, out_dir, lake_key, name=None, ax=None, vmin=None, vmax=None, num_x_ticks=8):
    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
        created_fig = True
    else:
        fig = ax.figure

    sns.heatmap(
        data,
        cmap='Spectral_r',  # Passing the string directly avoids matplotlib deprecation warnings
        xticklabels=False,  # Turned off here since we manually set them below
        yticklabels=False,  
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )

    # Assuming calc_dens is defined elsewhere in your scope
    dens = calc_dens(temp)
    ax.contour(
        np.arange(.5, temp.shape[1]),
        np.arange(.5, temp.shape[0]),
        dens,
        levels=[999],
        colors='black',
        linestyles='dotted'
    )

    ax.set_ylabel("Depth (m)", fontsize=15)

    # X Axis
    step_x = max(1, len(datetimes) // num_x_ticks)
    xticks_ix = np.arange(0, len(datetimes), step_x)
    
    ax.set_xticks(xticks_ix + 0.5)
    
    tick_times = datetimes[xticks_ix]
    ax.set_xticklabels(
        tick_times.strftime("%m/%d/%Y"),
        rotation=45,
        ha='right'
    )

    # Y Axis
    n_depth = data.shape[0]
    step_y = int(round(1 / dx))

    tick_positions = np.arange(0, n_depth, step_y)
    tick_labels = np.round(tick_positions * dx, 0)

    # Add +0.5 to tick positions so they center correctly on the heatmap squares
    ax.set_yticks(tick_positions + 0.5) 
    ax.set_yticklabels(tick_labels, rotation=0)

    # Save only if standalone plot
    if created_fig and name is not None:
        plt.tight_layout()
        # Assuming save_fig is defined elsewhere in your scope
        save_fig(fig, out_dir, lake_key, f"{name}.png")
        plt.close(fig)

    return ax


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
    
depth_group={
        'docl':'doctot',
        'docr':'doctot',
        'doctot':'doctot',
        'pocl':'poctot',
        'pocr':'poctot',
        'poctot':'poctot'}

output_dir=Path('Project/gradients/output')

lake_keys = ["dish_10ha_1mgl_1yr_low_tp_coastal_plains"]

for key in lake_keys:
  
  lake_output_dir=output_dir/key

  with h5py.File(f"{lake_output_dir}/{key}.h5", "r") as h5f:
    res = load_hdf5_to_dict(h5f)

  times=res['times'],
  times_flat = np.ravel(times)
  times_pd = pd.to_datetime(times_flat)
  volume=res['volume']
  depth=res['depth']
  area=res['area']
  dx=res['dx']
  startDate=res['starttime']
  endDate=res['endtime']
  temp = res["temp"]
  o2 = res["o2"]
  docl = res["docl"]
  docr = res["docr"]
  pocl = res["pocl"]
  pocr = res["pocr"]
  npp = res["npp"]
  atm_flux = res["atm_flux_output"]
  docl_resp = res["docl_respiration"]
  docr_resp = res["docr_respiration"]
  poc_resp = res["poc_respiration"]
  secchi = res["secchi"]

  doc_total = docl + docr
  poc_total = pocl + pocr

  variables = {
        "docr": docr,
        "docl": docl,
        "doctot": doc_total,
        "pocr": pocr,
        "pocl": pocl,
        "poctot": poc_total
    }

  # O2
  heatmap_plot(o2 / volume[:, None], res["temp"], times_pd, dx, out_dir=lake_output_dir, lake_key=key, name="o2_heat", vmin=0, vmax=20)

  # Temp
  heatmap_plot(res["temp"], res["temp"], times_pd, dx, out_dir=lake_output_dir, lake_key=key, name="wtemp_heat", vmin=0, vmax=30)

  # Carbon
  for varname, var in variables.items():
      heatmap_plot(var/volume[:,None],res["temp"], times_pd, dx, out_dir=lake_output_dir, lake_key=key, name=f"{varname}_heat", vmin=0, vmax=5)
  
  # Secchi
  fig, ax = plt.subplots(figsize=(10,5))
  ax.plot(times_pd, secchi.T, color='blue', label='Modeled')
  ax.invert_yaxis()
  ax.set_ylabel("Secchi (m)")
  save_fig(fig, lake_output_dir, key, "secchi")

  # Rates Panel
  surf_ix = depth_to_index(depth, 0)
  r_all = (
            (docl * docl_resp) +
            (docr * docr_resp) +
            (pocl * poc_resp) +
            (pocr * poc_resp)
        ) / volume[:,None]

  gpp_all = npp/volume[:,None] + r_all
  r = r_all[surf_ix,:]
  gpp = gpp_all[surf_ix,:]
  atm = atm_flux[0,:] / volume[0]
  fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
  ax[0].plot(times_pd, gpp, color='green')
  ax[0].set_ylabel("GPP (g/m3/d)")
  ax[1].plot(times_pd, r, color='red')
  ax[1].set_ylabel("R (g/m3/d)")
  ax[2].plot(times_pd, atm, color='purple')
  ax[2].set_ylabel("Atm Ex (g/m3/d)")
  save_fig(fig, lake_output_dir, key, "rates_panel")     

  # Save out CSV
  o2 = res["o2"] / volume[:, None]
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
            melt_var(temp, times_pd, depth, "WaterTemp_C"),
            melt_var(o2, times_pd, depth, "Water_DO_mg_per_L"),
            melt_var(doc, times_pd, depth, "Water_DOC_mg_per_L"),
            melt_var(poc, times_pd, depth, "Water_POC_mg_per_L"),
            melt_var(r_layer, times_pd, depth, "Resp_g_per_day"),
            melt_var(r_layer_m2, times_pd, depth, "Resp_g_per_m2_day"),
            melt_var(gpp_layer, times_pd, depth, "GPP_g_per_day"),
            melt_var(gpp_layer_m2, times_pd, depth, "GPP_g_per_m2_day"),  
            melt_var(nep_layer, times_pd, depth, "NEP_g_per_day"),
            melt_var(nep_layer_m2, times_pd, depth, "NEP_g_per_m2_day"),
  ]
    
  fm_lake = dfs[0]
  for df in dfs[1:]:
            fm_lake = fm_lake.merge(df, on=["datetime", "depth"], how="left")
            
  fm_lake["depth"] = fm_lake["depth"] - 0.25
        
  fm_lake.to_csv(lake_output_dir / f"{key}.csv",index=False)

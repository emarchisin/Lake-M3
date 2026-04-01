import pandas as pd
import itertools

# --- Read Templates ---
# Run Config Template
template_run = pd.read_csv('Project/gradients/base_config/run_config.csv')
template_run.set_index('var', inplace=True)
base_run_config = template_run['Lake1'].to_dict()

# Model Params Template
template_param = pd.read_csv('Project/gradients/base_config/model_params.csv')
template_param.set_index('var', inplace=True)
base_param_config = template_param['Lake1'].to_dict()
# Save the description and units to re-attach later
param_metadata = template_param[['Description', 'Units']].copy()

# Lake Config Template
template_lake = pd.read_csv('Project/gradients/base_config/lake_config.csv')
template_lake.set_index('var', inplace=True)
base_lake_config = template_lake['Lake1'].to_dict()
lake_param_metadata = template_lake[['Description', 'Units']].copy()

# Ice & Snow Config Template
template_ice = pd.read_csv('Project/gradients/base_config/ice_and_snow.csv')
template_ice.set_index('var', inplace=True)
base_ice_config = template_ice['Lake1'].to_dict()

# Load the Max Depths Lookup Table ---
max_depths = pd.read_csv("Project/gradients/drivers/lake_volumes.csv")

# --- Set Run Timing ---
custom_start_time = "2/1/16 0:00"
custom_end_time = "1/1/24 0:00"

shapes = ['dish', 'bowl', 'bucket']
areas = [10, 100, 1000]

# Morphometry reference parameters for calculating Zmax
mean_depth_refs = {
    'dish': {10: 1.0, 1000: 3.0},
    'bowl': {10: 3.0, 1000: 10.0},
    'bucket': {10: 10.0, 1000: 30.0}
}
shape_p = {'dish': 2.0, 'bowl': 1.0, 'bucket': 0.2}

# Outflow depth to Zmax ratio based on template (6.5 / 25)
outflow_ratio = 6.5 / 25.0

# residence times
rts = [1, 5, 10]

meteos = [
    'met/coastal_plains_2016-2024.csv',
    'met/southern_appalachians_2016-2024.csv',
    'met/xeric_2016-2024.csv',
    'met/southern_plains_2016-2024.csv',
    'met/northern_plains_2016-2024.csv',
    'met/western_mountains_2016-2024.csv',
    'met/temperate_plains_2016-2024.csv',
    'met/northern_appalachians_2016-2024.csv'
]

# Representative locations for the 8 ecoregions
# (Latitude, Longitude, Elevation_m)
ecoregion_geo_map = {
    'coastal_plains': (33.0, -80.0, 50),
    'southern_appalachians': (35.5, -83.5, 800),
    'xeric': (36.0, -115.0, 600),
    'southern_plains': (33.0, -100.0, 500),
    'northern_plains': (45.0, -100.0, 500),
    'western_mountains': (40.0, -106.0, 2500),
    'temperate_plains': (41.0, -93.0, 300),
    'northern_appalachians': (44.0, -71.0, 400)
}

# (tp_file, oc_mgL, f_sod) pairings
tp_oc_pairs = [
    ('tp/low_tp.csv', 1, 2e-6),
    ('tp/med_tp.csv', 1, 7e-6),
    ('tp/high_tp.csv', 15, 2e-5),
    ('tp/med_tp.csv', 30, 2e-5)
]

# Generate the 864 combinations
combinations = list(itertools.product(shapes, areas, rts, tp_oc_pairs, meteos))

# --- Build the Dictionaries ---
new_run_config = {}
new_param_config = {}
new_lake_config = {}
new_ice_config = {}

for combo in combinations:
    shape, area, rt, tp_oc_pair, meteo = combo
    tp, oc, f_sod = tp_oc_pair
    
    tp_name = tp.split('/')[-1].replace('.csv', '')
    meteo_name = meteo.split('/')[-1].replace('_2016-2024.csv', '')
    
    lake_name = f"{shape}_{area}ha_{oc}mgl_{rt}yr_{tp_name}_{meteo_name}"

    # Get max depth for lake
    match = max_depths[(max_depths['shape'] == shape) & (max_depths['area_ha'] == area)]
    z_max = match['zmax'].values[0]
    
    # Update Run Config 
    r_config = base_run_config.copy()
    r_config['start_time'] = custom_start_time
    r_config['end_time'] = custom_end_time
    r_config['nx'] = round(z_max) * 2 # keeping spatial step constant and .5 m, set spatial extent based on max depth
    r_config['hypso_ini_file'] = f"bath/{shape}_{area}ha.csv"
    r_config['oc_load_file'] = f"oc_load/{shape}_{area}ha_{oc}mgl_{rt}yr.csv"
    r_config['tp_ini_file'] = tp
    r_config['meteo_ini_file'] = meteo
    new_run_config[lake_name] = r_config
    
    # Update Model Params 
    p_config = base_param_config.copy()
    new_param_config[lake_name] = p_config
    
    # Update Lake Config 
    l_config = base_lake_config.copy()
    
    # Get locations for the current met ecoregion
    lat, lon, elev = ecoregion_geo_map[meteo_name]
    l_config['Latitude'] = lat
    l_config['Longitude'] = lon
    l_config['Elevation'] = elev
    l_config['hydro_res_time'] = rt
    l_config['f_sod'] = f_sod

    # Scaled Morphometry
    l_config['Zmax'] = round(z_max, 2)
    l_config['outflow_depth'] = round(z_max * outflow_ratio, 2)
    
    new_lake_config[lake_name] = l_config

    new_ice_config[lake_name] = base_ice_config.copy()


# --- Export DataFrames to CSV ---
# Run Config
final_run_df = pd.DataFrame(new_run_config)
final_run_df.index.name = 'var'
final_run_df.reset_index(inplace=True)
final_run_df.to_csv("Project/gradients/config/run_config.csv", index=False)

# Model Params
final_param_df = pd.DataFrame(new_param_config)
final_param_df = pd.concat([param_metadata, final_param_df], axis=1)
final_param_df.index.name = 'var'
final_param_df.reset_index(inplace=True)
final_param_df.to_csv("Project/gradients/config/model_params.csv", index=False)

# Lake Config
final_lake_df = pd.DataFrame(new_lake_config)
final_lake_df = pd.concat([lake_param_metadata, final_lake_df], axis=1)
final_lake_df.index.name = 'var'
final_lake_df.reset_index(inplace=True)
final_lake_df.to_csv("Project/gradients/config/lake_config.csv", index=False)

# Ice and Snow Config
final_ice_df = pd.DataFrame(new_ice_config)
final_ice_df.index.name = 'var'
final_ice_df.reset_index(inplace=True)
final_ice_df.to_csv("Project/gradients/config/ice_and_snow.csv", index=False)
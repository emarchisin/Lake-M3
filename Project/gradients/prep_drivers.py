import pandas as pd
import numpy as np

## TP
high_tp = pd.read_csv("Project/gradients/drivers/tp/high_tp.csv")

#medium TP (50% of mendota)
high_tp.assign(tp = high_tp["tp"] / 2).to_csv("Project/gradients/drivers/tp/med_tp.csv", index=False)

#low TP (10% of mendota)
high_tp.assign(tp = high_tp["tp"] / 10).to_csv("Project/gradients/drivers/tp/low_tp.csv", index=False)

def generate_bathymetry(surface_area_ha, lake_shape):
    """
    Generates a lake bathymetry profile based on surface area and shape.
    
    Parameters:
    surface_area_ha (float): Surface area of the lake in Hectares (Ha).
    lake_shape (str): The morphometry of the lake ('dish', 'bowl', or 'bucket').
    
    Returns:
    pd.DataFrame: A DataFrame with 'Depth_meter' and 'Area_meterSquared'.
    """
    # Reference mean depths for 10 Ha and 1000 Ha
    mean_depth_refs = {
        'dish': {10: 1.0, 1000: 3.0},
        'bowl': {10: 3.0, 1000: 10.0},
        'bucket': {10: 10.0, 1000: 30.0}
    }
    
    # Shape parameter `p` where Area(z) = A_0 * (1 - z / z_max)^p
    # Mean depth (z_mean) = z_max / (p + 1) => z_max = z_mean * (p + 1)
    shape_p = {
        'dish': 2.0,   # Cone-like
        'bowl': 1.0,   # Paraboloid
        'bucket': 0.2  # U-shaped/Cylindrical
    }
    
    if lake_shape.lower() not in mean_depth_refs:
        raise ValueError("lake_shape must be 'dish', 'bowl', or 'bucket'")
        
    lake_shape = lake_shape.lower()
    
    # Calculate the mean depth using log-log interpolation
    ref = mean_depth_refs[lake_shape]
    log_a10, log_a1000 = np.log(10), np.log(1000)
    log_d10, log_d1000 = np.log(ref[10]), np.log(ref[1000])
    
    log_area = np.log(surface_area_ha)
    log_mean_depth = log_d10 + (log_area - log_a10) * (log_d1000 - log_d10) / (log_a1000 - log_a10)
    mean_depth = np.exp(log_mean_depth)
    
    # Determine max depth based on shape parameter
    p = shape_p[lake_shape]
    z_max = mean_depth * (p + 1)
    
    # Convert surface area from Hectares to square meters
    A0 = surface_area_ha * 10000.0 
    
    # Create depth array (1-meter intervals)
    max_depth_int = int(np.ceil(z_max))
    depths = np.arange(0, max_depth_int + 1)
    
    # Calculate areas for each depth step
    # Ignore complex numbers for negative bases by using np.maximum
    relative_depth = np.maximum(0, 1 - depths / z_max)
    areas = A0 * (relative_depth ** p)
    
    # Ensure final depth area is exactly 0 if depth exceeds or equals z_max
    areas[depths >= z_max] = 0.0
    
    # Build and return DataFrame
    df = pd.DataFrame({
        'Depth_meter': depths,
        'Area_meterSquared': np.round(areas, 2)
    })

    # Drop multiple zero-area rows ---
    zero_indices = df[df['Area_meterSquared'] == 0].index
    if len(zero_indices) > 0:
        # Keep everything up to the first time area hits 0
        first_zero_idx = zero_indices[0]
        df = df.loc[:first_zero_idx]
    
    return df

# Set up combinations
shapes = ['dish', 'bowl', 'bucket']
areas_ha = [10, 100, 1000]

# Initialize a list to hold the volume/max_depth calculations
volume_data = []

# Generate and save a CSV for each combination
for shape in shapes:
    for area in areas_ha:
        df_bath = generate_bathymetry(area, shape)
        
        # Calculate volume using the trapezoidal rule (integration of Area over Depth)
        # Result is in cubic meters (m^3)
        volume_m3 = np.trapz(y=df_bath['Area_meterSquared'], x=df_bath['Depth_meter'])

        max_depth = df_bath['Depth_meter'].max()
        
        # Add to tracking list
        volume_data.append({
            'shape': shape,
            'area_ha': area,
            'volume_m3': volume_m3,
            'zmax': max_depth
        })

        # Format the filename
        filename = f"Project/gradients/drivers/bath/{shape}_{area}ha.csv"
        
        # Save to CSV
        df_bath.to_csv(filename, index=False)

# Compile into a master volumes DataFrame and save
df_volumes = pd.DataFrame(volume_data)
df_volumes.to_csv("Project/gradients/drivers/lake_volumes.csv", index=False)

# Define the new parameter options
oc_loads = [1, 15, 30]
residence_times_yr = [1, 5, 10]

# Generate daily dates from Jan 1, 2016 to Jan 1, 2026
date_series = pd.date_range(start="2016-01-01", end="2026-01-01", freq="D")

for index, row in df_volumes.iterrows():
    shape = row['shape']
    area = row['area_ha']
    volume = row['volume_m3']
    
    for oc in oc_loads:
        for rt in residence_times_yr:
            # Calculate daily discharge (assuming flat 365 days/year for simplicity)
            days_in_residence = rt * 365
            daily_discharge = volume / days_in_residence
            
            # Build the new dataframe
            df_load = pd.DataFrame({
                'datetime': date_series,
                'oc': oc,             # Broadcasts the constant OC value to all rows
                'discharge': daily_discharge  # Broadcasts the constant discharge to all rows
            })
            
            # Format a descriptive filename
            filename = f"Project/gradients/drivers/oc_load/{shape}_{area}ha_{oc}mgl_{rt}yr.csv"
            
            df_load.to_csv(filename, index=False)



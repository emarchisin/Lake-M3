import numpy as np
import pandas as pd
import h5py

def save_dict_to_hdf5(h5file, path, dic):
    for key, item in dic.items():
        new_path = f"{path}/{key}".strip("/")
        
        # 1. Standardize Lists
        is_original_list = False
        if isinstance(item, list):
            item = np.array(item)
            is_original_list = True
            
        # 2. Handle Nested Dictionaries
        if isinstance(item, dict):
            h5file.create_group(new_path)
            save_dict_to_hdf5(h5file, new_path, item)
            
        # 3. Handle Pandas DataFrames
        elif isinstance(item, pd.DataFrame):
            df_group = h5file.create_group(new_path)
            df_group.attrs["_type"] = "pandas_dataframe"
            
            # Pass columns and index directly to be caught by the logic below!
            df_dict = {str(col): item[col] for col in item.columns}
            df_dict["_index"] = item.index
            save_dict_to_hdf5(h5file, new_path, df_dict)
            
        # 4. Handle ANY Datetime Collection (DatetimeIndex, Series of dates, Datetime arrays)
        elif pd.api.types.is_datetime64_any_dtype(item) or \
             (isinstance(item, np.ndarray) and item.dtype == 'O' and len(item) > 0 and isinstance(item[0], (pd.Timestamp))):
             
            dt_item = pd.to_datetime(item)
            
            # --- THE FIX ---
            # .to_numpy() removes the Pandas Index/Series wrapper, leaving a pure C-compatible NumPy array
            str_data = dt_item.to_numpy().astype(str).astype('S') 
            
            ds = h5file.create_dataset(new_path, data=str_data, compression="gzip")
            ds.attrs["_type"] = "datetime_array"
            
            if is_original_list:
                ds.attrs["_was_list"] = True
            
        # 5. Handle remaining Pandas Series / Index
        elif isinstance(item, (pd.Series, pd.Index)):
            # Convert to ndarray and loop it back through to catch NaNs/Objects
            save_dict_to_hdf5(h5file, path, {key: item.to_numpy()})
            
        # 6. Handle NumPy Arrays (and former lists)
        elif isinstance(item, np.ndarray):
            if item.dtype == 'O':
                try:
                    clean_item = item.astype(float)
                    ds = h5file.create_dataset(new_path, data=clean_item, compression="gzip")
                except (ValueError, TypeError):
                    str_item = item.astype(str).astype('S')
                    ds = h5file.create_dataset(new_path, data=str_item, compression="gzip")
            else:
                ds = h5file.create_dataset(new_path, data=item, compression="gzip")
            
            if is_original_list:
                ds.attrs["_was_list"] = True
                
        # 7. Handle Single Datetimes
        elif isinstance(item, (pd.Timestamp)):
            h5file.attrs[key] = item.isoformat()
            h5file.attrs[f"{key}_type"] = "datetime"
            
        # 8. Handle Basic Types
        else:
            h5file.attrs[key] = item

def load_hdf5_to_dict(h5_item):
    rebuilt_dict = {}

    # 1. Load Attributes (handle single timestamps)
    for key, value in h5_item.attrs.items():
        if key.endswith("_type") and value == "datetime":
            continue 
        if h5_item.attrs.get(f"{key}_type") == "datetime":
            rebuilt_dict[key] = pd.Timestamp(value)
        else:
            rebuilt_dict[key] = value

    # 2. Load Groups and Datasets
    for key, item in h5_item.items():
        if isinstance(item, h5py.Group):
            if item.attrs.get("_type") == "pandas_dataframe":
                df_components = load_hdf5_to_dict(item)
                index = df_components.pop("_index", None)
                
                # Decode string columns from bytes
                for k, v in df_components.items():
                    if isinstance(v, np.ndarray) and v.dtype.kind == 'S':
                        df_components[k] = np.char.decode(v, 'utf-8')
                        
                if isinstance(index, np.ndarray) and index.dtype.kind == 'S':
                    index = np.char.decode(index, 'utf-8')
                
                rebuilt_dict[key] = pd.DataFrame(df_components, index=index)
            else:
                rebuilt_dict[key] = load_hdf5_to_dict(item)
            
        elif isinstance(item, h5py.Dataset):
            # Handle Datetime Arrays
            if item.attrs.get("_type") == "datetime_array":
                # Vectorized byte decoding
                str_data = np.char.decode(item[()], 'utf-8')
                parsed_times = pd.to_datetime(str_data)
                
                if item.attrs.get("_was_list"):
                    rebuilt_dict[key] = parsed_times.to_list()
                else:
                    rebuilt_dict[key] = np.array(parsed_times.to_list(), dtype=object)
            
            # Handle standard Data
            else:
                loaded_data = item[()]
                if item.attrs.get("_was_list"):
                    if loaded_data.dtype.kind in 'fc': # If it's floats, restore None
                        rebuilt_dict[key] = [None if np.isnan(x) else x for x in loaded_data]
                    else:
                        rebuilt_dict[key] = loaded_data.tolist()
                else:
                    rebuilt_dict[key] = loaded_data

    return rebuilt_dict
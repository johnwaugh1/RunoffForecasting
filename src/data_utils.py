import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np

def load_usgs_data(station_folder):
    """
    Load USGS observation data for a specific station
    
    Args:
        station_folder: Path to the station folder
        
    Returns:
        DataFrame with USGS observations
    """
    # Find the USGS observation file by pattern (adjusted for the new naming convention)
    usgs_files = glob.glob(os.path.join(station_folder, "*_Strt_*.csv"))
    
    if len(usgs_files) == 0:
        raise ValueError("No USGS observation files found")
    
    # Assuming only one file per station
    usgs_file = usgs_files[0]
    
    # Load the data
    usgs_df = pd.read_csv(usgs_file)
    
    # Convert DateTime to pandas datetime (assuming the DateTime column is present)
    usgs_df['DateTime'] = pd.to_datetime(usgs_df['DateTime'])
    
    # Set DateTime as index
    usgs_df.set_index('DateTime', inplace=True)
    
    return usgs_df

def load_nwm_forecasts(station_folder):
    """
    Load all NWM forecast files for a specific station
    
    Args:
        station_folder: Path to the station folder
        
    Returns:
        DataFrame containing all NWM forecasts
    """
    # Find all NWM forecast files inside the 'nwm_forecasts' folder
    nwm_forecasts_folder = os.path.join(station_folder, "nwm_forecasts")
    nwm_files = glob.glob(os.path.join(nwm_forecasts_folder, "streamflow_*.csv"))
    
    if len(nwm_files) == 0:
        raise ValueError("No NWM forecast files found in 'nwm_forecasts' folder")
    
    all_forecasts = []
    
    for file in nwm_files:
        try:
            # Load each forecast file
            forecast_df = pd.read_csv(file)
            
            # Convert times to datetime (adjust the column names if necessary)
            forecast_df['model_initialization_time'] = pd.to_datetime(
                forecast_df['model_initialization_time'], 
                format='%Y-%m-%d_%H:%M:%S'
            )
            forecast_df['model_output_valid_time'] = pd.to_datetime(
                forecast_df['model_output_valid_time'], 
                format='%Y-%m-%d_%H:%M:%S'
            )
            
            # Calculate lead time in hours
            forecast_df['lead_time'] = (
                forecast_df['model_output_valid_time'] - 
                forecast_df['model_initialization_time']
            ).dt.total_seconds() / 3600
            
            all_forecasts.append(forecast_df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    # Combine all forecasts
    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        return combined_forecasts
    else:
        raise ValueError("No valid forecast files found")

def align_forecast_observation(nwm_df, usgs_df):
    """
    Align NWM forecasts with USGS observations
    
    Args:
        nwm_df: DataFrame containing NWM forecasts
        usgs_df: DataFrame containing USGS observations
        
    Returns:
        DataFrame with aligned forecasts and observations
    """
    # Create an empty list to store aligned data
    aligned_data = []
    
    # Get unique initialization times
    init_times = nwm_df['model_initialization_time'].unique()
    
    for init_time in init_times:
        # Get forecasts for this initialization time
        forecast = nwm_df[nwm_df['model_initialization_time'] == init_time]
        
        for _, row in forecast.iterrows():
            valid_time = row['model_output_valid_time']
            lead_time = row['lead_time']
            forecast_value = row['streamflow_value']
            
            # Find the closest observation time (within 30 minutes)
            time_diff = (usgs_df.index - valid_time).total_seconds().abs()
            closest_idx = time_diff.argmin()
            
            if time_diff[closest_idx] <= 1800:  # Within 30 minutes
                obs_time = usgs_df.index[closest_idx]
                obs_value = usgs_df.loc[obs_time, 'USGSFlowValue']
                
                aligned_data.append({
                    'initialization_time': init_time,
                    'valid_time': valid_time,
                    'observation_time': obs_time,
                    'lead_time': lead_time,
                    'forecast_value': forecast_value,
                    'observed_value': obs_value,
                    'error': forecast_value - obs_value  # Calculate error
                })
    
    # Convert to DataFrame
    aligned_df = pd.DataFrame(aligned_data)
    return aligned_df

def calculate_metrics(observed, forecasted):
    """
    Calculate standard hydrologic performance metrics
    
    Args:
        observed: Array-like of observed values
        forecasted: Array-like of forecasted values
        
    Returns:
        Dictionary of metrics
    """
    # Remove any pairs with NaN values
    valid_idx = ~(np.isnan(observed) | np.isnan(forecasted))
    obs = np.array(observed[valid_idx])
    fore = np.array(forecasted[valid_idx])
    
    if len(obs) == 0:
        return {
            'CC': np.nan,
            'RMSE': np.nan,
            'PBIAS': np.nan,
            'NSE': np.nan
        }
    
    # Coefficient of correlation
    CC = np.corrcoef(obs, fore)[0, 1]
    
    # Root mean square error
    RMSE = np.sqrt(np.mean((fore - obs) ** 2))
    
    # Percent bias
    PBIAS = 100 * np.sum(fore - obs) / np.sum(obs)
    
    # Nash-Sutcliffe Efficiency
    NSE = 1 - (np.sum((fore - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
    
    return {
        'CC': CC,
        'RMSE': RMSE,
        'PBIAS': PBIAS,
        'NSE': NSE
    }

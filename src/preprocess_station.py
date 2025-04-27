import pandas as pd

def preprocess_station(station_dict):
    nwm_df = station_dict['nwm']
    usgs_df = station_dict['usgs']

    # Bring DateTime back as a column if it's the index
    if usgs_df.index.name == 'DateTime':
        usgs_df = usgs_df.reset_index()

    print("NWM columns:", nwm_df.columns)
    print("USGS columns:", usgs_df.columns)

    # Handle datetime conversion and strip timezone
    usgs_df['DateTime'] = pd.to_datetime(usgs_df['DateTime']).dt.round('h')
    usgs_df['DateTime'] = usgs_df['DateTime'].dt.tz_localize(None)

    nwm_df['model_output_valid_time'] = pd.to_datetime(nwm_df['model_output_valid_time'])
    nwm_df['model_output_valid_time'] = nwm_df['model_output_valid_time'].dt.tz_localize(None)

    # Find the streamflow column in NWM data
    flow_col = None
    for col in nwm_df.columns:
        if 'streamflow' in col.lower():
            flow_col = col
            break

    if flow_col is None:
        raise KeyError("Could not find a streamflow column in NWM data")

    merged = pd.merge(
        nwm_df,
        usgs_df,
        how='inner',
        left_on='model_output_valid_time',
        right_on='DateTime'
    )

    merged = merged[[
        'model_initialization_time',
        'model_output_valid_time',
        flow_col,
        'USGSFlowValue'
    ]]

    merged = merged.rename(columns={
        flow_col: 'NWM_streamflow',
        'USGSFlowValue': 'USGS_streamflow'
    })

    return merged

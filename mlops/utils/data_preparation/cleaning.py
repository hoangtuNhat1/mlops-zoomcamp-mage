import pandas as pd 

def clean(df: pd.DataFrame, include_extreme_durations: bool = False,) -> pd.DataFrame : 
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    if not include_extreme_durations : 
        df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
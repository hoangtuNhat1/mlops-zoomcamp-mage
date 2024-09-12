# Import necessary libraries
import pandas as pd
from mage_ai.data_preparation.decorators import data_loader

# Define data ingest block with the `data_loader` decorator
@data_loader
def load_data(*args, **kwargs):
    ilename = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet'
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        
        # Correcting datetime columns for CSV data
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
        
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename, engine='pyarrow')  # ensure 'pyarrow' or 'fastparquet' is installed
        
        # Correcting datetime columns for Parquet data
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    
    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter the duration to be between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert specific columns to categorical types
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

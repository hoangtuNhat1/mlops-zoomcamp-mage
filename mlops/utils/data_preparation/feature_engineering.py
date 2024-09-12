import pandas as pd 

from pandas import DataFrame 
def combine_features(df: DataFrame) -> DataFrame : 
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df 
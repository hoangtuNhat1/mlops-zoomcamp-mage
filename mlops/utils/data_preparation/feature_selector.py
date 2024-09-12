categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']
target = ['duration']
import pandas as pd 
from typing import List, Optional 
def select_feature(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame : 
    columns = categorical + numerical + target
    if features : 
        columns += features
    return df[columns]
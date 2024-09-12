import pandas as pd
from typing import Tuple

def split_data(df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into training and validation sets.

    Parameters:
        df (pd.DataFrame): The dataframe to split.
        val_size (float): Proportion of the data to be used for validation (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation dataframes.
    """
    # Shuffle and split the data into training and validation sets
    df_train = df.sample(frac=(1 - val_size), random_state=random_state)
    df_val = df.drop(df_train.index)

    return df_train, df_val

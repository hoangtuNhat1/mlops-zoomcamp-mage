if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd 
from utils.data_preparation.cleaning import clean 
from utils.data_preparation.feature_engineering import combine_feature
from utils.data_preparation.feature_selector import select_feature
from utils.data_preparation.split_data import split_data
from typing import Tuple
@transformer
def transform(df: pd.DataFrame, **kwargs)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    df = clean(df) 
    df = combine_feature(df)
    df = select_feature(df)
    df_train, df_test = split_data(df)
    return df, df_train, df_test



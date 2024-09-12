if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
from typing import Tuple, Dict, Union
from xgboost import Booster
from scipy.sparse import csr_matrix
from pandas import Series

@custom
def source(
    training_result: Tuple[Booster, csr_matrix, Series], 
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],
        csr_matrix,
        Series,
    ],
    **kwargs) -> Tuple[Booster, csr_matrix, csr_matrix]:
    
    _, X_train, y_train = settings
    model, _ = training_result
    return model, X_train, y_train




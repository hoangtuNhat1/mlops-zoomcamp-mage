from sklearn.feature_extraction import DictVectorizer
from typing import List, Dict, Tuple, Optional
import pandas as pd

def vectorize_feature(training_set: pd.DataFrame, validation_set: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Fit the DictVectorizer on the training set and transform both training and validation sets.

    Args:
        training_set (pd.DataFrame): The training dataset.
        validation_set (pd.DataFrame): The validation dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed training and validation datasets.
    """
    dv = DictVectorizer()
    train_dicts = training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    
    X_val = None 
    if validation_set is not None : 
        val_dicts = validation_set[training_set.columns].to_dict(orient='records')
        X_val = dv.transform(val_dicts)
    return X_train, X_val, dv
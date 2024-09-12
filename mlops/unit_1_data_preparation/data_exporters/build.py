if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
from utils.data_preparation.encoders import vectorize_feature

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    df, df_train, df_val = data
    
    # Debug: Check the shape of df and df_val
    X, _, _ = vectorize_feature(df)
    X_train, X_val, dv = vectorize_feature(df_train, df_val)
    
    # Debug: Check the shape of X and X_val after vectorization
    
    target = 'duration'
    y = df[target]
    y_train = df_train[target]
    y_val = df_val[target]
    
    # Debug: Check the shape of y_train and y_val
    
    return X, y, X_train, y_train, X_val, y_val, dv
@test
def test_dataset(X, y, X_train, y_train, X_val, y_val, dv) -> None:

    assert X.shape[0] == 977, f'Expected 983 samples in X_train, but got {X_train.shape[0]}'
    assert X.shape[1] == 727, f'Expected 693 features in X, but got {X.shape[1]}'
    assert len(y) == X.shape[0], f'Expected 786 samples in y, but got {len(y)}'
@test
def test_train_dataset(X, y, X_train, y_train, X_val, y_val, dv) -> None:

    assert X_train.shape[0] == 782, f'Expected 983 samples in X_train, but got {X_train.shape[0]}'
    assert X_train.shape[1] == 609, f'Expected 693 features in X, but got {X.shape[1]}'
    assert len(y_train) == X_train.shape[0], f'Expected 786 samples in y, but got {len(y)}'
@test
def test_train_dataset(X, y, X_train, y_train, X_val, y_val, dv) -> None:

    assert X_val.shape[0] == 195, f'Expected 983 samples in X_train, but got {X_train.shape[0]}'
    assert X_val.shape[1] == 609, f'Expected 693 features in X, but got {X.shape[1]}'
    assert len(y_val) == X_val.shape[0], f'Expected 786 samples in y, but got {len(y)}'

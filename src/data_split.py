import pandas as pd
import numpy as np

def random_train_test_split(X: pd.DataFrame, 
                            y: pd.Series, 
                            test_size: float = 0.2
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Randomly split dataset into train and test subsets.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    elif test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(X))
    train_end = int(len(X) * (1 - test_size))
    train_idx = shuffled_idx[:train_end]
    test_idx = shuffled_idx[train_end:]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    return X_train, X_test, y_train, y_test

def random_val_split(X: pd.DataFrame,
                     y: pd.Series, 
                     val_size: float = 0.1, 
                     test_size: float = 0.2
                     )-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:    
    """
    Randomly split dataset into train, validation and test subsets.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    elif val_size + test_size > 1:
        raise ValueError("val_size + test_size must be less than or equal to 1")
    elif val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative")
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(X))
    train_end = int(len(X) * (1 - test_size - val_size))
    val_end = int(len(X) * (1 - test_size))

    train_idx = shuffled_idx[:train_end]
    val_idx = shuffled_idx[train_end:val_end]
    test_idx = shuffled_idx[val_end:]

    return X.loc[train_idx], X.loc[val_idx], X.loc[test_idx], y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
import pandas as pd
import numpy as np

def random_train_test_split(X: pd.DataFrame, 
                            y: pd.Series, 
                            test_size: float = 0.2,
                            random_state: int = 42
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Randomly split dataset into train and test subsets.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    elif test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1")
    
    np.random.seed(random_state)

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
                     test_size: float = 0.2,
                     random_state: int = 42
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
    
    np.random.seed(random_state)
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    shuffled_idx = np.random.permutation(len(X))
    train_end = int(len(X) * (1 - test_size - val_size))
    val_end = int(len(X) * (1 - test_size))

    train_idx = shuffled_idx[:train_end]
    val_idx = shuffled_idx[train_end:val_end]
    test_idx = shuffled_idx[val_end:]

    return X.loc[train_idx], X.loc[val_idx], X.loc[test_idx], y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]

def date_based_split(X: pd.DataFrame, 
                      y: pd.Series, 
                      date_col: str, 
                      date_split: str,
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Randomly split dataset into train and test subsets based on date.
    """
    if date_col not in X.columns:
        raise ValueError(f"{date_col} is not a valid column name")
    elif len(X) != len(y):
        raise ValueError("X and y must have the same length")
    
    if not pd.api.types.is_datetime64_dtype(X[date_col]):
        try:
            X[date_col] = pd.to_datetime(X[date_col], errors='raise')
        except Exception as e:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime: {e}")
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    train_idx = X[X[date_col] < date_split].index
    test_idx = X[X[date_col] >= date_split].index

    if len(train_idx) == 0 or len(test_idx) == 0:
        return None, None, None, None
    if len(train_idx) == len(X) or len(test_idx) == len(X):
        if len(train_idx) == len(X):
            return X, None, y, None
        else:
            return None, X, None, y
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

def date_based_val_split(X: pd.DataFrame, 
                      y: pd.Series, 
                      date_col: str, 
                      validation_date: str,
                      test_date: str,
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Randomly split dataset into train, validation and test subsets based on date.
    """
    if date_col not in X.columns:
        raise ValueError(f"{date_col} is not a valid column name")
    elif len(X) != len(y):
        raise ValueError("X and y must have the same length")
    
    if validation_date > test_date:
        raise ValueError(
            f"validation_date ({validation_date}) must be before test_date ({test_date})"
        )
    
    if not pd.api.types.is_datetime64_dtype(X[date_col]):
        try:
            X[date_col] = pd.to_datetime(X[date_col], errors='raise')
        except Exception as e:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime: {e}")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    train_idx = X[X[date_col] < validation_date].index
    val_idx = X[(X[date_col] < test_date) & (X[date_col] >= validation_date)].index
    test_idx = X[X[date_col] >= test_date].index

    return X.loc[train_idx], X.loc[val_idx], X.loc[test_idx], y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
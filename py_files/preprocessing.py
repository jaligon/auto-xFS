from typing import Tuple

import numpy as np
import pandas as pd


def to_ndarray(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
	"""
    Convert pandas DataFrame and Series to numpy ndarrays.

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to be converted to a numpy ndarray. Each row in the DataFrame represents a sample,
        and each column represents a feature.
    y : pd.Series
        The Series to be converted to a numpy ndarray. Each element in the Series represents a target value.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy ndarrays. The first ndarray is the converted X, and the second ndarray is the converted y.
    """
	
	"""
	Convert X and y to numpy ndarray
	and return the converted ndarrays as a tuple
	"""
	X_array = X.to_numpy()
	y_array = y.to_numpy()
	return (X_array.astype(np.float32), y_array.astype(np.float32))

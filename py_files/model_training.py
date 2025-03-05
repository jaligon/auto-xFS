from typing import Type
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from tensorflow.keras import Model


def create_init_params(X_selected: pd.DataFrame, tf_model_init_params: dict):
	init_params = []
	for i in range(len(tf_model_init_params)):
		layer_class = tf_model_init_params[i]["layer_class"]
		params = tf_model_init_params[i]["params"]
		if i == 0:
			layer = layer_class(**params, input_shape=(X_selected.shape[1],))
		else:
			layer = layer_class(**params)
		init_params.append(layer)
	return init_params

def train_one_model(X: pd.DataFrame, y: pd.Series, features_selected: np.ndarray,
                    model_class: Union[Type[BaseEstimator], Type[Model]],
                    tf_model_init_params: dict = None, tf_model_compile_params: dict = None,
                    tf_model_fit_params: dict = None, model_kwargs: dict = None) -> dict:
	features = X.columns[features_selected]
	X_temp = X[features]
	if not issubclass(model_class, Model):
		model = model_class(**model_kwargs)
		model.fit(X_temp.values, y)
	else:
		model = model_class(create_init_params(X_temp, tf_model_init_params))
		model.compile(**tf_model_compile_params)
		model.build()
		model.fit(X_temp.values, y, **tf_model_fit_params)
	y_pred = model.predict(X_temp.values)
	return {"y_pred": y_pred, "model_trained": model}

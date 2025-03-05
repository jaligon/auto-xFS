from typing import Type

import numpy as np
import pandas as pd
from BorutaShap import BorutaShap
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from py_files.preprocessing import to_ndarray


def select_features_by_model(X: pd.DataFrame, y: pd.Series, rank: np.ndarray,
                             model_class: Type[BaseEstimator]) -> np.ndarray:
	features_selected = []
	max_score = 0
	features_temp = []
	for i in range(len(rank)):
		features_temp.append(X.columns[rank[i]])
		X_temp = X[features_temp]
		model = model_class(random_state=42)
		model.fit(X_temp, y)
		score = model.score(X_temp, y)
		if score > max_score:
			features_selected = features_temp.copy()
			max_score = score
	features_selected_mask = np.isin(X.columns, features_selected)
	return features_selected_mask


def select_features_by_percentage(X: pd.DataFrame, rank: np.ndarray, percentage: float) -> np.ndarray:
	count = int(len(X.columns) * (percentage / 100))
	features_selected = []
	for i in range(count):
		features_selected.append(X.columns[rank[i]])
	features_selected_mask = np.isin(X.columns, features_selected)
	return features_selected_mask


def calculate_fisher_score(X: pd.DataFrame, y: pd.Series) -> dict:
	X_array, y_array = to_ndarray(X, y)
	fisher = fisher_score.fisher_score(X_array, y_array)
	rank = np.argsort(fisher)[::-1]
	return {"fisher_score": fisher, "fisher_rank": rank}


def calculate_reliefF_score(X: pd.DataFrame, y: pd.Series) -> dict:
	X_array, y_array = to_ndarray(X, y)
	relieff = reliefF.reliefF(X_array, y_array)
	rank = np.argsort(relieff)[::-1]
	return {"reliefF_score": relieff, "reliefF_rank": rank}


def calculate_spec_score(X: pd.DataFrame) -> dict:
	X_array = X.to_numpy()
	spec_score = SPEC.spec(X_array)
	rank = np.argsort(spec_score)[::-1]
	return {"spec_score": spec_score, "spec_rank": rank}


def calculate_f_score(X: pd.DataFrame, y: pd.Series) -> dict:
	X_array, y_array = to_ndarray(X, y)
	f = f_score.f_score(X_array, y_array)
	rank = np.argsort(f)[::-1]
	return {"f_score": f, "f_rank": rank}

def calculate_rfs_score(X: pd.DataFrame, y: pd.Series) -> dict:
	X_array = X.to_numpy()
	label_binarizer = LabelBinarizer()
	label_binarizer.fit(y)
	y_transformed = label_binarizer.transform(y)
	rfs_matrix = RFS.rfs(X_array, y_transformed)
	rfs_score = np.sum(rfs_matrix, axis=1)
	rank = np.argsort(rfs_score)[::-1]
	return {"rfs_score": rfs_score, "rfs_rank": rank}


def calculate_chi2_score(X: pd.DataFrame, y: pd.Series) -> dict:
	X_array, y_array = to_ndarray(X, y)
	if np.all(X_array > 0):
		chi2 = chi_square.chi_square(X_array, y_array)
	else:
		scaler = MinMaxScaler()
		X_transformed = scaler.fit_transform(X_array)
		chi2 = chi_square.chi_square(X_transformed, y_array)
	rank = np.argsort(chi2)[::-1]
	return {"chi2_score": chi2, "chi2_rank": rank}


def calculate_mrmr_score(X: pd.DataFrame, y: pd.Series, classification: bool = True) -> dict:
	X_array, y_array = to_ndarray(X, y)
	if classification:
		label_encoder = LabelEncoder()
		label_encoder.fit(y_array)
		y_array = label_encoder.transform(y_array)
	mrmr = MRMR.mrmr(X_array, y_array, n_selected_features=len(X.columns))
	return {"mrmr_score": mrmr[1], "mrmr_rank": mrmr[0]}


def calculate_cmim_score(X: pd.DataFrame, y: pd.Series, classification: bool = True) -> dict:
	X_array, y_array = to_ndarray(X, y)
	
	if classification:
		label_encoder = LabelEncoder()
		label_encoder.fit(y_array)
		y_array = label_encoder.transform(y_array)
	
	cmim = CMIM.cmim(X_array, y_array, n_selected_features=len(X.columns))
	return {"cmim_score": cmim[1], "cmim_rank": cmim[0]}


def calculate_jmi_score(X: pd.DataFrame, y: pd.Series, classification: bool = True) -> dict:
	X_array, y_array = to_ndarray(X, y)
	
	if classification:
		label_encoder = LabelEncoder()
		label_encoder.fit(y_array)
		y_array = label_encoder.transform(y_array)
	
	jmi = JMI.jmi(X_array, y_array, n_selected_features=len(X.columns))
	return {"jmi_score": jmi[1], "jmi_rank": jmi[0]}


def select_features_by_borutashap(X: pd.DataFrame, y: pd.Series, model_class: Type[BaseEstimator],
                                  classification: bool) -> np.ndarray:
	model = model_class()
	feature_selector = BorutaShap(model, importance_measure="shap", classification=classification)
	feature_selector.fit(X=X, y=y.to_numpy(), random_state=42, verbose=False)
	features_selected = list(feature_selector.accepted)
	features_selected_mask = np.isin(X.columns, features_selected)
	return features_selected_mask


def calculate_random_forest_score(X: pd.DataFrame, y: pd.Series, classification: bool = True) -> dict:
	X_array, y_array = to_ndarray(X, y)
	if classification:
		rfc = RandomForestClassifier()
		rfc.fit(X_array, y_array)
		rank = np.argsort(rfc.feature_importances_)[::-1]
		return {"random_forest_score": rfc.feature_importances_, "random_forest_rank": rank}
	else:
		rfr = RandomForestRegressor()
		rfr.fit(X_array, y_array)
		rank = np.argsort(rfr.feature_importances_)[::-1]
		return {"random_forest_score": rfr.feature_importances_, "random_forest_rank": rank}


def calculate_lasso_score(X: pd.DataFrame, y: pd.Series) -> dict:
	X_array, y_array = to_ndarray(X, y)
	lasso = Lasso()
	lasso.fit(X_array, y_array)
	rank = np.argsort(np.abs(lasso.coef_))[::-1]
	return {"lasso_score": np.abs(lasso.coef_), "lasso_rank": rank}

def select_features(X: pd.DataFrame, y: pd.Series, classification: bool = True,
                    methods_choosed: list = [], by: str = "model", *,
                    model_class: Type[BaseEstimator] = None, percentage: float = None,
                    return_scores: bool = False) -> dict:
	fs_methods = {
		"filter_methods": {
			"fisher": {"function": calculate_fisher_score, "args": [X, y]},
			"reliefF": {"function": calculate_reliefF_score, "args": [X, y]},
			"spec": {"function": calculate_spec_score, "args": [X]},
			"f": {"function": calculate_f_score, "args": [X, y]},
			"chi2": {"function": calculate_chi2_score, "args": [X, y]},
			"rfs": {"function": calculate_rfs_score, "args": [X, y]},
			"mrmr": {"function": calculate_mrmr_score, "args": [X, y, classification]},
			"cmim": {"function": calculate_cmim_score, "args": [X, y, classification]},
			"jmi": {"function": calculate_jmi_score, "args": [X, y, classification]}
		},
		"wrapper_methods": {
			"BorutaShap": {"function": select_features_by_borutashap, "args": [X, y, model_class, classification]}
		},
		"embedded_methods": {
			"random_forest": {"function": calculate_random_forest_score, "args": [X, y, classification]},
			"lasso": {"function": calculate_lasso_score, "args": [X, y]}
		}
	}
	
	scores = {}
	features_selected = {}
	for method_type in fs_methods.keys():
		for method in fs_methods[method_type].keys():
			if method in methods_choosed:
				method_function = fs_methods[method_type][method]["function"]
				if method_type in ["filter_methods", "embedded_methods"]:
					result_dict = method_function(*fs_methods[method_type][method]["args"])
					score = result_dict[method + "_score"]
					rank = result_dict[method + "_rank"]
					scores[method] = score
					if by == "model":
						features_selected_by_method = select_features_by_model(X, y, rank, model_class)
					elif by == "percentage":
						features_selected_by_method = select_features_by_percentage(X, rank, percentage)
				else:
					features_selected_by_method = method_function(*fs_methods[method_type][method]["args"])
				array_str = np.array2string(features_selected_by_method, max_line_width=np.inf,
				                            separator=",").replace(" ", "")
				if array_str not in features_selected.keys():
					features_selected[array_str] = {"mask": features_selected_by_method, "methods": {}}
				if method_type not in features_selected[array_str]["methods"].keys():
					features_selected[array_str]["methods"][method_type] = []
				features_selected[array_str]["methods"][method_type].append(method)
	if return_scores:
		return {"scores": scores, "features_selected": features_selected}
	else:
		return {"features_selected": features_selected}

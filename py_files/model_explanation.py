from typing import Union

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from tensorflow.keras import Model

from coalitional_explanation_methods.coalitional_methods import coalitional_method
from coalitional_explanation_methods.complete_method import complete_method


def explain_model_by_lime(X: pd.DataFrame, features_selected: np.ndarray, model: BaseEstimator,
                          mode: str = "classification",
                          discretize_continuous: bool = True, num_samples: int = 5000) -> shap.Explanation:
	assert mode in ["classification", "regression"]
	features = X.columns[features_selected]
	X_selected = X[features]
	X_array = X_selected.to_numpy()
	lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_array, feature_names=features.to_list(),
	                                                        mode=mode, discretize_continuous=discretize_continuous)
	explanations = []
	if mode == "classification":
		for i in range(X_selected.shape[0]):
			explanation = sorted(lime_explainer.explain_instance(X_selected.iloc[i].to_numpy(),
			                                                     predict_fn=model.predict_proba,
			                                                     num_features=len(features),
			                                                     num_samples=num_samples).as_map()[1])
			explanations.append(explanation)
	else:
		for i in range(X_selected.shape[0]):
			explanation = sorted(lime_explainer.explain_instance(X_selected.iloc[i].to_numpy(),
			                                                     predict_fn=model.predict,
			                                                     num_features=len(features),
			                                                     num_samples=num_samples).as_map()[1])
			explanation_values = []
			for j in range(len(explanation)):
				explanation_values.append(explanation[j][1])
			explanations.append(explanation_values)
	lime_explanation = shap.Explanation(np.array(explanations), base_values=0, data=X_array,
	                                    feature_names=features.to_list())
	return lime_explanation


def explain_model_by_shap(X: pd.DataFrame, features_selected: np.ndarray, model: Union[BaseEstimator, Model],
                          classification: bool,
                          shap_type: str,
                          num_background_samples = None,
                          # class_index: int = None,
                          # sample_index: int = None
                          ) -> shap.Explanation:
	features = X.columns[features_selected]
	X_selected = X[features]
	if shap_type == "TreeSHAP":
		explainer = shap.TreeExplainer(model)
	elif shap_type == "DeepSHAP":
		if num_background_samples is None:
			explainer = shap.DeepExplainer(model=model, data=X_selected)
		else:
			explainer = shap.DeepExplainer(model=model, data=shap.kmeans(X_selected, num_background_samples).data)
	elif shap_type == "KernelSHAP":
		if num_background_samples is None:
			explainer = shap.KernelExplainer(model=model.predict_proba if classification else model.predict, data=X_selected)
		else:
			explainer = shap.KernelExplainer(model=model.predict_proba if classification else model.predict, data=shap.kmeans(X_selected, num_background_samples).data)
	shap_values = explainer.shap_values(X_selected.values)
	
	if classification:
		shap_explanation = shap.Explanation(values=shap_values[1],
		                                    base_values=explainer.expected_value[1],
		                                    data=X_selected.to_numpy(), feature_names=X_selected.columns.tolist())
	else:
		if type(shap_values) == list:
			shap_values = shap_values[0]
		shap_explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value,
		                                    data=X_selected.to_numpy(),
		                                    feature_names=X_selected.columns.tolist())
	return shap_explanation


def explain_model_by_coalitional_method(X: pd.DataFrame, features_selected: np.ndarray, y: pd.Series,
                                                    model: Union[BaseEstimator, Model],
                                                    rate: float = 0.25, fvoid: float = None,
                                                    method: str = "spearman",
                                                    reverse: bool = False,
                                                    complexity: bool = False,
                                                    scaler: bool = False,
                                                    classification: bool = True):
	features = X.columns[features_selected]
	X_selected = X[features]
	problem_type = "Classification" if classification else "Regression"
	explanation_result = coalitional_method(X_selected, y, model, rate=rate, problem_type=problem_type, fvoid=fvoid,
	                                        method=method, reverse=reverse, complexity=complexity, scaler=scaler,
	                                        progression_bar=False)
	explanation_values = explanation_result[0].to_numpy()
	coalitional_explanation = shap.Explanation(values=explanation_values, base_values=0,
	                                           data=X_selected.to_numpy(),
	                                           feature_names=X_selected.columns)
	return coalitional_explanation


def explain_model_by_complete_method(X: pd.DataFrame, features_selected: np.ndarray, y: pd.Series,
                                     model: Union[BaseEstimator, Model],
                                     classification: bool = True, fvoid: float = None):
	features = X.columns[features_selected]
	X_selected = X[features]
	problem_type = "Classification" if classification else "Regression"
	explanation_result = complete_method(X_selected, y, model, problem_type, fvoid, progression_bar=False)
	explanation_values = explanation_result[0].to_numpy()
	complete_explanation = shap.Explanation(values=explanation_values, base_values=0, data=X_selected.to_numpy(),
	                                        feature_names=X_selected.columns)
	return complete_explanation

from typing import Union, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from tensorflow.keras import Model

from py_files.feature_selection import select_features
from py_files.fs_methods_comparation import compare_fs_methods_by_average_of_rank, \
	compare_fs_methods_by_average_of_scores, \
	compare_fs_methods_by_preference, compare_fs_methods_by_pareto
from py_files.metrics import calculate_retention_rate, calculate_rmse_score, calculate_model_performance_metric, \
	calculate_kendalltau, calculate_relative_influence_changes, calculate_RI, calculate_RIA, calculate_dcor
from py_files.model_explanation import explain_model_by_lime, explain_model_by_shap, \
	explain_model_by_coalitional_method, explain_model_by_complete_method
from py_files.model_training import train_one_model
from py_files.rs_utils import all_unique
from py_files.rs_utils import is_valid_list


def recommend_fs_methods_for_one_model(X: pd.DataFrame, y: pd.Series,
                                       model_class_for_training: Union[Type[BaseEstimator], Type[Model]],
                                       classification: bool = True, fs_methods_choosed: list = [],
                                       by: str = "model", return_scores: bool = False,
                                       user_defined_feature_subsets: list = [], explanation: str = "LIME",
                                       metrics_choosed: list = None,
                                       strategy_for_comparation: str = "pareto", return_df: bool = False, *,
                                       model_class_for_fs: Type[BaseEstimator] = None,
                                       percentage: float = None,
                                       preference: list = None, model_kwargs: dict = {},
                                       tf_model_init_params: dict = {},
                                       tf_model_compile_params: dict = {},
                                       tf_model_fit_params: dict = {},
                                       lime_params: dict = {},
                                       shap_params: dict = {},
                                       coalitional_method_params: dict = {},
                                       complete_method_params: dict = {},
                                       ) -> pd.DataFrame:
	if user_defined_feature_subsets != []:
		assert all(len(arr) == X.shape[1] for arr in
		           user_defined_feature_subsets), "All user-defined feature subsets must have the same length as X.columns."
	assert all_unique(user_defined_feature_subsets), "User can't difine same feature_subsets."
	assert by in ["model", "percentage"], "The parameter \"by\" must be \"model\" or \"percentage\"."
	if model_class_for_fs is None:
		raise Exception("The parameter \"model_class\" can not be None.")
	if by == "percentage" and (percentage is None or not (0 <= percentage <= 100)):
		raise Exception("The parameter \"percentage\" can not be None, and must be between 0 and 100.")

	assert all(
		metric in ["retention_rate", "accuracy", "mae", "rmse", "r2", "kendalltau", "relative_influence_changes", "RI",
		           "RIA", "dcor"]
		for metric in
		metrics_choosed), "The metrics must be selected from retention_rate, accuracy, mae, rmse, r2, kendalltau, relative_influence_changes, RI, RIA and dcor."
	if classification and any(metric in metrics_choosed for metric in
	                          ["mae", "rmse",
	                           "r2"]):
		raise Exception("The metrics mae, rmse, r2 can't be used for a classification problem.")
	if not classification and "accuracy" in metrics_choosed:
		raise Exception("The metric accuracy can't be used for a regression problem.")
	assert is_valid_list(metrics_choosed), "accuracy or r2 must be before RIA."
	assert strategy_for_comparation in ["average_by_rank", "average_by_scores", "preference",
	                                    "pareto"], "The strategy for comparation must be selected from average_by_rank, average_by_scores, preference and pareto."
	if strategy_for_comparation == "average_by_scores" and any(metric in metrics_choosed for metric in
	                                                           ["relative_influence_changes", "RI",
	                                                            "RIA"]):
		raise Exception(
			"relative_influence_changes, RI and RIA can't be used if strategy_for_comparation is average_by_scores.")
	if strategy_for_comparation == "preference" and preference is None:
		raise Exception(
			"The parameter \"preference\" can not be None.")
	
	fs_results = select_features(X, y, classification, fs_methods_choosed, by, model_class=model_class_for_fs,
	                             percentage=percentage, return_scores=return_scores)
	
	features_selected = fs_results["features_selected"]
	
	for i in range(len(user_defined_feature_subsets)):
		if np.array2string(user_defined_feature_subsets[i], max_line_width=np.inf,
		                   separator=",").replace(" ", "") in features_selected.keys():
			if "user_defined" not in features_selected[
				np.array2string(user_defined_feature_subsets[i], max_line_width=np.inf, separator=",").replace(" ", "")][
				"methods"].keys():
				features_selected[
					np.array2string(user_defined_feature_subsets[i], max_line_width=np.inf, separator=",").replace(" ", "")]["methods"][
					"user_defined"] = []
			features_selected[np.array2string(user_defined_feature_subsets[i], max_line_width=np.inf, separator=",").replace(" ", "")][
				"methods"]["user_defined"].append("user_defined_" + str(i + 1))
		else:
			features_selected[np.array2string(user_defined_feature_subsets[i], max_line_width=np.inf,
			                                  separator=",").replace(" ", "")] = {"mask": user_defined_feature_subsets[i],
			                                                     "methods": {
				                                                     "user_defined": ["user_defined_" + str(i + 1)]}}
	
	print(
		"Feature selection finished, the number of feature subsets (with the user defined feature subsets): " + str(
			len(features_selected)))
	contains_all_features = np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
	                                        separator=",").replace(" ", "") in features_selected.keys()
	if any(metric in metrics_choosed for metric in
	       ["kendalltau", "relative_influence_changes", "RI", "RIA"]) and not contains_all_features:
		features_selected[np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf, separator=",").replace(" ", "")] = {
			"mask": np.ones(X.shape[1], dtype=bool), "methods": {"all_features": "all_features"}}
	model_training_results = {}
	for features in features_selected.keys():
		model_training_result = train_one_model(X, y, features_selected[features]["mask"], model_class_for_training,
		                                        tf_model_init_params=tf_model_init_params,
		                                        tf_model_compile_params=tf_model_compile_params,
		                                        tf_model_fit_params=tf_model_fit_params,
		                                        model_kwargs=model_kwargs)
		model_training_results[features] = model_training_result
	print("Model training finished.")
	explanation_results = {}
	for features in model_training_results.keys():
		if explanation == "LIME":
			explanation_result = explain_model_by_lime(X, features_selected[features]["mask"],
			                                           model_training_results[features]["model_trained"],
			                                           mode="classification" if classification else "regression", **lime_params)
		elif "SHAP" in explanation:
			explanation_result = explain_model_by_shap(X, features_selected[features]["mask"],
			                                           model_training_results[features]["model_trained"],
			                                           classification, shap_type=explanation, **shap_params)
		elif explanation == "Coalitional Method":
			explanation_result = explain_model_by_coalitional_method(X, features_selected[features]["mask"],
			                                                         y, model_training_results[features][
				                                                         "model_trained"],
			                                                         classification=classification,
			                                                         **coalitional_method_params)
		else:
			explanation_result = explain_model_by_complete_method(X, features_selected[features]["mask"], y,
			                                                      model_training_results[features][
				                                                      "model_trained"], classification=classification,
			                                                      **complete_method_params)

		if not isinstance(model_class_for_training, BaseEstimator):
			model_training_results[features]["model_trained"] = {"tf_model_init_params": tf_model_init_params,
																 "tf_model_compile_params": tf_model_compile_params,
																 "tf_model_fit_params": tf_model_fit_params}

		explanation_results[features] = explanation_result
	print("Model explanation finished.")
	metrics = {"retention_rate": {"retention_rate": calculate_retention_rate},
	           "model_performance_metrics": {"accuracy": accuracy_score,
	                                         "mae": mean_absolute_error,
	                                         "rmse": calculate_rmse_score,
	                                         "r2": r2_score
	                                         },
	           "explanation_based_metrics": {"similarity_based": {"kendalltau": calculate_kendalltau,
	                                                              "relative_influence_changes": calculate_relative_influence_changes,
	                                                              "RI": calculate_RI,
	                                                              "RIA": calculate_RIA
	                                                              },
	                                         "discernibility": {"dcor": calculate_dcor}
	                                         }
	           }
	metrics_results = {}
	for metric in metrics_choosed:
		metrics_results[metric] = {}
		for features in model_training_results.keys():
			if not contains_all_features and features == np.array2string(np.ones(X.shape[1], dtype=bool),
			                                                             max_line_width=np.inf, separator=",").replace(" ", ""):
				continue
			if metric in metrics["retention_rate"].keys():
				metric_function = metrics["retention_rate"][metric]
				metric_score = metric_function(features_selected[features]["mask"])
			elif metric in metrics["model_performance_metrics"].keys():
				metric_function = metrics["model_performance_metrics"][metric]
				metric_score = calculate_model_performance_metric(metric_function, y,
				                                                  model_training_results[features]["y_pred"])
			else:
				if metric in metrics["explanation_based_metrics"]["similarity_based"].keys():
					metric_function = metrics["explanation_based_metrics"]["similarity_based"][metric]
					if metric != "RIA":
						metric_score = metric_function(
							explanation_results[np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
							                                    separator=",").replace(" ", "")],
							explanation_results[features], features_selected[features]["mask"])
					else:
						accuracy_needed = "accuracy" if classification else "r2"
						if np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
						                   separator=",").replace(" ", "") not in metrics_results[
							accuracy_needed].keys():
							accuracy_all_features = calculate_model_performance_metric(
								metrics["model_performance_metrics"][accuracy_needed], y,
								model_training_results[
									np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
									                separator=",").replace(" ", "")]["y_pred"])
							metrics_results[accuracy_needed][
								np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
								                separator=",").replace(" ", "")] = accuracy_all_features
						metric_score = metric_function(
							explanation_results[np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
							                                    separator=",").replace(" ", "")],
							explanation_results[features], features_selected[features]["mask"],
							metrics_results[accuracy_needed][
								np.array2string(np.ones(X.shape[1], dtype=bool), max_line_width=np.inf,
								                separator=",").replace(" ", "")],
							metrics_results[accuracy_needed][features])
				else:
					metric_function = metrics["explanation_based_metrics"]["discernibility"][metric]
					metric_score = metric_function(explanation_results[features])
			metrics_results[metric][features] = metric_score
	print("Calculation of metrics finished.")
	
	if strategy_for_comparation == "average_by_rank":
		comparation_results = compare_fs_methods_by_average_of_rank(metrics_results, return_df)
	elif strategy_for_comparation == "average_by_scores":
		comparation_results = compare_fs_methods_by_average_of_scores(metrics_results, return_df)
	elif strategy_for_comparation == "preference":
		comparation_results = compare_fs_methods_by_preference(metrics_results, preference, return_df)
	else:
		comparation_results = compare_fs_methods_by_pareto(metrics_results, return_df)
	print("Comparation finished.")
	pd.set_option('display.max_columns', 100)
	pd.set_option('display.max_rows', 100)
	pd.set_option('display.max_colwidth', 300)
	if return_scores:
		print("Feature selection scores: ")
	df_result = pd.DataFrame(columns=["mask", "FS methods", "subset of features"])
	for features in comparation_results["rank"]:
		df_result.loc[len(df_result)] = [features_selected[features]["mask"],
		                                 str(features_selected[features]["methods"]),
		                                 str([feature for feature in X.columns[features_selected[features]["mask"]]])]
	
	all_results = {"fs_results": fs_results, "model_training_results": model_training_results,
	               "explanation_results": explanation_results, "metrics_results": metrics_results,
	               "comparation_results": comparation_results}
	
	return all_results

from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd
import shap
from dcor import distance_correlation
from dcor._dcor_internals import CompileMode
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error


def calculate_retention_rate(features_selected: np.ndarray) -> float:
	retention_rate = sum(features_selected) / len(features_selected)
	return retention_rate



def calculate_rmse_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
	"""
	Calculate the Root Mean Square Error (RMSE) score of the prediction.

	This function takes the true labels and the predicted labels as input,
	and returns the RMSE score of the prediction. The RMSE score is
	calculated as the square root of the average of the squared differences
	between the true and predicted labels.

	Parameters
	----------
	y_true : pd.Series
		The true labels for the data.
	y_pred : np.ndarray
		The predicted labels for the data.

	Returns
	-------
	float
		The RMSE score of the prediction.
	"""
	
	mse = mean_squared_error(y_true, y_pred)
	rmse = sqrt(mse)
	return rmse

def calculate_model_performance_metric(metric_function: Callable, y_true: pd.Series, y_pred: np.ndarray):
	metric_score = metric_function(y_true.to_numpy(), y_pred)
	return metric_score

def calculate_kendalltau(all_features_explanation: shap.Explanation, selected_features_explanation: shap.Explanation,
                          selected_features: np.ndarray) -> float:
	"""
	Calculate the Kendall's Tau correlation coefficient between the order of importance of all features and selected features.

	This function takes the explanations of all features and selected features as input,
	and returns the Kendall's Tau correlation coefficient. The coefficient is calculated based on the order of importance
	of the features in the all features explanation and the selected features explanation.

	Parameters
	----------
	all_features_explanation : shap.Explanation
		The SHAP explanation for all features.
	selected_features_explanation : shap.Explanation
		The SHAP explanation for the selected features.
	selected_features : np.ndarray
		The indices of the selected features.

	Returns
	-------
	float
		The Kendall's Tau correlation coefficient.
	"""
	
	all_features_explanation_values = all_features_explanation.values
	selected_features_explanation_values = selected_features_explanation.values
	avg_all_features_explanation = np.average(abs(all_features_explanation_values), axis=0)
	avg_selected_features_explanation = np.average(abs(selected_features_explanation_values), axis=0)
	avg_common_features_explanation = avg_all_features_explanation[selected_features]
	order_all_features = np.argsort(avg_common_features_explanation)[::-1]
	order_selected_features = np.argsort(avg_selected_features_explanation)[::-1]
	tau, _ = kendalltau(order_all_features, order_selected_features)
	return tau


def calculate_relative_influence_changes(all_features_explanation: shap.Explanation,
                                         selected_features_explanation: shap.Explanation,
                                         selected_features: np.ndarray) -> float:
	"""
	Calculate the relative changes in influence of selected features.

	This function takes the explanations of all features and selected features as input,
	and returns the sum of absolute differences between the average relative influences of all features and selected features.
	The relative influence of a feature is calculated as the feature's absolute explanation value divided by the sum of absolute explanation values of all features.

	Parameters
	----------
	all_features_explanation : shap.Explanation
		The SHAP explanation for all features.
	selected_features_explanation : shap.Explanation
		The SHAP explanation for the selected features.
	selected_features : np.ndarray
		The indices of the selected features.

	Returns
	-------
	float
		The sum of absolute differences between the average relative influences of all features and selected features.
	"""
	
	all_features_explanation_values = all_features_explanation.values
	selected_features_explanation_values = selected_features_explanation.values
	for i, m in enumerate(selected_features):
		if not m:
			selected_features_explanation_values = np.insert(selected_features_explanation_values, i,
			                                                 np.zeros(len(all_features_explanation.data)), axis=1)
	sum_all_features = np.sum(abs(all_features_explanation_values), axis=1)[:, None]
	r_avg_all_features = np.average(np.divide(all_features_explanation_values, sum_all_features,
	                                          out=np.full_like(all_features_explanation_values, np.nan),
	                                          where=sum_all_features != 0), axis=0)
	sum_selected_features = np.sum(abs(selected_features_explanation_values), axis=1)[:, None]
	r_avg_selected_features = np.average(np.divide(selected_features_explanation_values, sum_selected_features,
	                                               out=np.full_like(selected_features_explanation_values, np.nan),
	                                               where=sum_selected_features != 0), axis=0)
	
	return np.sum(abs(r_avg_all_features - r_avg_selected_features))


def calculate_differences(all_features_explanation: shap.Explanation, selected_features_explanation: shap.Explanation,
                          selected_features: np.ndarray) -> tuple:
	"""
	Calculate the differences in influence, relative influence, and rank of selected features.

	This function takes the explanations of all features and selected features as input,
	and returns a tuple containing three lists: the differences in influence, the differences in relative influence,
	and the differences in rank of the selected features. The influence of a feature is its absolute explanation value,
	the relative influence is the feature's influence divided by the sum of influences of all features, and the rank is
	the order of the feature when sorted by influence.

	Parameters
	----------
	all_features_explanation : shap.Explanation
		The SHAP explanation for all features.
	selected_features_explanation : shap.Explanation
		The SHAP explanation for the selected features.
	selected_features : np.ndarray
		The indices of the selected features.

	Returns
	-------
	tuple
		A tuple containing three lists: the differences in influence, the differences in relative influence,
		and the differences in rank of the selected features.
	"""
	
	all_features_explanation_values = all_features_explanation.values
	selected_features_explanation_values = selected_features_explanation.values
	for i, m in enumerate(selected_features):
		if not m:
			selected_features_explanation_values = np.insert(selected_features_explanation_values, i,
			                                                 np.zeros(len(all_features_explanation.data)), axis=1)
	sum_all_features = np.sum(abs(all_features_explanation_values), axis=1)[:, None]
	r_avg_all_features = np.divide(all_features_explanation_values, sum_all_features,
	                               out=np.full_like(all_features_explanation_values, np.nan),
	                               where=sum_all_features != 0)
	sum_selected_features = np.sum(abs(selected_features_explanation_values), axis=1)[:, None]
	r_avg_selected_features = np.divide(selected_features_explanation_values, sum_selected_features,
	                                    out=np.full_like(selected_features_explanation_values, np.nan),
	                                    where=sum_selected_features != 0)
	avg_all_features = np.average(abs(all_features_explanation_values), axis=0)
	avg_selected_features = np.average(abs(selected_features_explanation_values), axis=0)
	
	order_all_features = np.argsort(avg_all_features)[::-1]
	order_selected_features = np.argsort(avg_selected_features)[::-1]
	
	for i, m in enumerate(selected_features):
		if not m:
			order_selected_features = np.delete(order_selected_features, np.argwhere(order_selected_features == i))
			order_selected_features = np.append(order_selected_features, i)
	
	diff_inf = [np.average(abs(all_features_explanation_values - selected_features_explanation_values),
	                       axis=0)[i] for i, m in enumerate(selected_features) if m]
	diff_rinf = [np.average(abs(r_avg_all_features - r_avg_selected_features), axis=0)[i] for i, m in
	             enumerate(selected_features) if m]
	diff_rank = [
		abs(order_all_features.tolist().index(i) - order_selected_features.tolist().index(i)) / len(selected_features)
		for i, m in enumerate(selected_features) if m]
	return (diff_inf, diff_rinf, diff_rank)


def calculate_RI(all_features_explanation: shap.Explanation, selected_features_explanation: shap.Explanation,
                 selected_features: np.ndarray) -> float:
	"""
	Calculate the Relative Importance (RI) of selected features.

	This function takes the explanations of all features and selected features as input,
	and returns the RI of the selected features. The RI is calculated as the average of the product of the differences
	in relative influence and rank of the selected features, subtracted by the square of the machine epsilon.

	The differences in relative influence and rank are calculated using the `calculate_differences` function.

	Parameters
	----------
	all_features_explanation : shap.Explanation
		The SHAP explanation for all features.
	selected_features_explanation : shap.Explanation
		The SHAP explanation for the selected features.
	selected_features : np.ndarray
		The indices of the selected features.

	Returns
	-------
	float
		The RI of the selected features.
	"""
	
	esp = np.finfo(float).eps
	_, diff_rinf, diff_rank = calculate_differences(all_features_explanation, selected_features_explanation,
	                                                selected_features)
	
	diff_rinf = np.add(np.power(diff_rinf, 1 / 4), esp)
	diff_rank = np.add(diff_rank, esp)
	return np.average(np.multiply(diff_rinf, diff_rank) - esp ** 2)


def calculate_RIA(all_features_explanation: shap.Explanation, selected_features_explanation: shap.Explanation,
                  selected_features: np.ndarray, all_accuracy: float, selected_accuracy) -> float:
	"""
	Calculate the Relative Importance Accuracy (RIA) of selected features.

	This function takes the explanations of all features and selected features, the accuracy of all features and selected features as input,
	and returns the RIA of the selected features. The RIA is calculated as the average of the product of the differences
	in relative influence and rank of the selected features and the difference in accuracy, subtracted by the cube of the machine epsilon.

	The differences in relative influence and rank are calculated using the `calculate_differences` function.

	Parameters
	----------
	all_features_explanation : shap.Explanation
		The SHAP explanation for all features.
	selected_features_explanation : shap.Explanation
		The SHAP explanation for the selected features.
	selected_features : np.ndarray
		The indices of the selected features.
	all_accuracy : float
		The accuracy of the model with all features.
	selected_accuracy : float
		The accuracy of the model with selected features.

	Returns
	-------
	float
		The RIA of the selected features.
	"""
	
	esp = np.finfo(float).eps
	_, diff_rinf, diff_rank = calculate_differences(all_features_explanation, selected_features_explanation,
	                                                selected_features)
	diff_rinf = np.add(np.power(diff_rinf, 1 / 4), esp)
	diff_rank = np.add(diff_rank, esp)
	
	diff_acc = selected_accuracy - all_accuracy + esp
	return np.average(np.multiply(diff_rinf, diff_rank) * diff_acc - esp ** 3)


def calculate_dcor(selected_features_explanation: shap.Explanation) -> float:
	"""
	Calculate the distance correlation (dCor) for the selected features.

	This function takes the SHAP explanation for the selected features as input,
	and returns the weighted sum of the distance correlations of each feature.
	The distance correlation measures the statistical dependence between the feature values and the explanation values.
	The weight of each feature is its average absolute explanation value divided by the sum of average absolute explanation values of all features.

	Parameters
	----------
	selected_features_explanation : shap.Explanation
		The SHAP explanation for the selected features.

	Returns
	-------
	float
		The weighted sum of the distance correlations of the selected features.
	"""
	
	data = selected_features_explanation.data.astype(np.float32)
	explanation_values = selected_features_explanation.values.astype(np.float32)
	num_features = data.shape[1]
	dcor_feature = []
	for i in range(num_features):
		dcor_feature.append(
			distance_correlation(data[:, i], explanation_values[:, i], compile_mode=CompileMode.NO_COMPILE))
	mean_explanation_values = np.mean(np.abs(explanation_values), axis=0)
	weight_feature = []
	for i in range(len(mean_explanation_values)):
		weight_feature.append(mean_explanation_values[i] / sum(mean_explanation_values))
	dcor_weighted_feature = np.array(dcor_feature) * np.array(weight_feature)
	dcor_subset = sum(dcor_weighted_feature)
	return dcor_subset

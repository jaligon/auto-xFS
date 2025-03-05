import ast
from typing import Union

import numpy as np
import pandas as pd


def compare_fs_methods_by_average_of_rank(metric_scores: Union[dict, pd.DataFrame], return_df: bool) -> dict:
	df_metric_scores = pd.DataFrame(metric_scores) if isinstance(metric_scores, dict) else metric_scores
	metrics = df_metric_scores.columns.tolist()
	for metric in metrics:
		df_metric_scores_temp = df_metric_scores.copy().sort_values(by=metric,
		                                                            ascending=metric in ["relative_influence_changes",
		                                                                                 "RI", "retention_rate"])
		df_metric_scores_temp["rank_" + metric] = [i + 1 for i in range(df_metric_scores_temp.shape[0])]
		df_metric_scores["rank_" + metric] = None
		for index in df_metric_scores.index:
			df_metric_scores.loc[index, ("rank_" + metric)] = df_metric_scores_temp.loc[index, ("rank_" + metric)]
	rank_columns = [col for col in df_metric_scores.columns if 'rank' in col]
	df_metric_scores["average rank"] = df_metric_scores[rank_columns].mean(axis=1)
	df_metric_scores_sorted = df_metric_scores.sort_values(by="average rank")
	if return_df:
		return {"rank": df_metric_scores_sorted.index.tolist(), "df_sorted": df_metric_scores_sorted}
	else:
		return {"rank": df_metric_scores_sorted.index.tolist()}


def compare_fs_methods_by_average_of_scores(metric_scores: Union[dict, pd.DataFrame], return_df: bool) -> dict:
	df_metric_scores = pd.DataFrame(metric_scores) if isinstance(metric_scores, dict) else metric_scores
	if "kendalltau" in df_metric_scores.columns:
		df_metric_scores["kendalltau_normalized"] = (df_metric_scores["kendalltau"] + 1) / 2
	cols = [col for col in df_metric_scores.columns if col != "kendalltau"]
	df_metric_scores["average score"] = df_metric_scores[cols].mean(axis=1)
	df_metric_scores_sorted = df_metric_scores.sort_values(by="average score", ascending=False)
	if return_df:
		return {"rank": df_metric_scores_sorted.index.tolist(), "df_sorted": df_metric_scores_sorted}
	else:
		return {"rank": df_metric_scores_sorted.index.tolist()}


def compare_fs_methods_by_preference(metric_scores: Union[dict, pd.DataFrame], preference: list, return_df: bool) -> dict:
	df_metric_scores = pd.DataFrame(metric_scores) if isinstance(metric_scores, dict) else metric_scores
	df_metric_scores_sorted = df_metric_scores.sort_values(by=preference, ascending=np.isin(preference, [
		"relative_influence_changes", "RI", "retention_rate"]).tolist())
	if return_df:
		return {"rank": df_metric_scores_sorted.index.tolist(), "df_sorted": df_metric_scores_sorted}
	else:
		return {"rank": df_metric_scores_sorted.index.tolist()}


def is_pareto_efficient(metric_scores: np.ndarray):
	"""
	Determine the Pareto efficiency of each point in a set of multi-dimensional points.

	This function takes a 2D numpy array of metric scores as input, where each row represents a point and each column represents a dimension,
	and returns a boolean numpy array indicating whether each point is Pareto efficient.

	A point is considered Pareto efficient if there is no other point that is better in all dimensions.
	"Better" is defined as greater for this function, so it assumes that a higher score is better for all metrics.

	The function first initializes a boolean numpy array `is_efficient` with the same length as the number of points, and sets all elements to True.
	Then, for each point, if it is still considered efficient, the function compares it with all other points that are also considered efficient.
	If there is any other point that is better in all dimensions, the current point is no longer considered efficient.

	Parameters
	----------
	metric_scores : np.ndarray
		A 2D numpy array of metric scores, where each row represents a point and each column represents a dimension.

	Returns
	-------
	np.ndarray
		A boolean numpy array indicating whether each point is Pareto efficient.
	"""
	
	is_efficient = np.ones(metric_scores.shape[0], dtype=bool)
	for i, c in enumerate(metric_scores):
		if is_efficient[i]:
			# Keep this point if there is no point that is better on all dimensions
			is_efficient[is_efficient] = np.any(metric_scores[is_efficient] > c, axis=1)
			is_efficient[i] = True  # Ensure the current point is not mistakenly removed
			# Use `>` instead of `>=` (v1) to ensure strict dominance, rather than allowing equal values.
			# Otherwise, points with equal values in some dimensions may incorrectly be considered Pareto-optimal
			# eg. ([1, 1], [2, 3], [3, 3]) => both [2, 3] [3, 3] would in pareto if no strict dominance (case of >=)
	return is_efficient


def compare_fs_methods_by_pareto(metric_scores: Union[dict, pd.DataFrame], return_pareto_df: bool) -> dict:
	df_metric_scores = pd.DataFrame(metric_scores)
	masks_index = df_metric_scores.index
	return_dic = {}

	explanation_similarity = ["kendalltau", "relative_influence_changes", "RI", "RIA"]
	inverse_metrics = ["relative_influence_changes", 'RI', 'retention_rate']
	pareto_dic={k: 0 for k in masks_index} # 0: non-pareto; 1: pareto; -1: fullset

	fullset_bool = masks_index.map(lambda x: all(ast.literal_eval(x)))
	for m in inverse_metrics:
		if m in df_metric_scores.columns:
			df_metric_scores[m] = -df_metric_scores[m]

	if sum(fullset_bool) > 0 and any(col in explanation_similarity for col in df_metric_scores.columns):
		pareto_dic[df_metric_scores.index[fullset_bool][0]] = -1
		df_metric_scores = df_metric_scores[~fullset_bool]


	df_metric_scores = df_metric_scores.loc[:, df_metric_scores.nunique() > 1]
	pareto_mask_wo_fullset = is_pareto_efficient(df_metric_scores.to_numpy())

	index_pareto = df_metric_scores[pareto_mask_wo_fullset].index
	pareto_dic.update({k: 1 for k in index_pareto})

	df_pareto = pd.DataFrame(metric_scores) if isinstance(metric_scores, dict) else metric_scores
	df_pareto = df_pareto.loc[index_pareto, :]
	return_dic['rank'] = df_pareto.index.tolist() # maybe unnecessary
	return_dic['pareto_info'] = pareto_dic
	if return_pareto_df:
		return_dic['df_sorted']=df_pareto
	return return_dic

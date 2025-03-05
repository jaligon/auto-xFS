def is_valid_list(metrics_choosed: list) -> bool:
	if "RIA" in metrics_choosed:
		if "accuracy" in metrics_choosed:
			return metrics_choosed.index("accuracy") < metrics_choosed.index("RIA")
		if "r2" in metrics_choosed:
			return metrics_choosed.index("r2") < metrics_choosed.index("RIA")
		raise Exception("accuracy or r2 must be selected if RIA is selected")
	return True

def all_unique(user_defined_feature_subsets: list) -> bool:
	tuple_list = [tuple(arr) for arr in user_defined_feature_subsets]
	return len(tuple_list) == len(set(tuple_list))

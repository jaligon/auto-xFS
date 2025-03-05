import base64
import csv
import datetime
import inspect
import io
import pickle
import re
import sys
import uuid
from distutils.util import strtobool

import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import redis
import shap
import tensorflow as tf
from dash import Dash, html, dcc, Output, Input, State, ALL
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sqlalchemy import create_engine, text
from sqlalchemy import inspect as s_inspect
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier, XGBRegressor

from py_files.fs_methods_comparation import compare_fs_methods_by_pareto, compare_fs_methods_by_average_of_rank, \
	compare_fs_methods_by_average_of_scores, compare_fs_methods_by_preference

tf.compat.v1.disable_v2_behavior()
# sys.path.append("./py_files")
sys.path.append("./coalitional_explanation_methods")
from py_files.recommendation_system import recommend_fs_methods_for_one_model

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
		   suppress_callback_exceptions=True)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_colwidth', 300)

pool = redis.ConnectionPool(host='localhost', port=6379, db=0, password='', max_connections=20)

DATABASE_URL = "sqlite:///web_app/recommendation_system.db"

engine = create_engine(
	DATABASE_URL,
	pool_size=20,
	max_overflow=10,
	pool_timeout=30,
	pool_recycle=1800
)

Session = sessionmaker(autoflush=False, bind=engine)

models = {
	"sklearn": {
		"Decision Tree": {"classification": DecisionTreeClassifier, "regression": DecisionTreeRegressor},
		"Random Forest": {"classification": RandomForestClassifier, "regression": RandomForestRegressor},
		"SVM": {"classification": SVC, "regression": SVR},
		"KNN": {"classification": KNeighborsClassifier, "regression": KNeighborsRegressor},
	},
	"xgboost": {
		"xgboost": {"classification": XGBClassifier, "regression": XGBRegressor},
	},
	"tensorflow": {
		"Tensorflow Sequential": Sequential
	}
}

layers = {
	"Dense": Dense
}
fs_methods = {"fisher": {"regression": False},
			  "reliefF": {"regression": True},
			  "spec": {"regression": True},
			  "f": {"regression": True},
			  "chi2": {"regression": False},
			  "rfs": {"regression": False},
			  "mrmr": {"regression": True},
			  "cmim": {"regression": True},
			  "jmi": {"regression": True},
			  "BorutaShap": {"regression": True},
			  "random forest": {"regression": True},
			  "lasso": {"regression": True}}

tree_based_models = (DecisionTreeClassifier,
                     DecisionTreeRegressor,
                     RandomForestClassifier,
                     RandomForestRegressor,
                     XGBClassifier,
                     XGBRegressor)

metrics_map = {
	"retention rate": ["retention rate"],
	"accuracy": ["accuracy", "r2"],
	"mean absolute error": ["mae"],
	"root mean square error": ["rmse"],
	"explanation similarity": ["kendalltau", "relative influence changes", "RI", "RIA"],
	"explanation discernibility": ["dcor"]
}

round_digits = 4
consent_text = "By choosing this option, you consent to enabling meta-learning for hyperparameter recommendations. " \
			   "Metadata from your dataset will be used to generate recommendations based on past user experiences, " \
			   "and your selection may contribute to improving suggestions for others. Only metadata will be stored " \
			   "on our server—your raw data and results will not be retained by this application."

def is_numeric(s):
	try:
		pd.to_numeric(s)
		return True
	except ValueError:
		return False

def is_continuous_numeric(series):
	if pd.api.types.is_numeric_dtype(series):
		return np.array_equal(series.sort_values().values, np.arange(series.min(), series.max() + 1))
	return False

def file_to_df(contents, filename):
# csv preprocessing
	preprocess = []
	content_type, content_string = contents.split(',')

	decoded = base64.b64decode(content_string).decode('utf-8')
	if '.csv' in filename:
		sample = decoded[:1024]
		dialect = csv.Sniffer().sniff(sample)
		has_header = csv.Sniffer().has_header(sample)
		df = pd.read_csv(io.StringIO(decoded), delimiter=dialect.delimiter,
						 decimal=',' if dialect.delimiter == ';' else '.',
						 header=0 if has_header else None)
	else:
		raise Exception("only csv file can be processed")

	if df.isna().sum().sum() > 0:
		df = df.dropna()
		preprocess.append("Remove missing values")

	y = df[df.columns[-1]]
	X = df.iloc[:, 0:len(df.columns) - 1]
	X = X.loc[:, ~X.columns.to_series().apply(lambda col: is_continuous_numeric(X[col]))]
	if X.shape[1] < df.shape[1] - 1:
		preprocess.append('drop index')

	preprocessor = ColumnTransformer(
		transformers=[
			("cat", Pipeline([
				# ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
				# ("ordinal", OrdinalEncoder()),
				("One-hot encoding", OneHotEncoder(sparse=False, drop='if_binary')),
			]), X.select_dtypes(include=["object"]).columns),
		],
		remainder="passthrough"
	)

	X = pd.DataFrame(preprocessor.fit_transform(X), columns=[name.split("_", 1)[-1][1:]
															 for name in preprocessor.get_feature_names_out()])
	for trans in preprocessor.transformers:
		if len(trans[2]) > 0:
			preprocess.append(trans[1].steps[0][0])

	problem_type = determine_problem_type(y)
	if problem_type == 'classification' and y.dtype == 'object':
		encoder = LabelEncoder()
		y = pd.Series(encoder.fit_transform(y))
		preprocess.append('Target label encoding')
	return X, y, preprocess, problem_type


def create_graph(X: pd.DataFrame, fs_results: dict, metrics_results: dict, metrics_choosed: list, strategy: str,
                 preference: list = None):
	metrics_choosed = [m.replace(" ", "_") for m in metrics_choosed]
	metrics_df = pd.DataFrame(metrics_results).dropna().round(round_digits)
	# metrcis for fullset as reference were not calculated, N/A in its place

	if strategy == "pareto":
		comparation_result = compare_fs_methods_by_pareto(metrics_df, False)
		global_pareto_map = {1: 'Global pareto of all calculated metrics',
							 0: 'Non-Pareto Solution',
							-1: 'Pareto is N/A for the fullset when using similarity-based metrics'}
		metrics_df["pareto_solution"] = metrics_df.index.map(comparation_result['pareto_info']).map(global_pareto_map)
		metrics_df["subset"] = ""
		metrics_df["fs methods"] = ""
		for features in metrics_df.index.tolist():
			mask = fs_results["features_selected"][features]["mask"]
			subset = []
			for i in range(len(mask)):
				if mask[i]:
					subset.append(i)
			subset_str = str(subset)
			j = 25
			while j < len(subset_str):
				if subset_str[j] == ",":
					subset_str = subset_str[0: j + 1] + "<br>" + subset_str[j + 1:]
				else:
					while j < len(subset_str) and subset_str[j] != ",":
						j += 1
					subset_str = subset_str[0: j + 1] + "<br>" + subset_str[j + 1:]
				j += 25
			metrics_df.loc[features, "subset"] = subset_str
			fs_methods_list = []
			for method_type in fs_results["features_selected"][features]["methods"].keys():
				fs_methods_list += fs_results["features_selected"][features]["methods"][method_type]
			metrics_df.loc[features, "fs methods"] = str(fs_methods_list).replace(
				"[", "").replace("]", "").replace("'", "").replace("_", " ")

		current_pareto_info = compare_fs_methods_by_pareto(metrics_df[metrics_choosed], False)["pareto_info"]
		local_pareto_map = {1: 'local pareto for the currently selected axes',
							0: 'not a Pareto for the current axes',
							-1: 'also N/A with selected metrics'}  # if current Pareto N/A due to uncomputable similarity-based metrics, global N/A also
		metrics_df["current_pareto"] = metrics_df.index.map(current_pareto_info).map(local_pareto_map)
		symbol_map = {local_pareto_map[1]: "diamond-open",
					  local_pareto_map[0]: "circle",
					  local_pareto_map[-1]: "x"}

		metrics_df = metrics_df.reset_index().rename(columns={"index": "mask", "dcor": "discernibility"})
		metrics_choosed = [i if i != 'dcor' else 'discernibility' for i in metrics_choosed]
		if len(metrics_choosed) == 3:
			graph = px.scatter_3d(metrics_df, x=metrics_choosed[0], y=metrics_choosed[1],
								  z=metrics_choosed[2], color='pareto_solution',
								  symbol="current_pareto", symbol_map=symbol_map,
								  # category_orders={"pareto_solution":[global_pareto_map[1], global_pareto_map[0], global_pareto_map[-1]]},
								  category_orders={"pareto_solution": ["Yes", "No"]},
								  title="Results of Feature Selection Methods Recommendation<br>Strategy: " + strategy,
								  hover_data={"pareto_solution": False,
											  "current_pareto": False,
											  "subset": True,
											  "fs methods": True},
								  labels={
									  metrics_choosed[0]: metrics_choosed[0].replace("_", " "),
				                      metrics_choosed[1]: metrics_choosed[1].replace("_", " "),
				                      metrics_choosed[2]: metrics_choosed[2].replace("_", " ")
			                      }, custom_data=["mask"])
		elif len(metrics_choosed) == 2:

			graph = px.scatter(metrics_df, x=metrics_choosed[0], y=metrics_choosed[1], color='pareto_solution',
							   symbol="current_pareto", symbol_map=symbol_map,
			                   category_orders={"pareto_solution": ["Yes", "No"]},
			                   title="Results of Feature Selection Methods Recommendation<br>Strategy: " + strategy,
			                   hover_data={"pareto_solution": False,
			                               "subset": True,
			                               "fs methods": True},
			                   labels={
				                   metrics_choosed[0]: metrics_choosed[0].replace("_", " "),
				                   metrics_choosed[1]: metrics_choosed[1].replace("_", " "),
			                   }, custom_data=["mask"])
		
		graph.update_traces(marker=dict(size=10, line=dict(width=2, color='White')))
		graph.update_layout(
			legend=dict(
				title_text='',
				orientation="h",
				yanchor="bottom",
				y=-0.2,
				xanchor="center",
				x=0.5
			)
		)
		return graph
	else:
		if strategy == "preference":
			preference = [p.replace(" ", "_") for p in preference if p is not None]
			comparation_result = compare_fs_methods_by_preference(metrics_df, preference, True)
		elif strategy == "average by scores":
			comparation_result = compare_fs_methods_by_average_of_scores(metrics_df, True)
		else:
			comparation_result = compare_fs_methods_by_average_of_rank(metrics_df, True)
		df_sorted = comparation_result["df_sorted"].round(round_digits)

		for features in metrics_df.index.tolist():
			mask = fs_results["features_selected"][features]["mask"]
			subset = []
			for i in range(len(mask)):
				if mask[i]:
					subset.append(i)
			subset_str = str(subset)
			j = 25
			while j < len(subset_str):
				if subset_str[j] == ",":
					subset_str = subset_str[0: j + 1] + "<br>" + subset_str[j + 1:]
				else:
					while j < len(subset_str) and subset_str[j] != ",":
						j += 1
					subset_str = subset_str[0: j + 1] + "<br>" + subset_str[j + 1:]
				j += 25
			df_sorted.loc[features, "subset"] = subset_str
			fs_methods_list = []
			for method_type in fs_results["features_selected"][features]["methods"].keys():
				fs_methods_list += fs_results["features_selected"][features]["methods"][method_type]
			df_sorted.loc[features, "fs methods"] = str(fs_methods_list).replace(
				"[", "").replace("]", "").replace("'", "").replace("_", " ")
		
		df_sorted["rank"] = range(1, len(df_sorted) + 1)
		num_colors = len(df_sorted)
		color_scale = px.colors.sequential.Blues[::-1]
		color_list = [color_scale[int(i * len(color_scale) / num_colors)] for i in range(num_colors)]
		
		color_discrete_map = {str(i + 1): color for i, color in enumerate(color_list)}
		
		df_sorted_reset = df_sorted.reset_index().dropna()
		df_sorted_reset.rename(columns={"index": "mask"}, inplace=True)
		hover_dic = {col:True for col in df_sorted_reset.columns if col not in metrics_choosed and col != 'mask'}
		# hover_dic['color'] = False
		if len(metrics_choosed) == 3:
			graph = px.scatter_3d(df_sorted_reset, x=metrics_choosed[0], y=metrics_choosed[1],
			                      z=metrics_choosed[2], color=df_sorted_reset["rank"].astype(str),
			                      title="Results of Feature Selection Methods Recommendation<br>Strategy: " + strategy,
			                      hover_data=hover_dic,
			                      color_discrete_map=color_discrete_map,
			                      labels={
				                      metrics_choosed[0]: metrics_choosed[0].replace("_", " "),
				                      metrics_choosed[1]: metrics_choosed[1].replace("_", " "),
				                      metrics_choosed[2]: metrics_choosed[2].replace("_", " ")
			                      }, custom_data=["mask"])
		
		elif len(metrics_choosed) == 2:
			graph = px.scatter(df_sorted_reset, x=metrics_choosed[0], y=metrics_choosed[1],
			                   color=df_sorted_reset["rank"].astype(str),
			                   title="Results of Feature Selection Methods Recommendation<br>Strategy: " + strategy,
			                   hover_data=hover_dic,
			                   color_discrete_map=color_discrete_map,
			                   labels={
				                   metrics_choosed[0]: metrics_choosed[0].replace("_", " "),
				                   metrics_choosed[1]: metrics_choosed[1].replace("_", " ")
			                   }, custom_data=["mask"])
		
		graph.update_traces(marker=dict(size=10, line=dict(width=2, color='White')))
		graph.update_layout(
			legend_title_text='rank'
		)
		return graph


def create_metrics_options(metrics_list: list = ["retention rate", "accuracy", "r2", "mae", "rmse", "kendalltau",
                                                 "relative influence changes", "RI", "RIA", "dcor"]):
	metrics_options = []
	for metric in metrics_list:
		for k, v in metrics_map.items():
			if metric in v:
				option = {"label": f"{k} ({metric})", "value": metric}
				metrics_options.append(option)
				break
	return metrics_options


app.layout = html.Div(children=[
	html.Div(
		[html.Div([
			html.A(
				href="https://restore-lab.fr/", target="_blank",
				style={'display': 'block', 'max-width': '65%', },
				children=html.Img(src='/assets/logo_RESTORE.png',
								  style={'max-width': '100%', 'max-height': '100%', 'objectFit': 'contain'})
			),
			html.A(
				href="https://www.irit.fr/", target="_blank",
				style={'display': 'block', 'max-width': '35%'},
				children=html.Img(src='/assets/logo_IRIT.png',
								  style={'max-width': '100%', 'max-height': '100%', 'objectFit': 'contain'})
			)
		], style={'display': 'flex', 'justifyContent': 'left', 'max-width': '50%'}),
			html.Div([
				html.A(
					href="https://www.laregion.fr/", target="_blank",
					style={'display': 'block', 'max-width': '100%', 'margin': '2px'},
					children=html.Img(src='/assets/logo-occitanie.png',
									  style={'max-width': '100%', 'max-height': '100%', 'objectFit': 'contain'})
				)
			], style={'display': 'flex', 'justifyContent': 'left', 'max-width': '10%'}),
			html.Div([
				html.A(
					href="https://www.ut-capitole.fr/", target="_blank",
					style={'display': 'block', 'max-width': '25%', 'margin': '2px'},
					children=html.Img(src='/assets/logo-UTC.jpg',
									  style={'max-width': '100%', 'max-height': '100%', 'objectFit': 'contain'})
				),
				html.A(
					href="https://www.univ-tlse3.fr/", target="_blank",
					style={'display': 'block', 'max-width': '75%'},
					children=html.Img(src='/assets/logo_UT3.jpg',
									  style={'max-width': '100%', 'max-height': '100%', 'objectFit': 'contain'})
				)
			], style={'display': 'flex', 'justifyContent': 'right', 'max-width': '40%'})],
		style={
			'display': 'flex',
			'justifyContent': 'space-between',
			'padding': '0 25px',
			'max-height': '100px',
			'overflow': 'hidden'
		}
	),
	html.Div(children=[
		html.Div(children=html.H1("auto-xFS: Feature Selection Methods Recommendation System"),
				 style={"textAlign": "center", "fontSize": "60px"}),
		dcc.Store(id="n_model_params", data=1),
		dcc.Store(id="n_tensorflow_model_init_params", data=1),
		dcc.Store(id="n_tensorflow_model_compile_params", data=1),
		dcc.Store(id="n_user_defined_subsets", data=1),
		dcc.Store(id="status", data="not submitted"),
		dbc.Table(children=[
			html.Tr([
				html.Th("Parameter", style={"textAlign": "center"}), html.Th("Value", style={"textAlign": "center"}),
				html.Th("Operation", style={"textAlign": "center"})
			]),
			html.Tr([html.Th("Meta-learning function", style={"textAlign": "center"}),
					 html.Td(html.Div([dbc.Switch(id='user_consent', value=False,
												  className="d-flex align-items-center me-10",
												  input_style={"margin-right" : "1em"},
												  label=consent_text,
												  label_style={"margin-bottom" : 0}
												  )],
							 style={"textAlign": "left"})),
					 html.Td()]),
			html.Tr([html.Th("Data set", style={"textAlign": "center"}),
			         html.Td(html.Div([dcc.Upload(
				         html.A('Upload your data set',
				                style={"display": "inline-block",
				                       "backgroundColor": "#0d6efd", "margin": "0 auto", "color": "white",
				                       "borderRadius": "5px", "padding": "10px"},)
				         , id="dataset", style={"cursor": "pointer"}, accept='.pkl, .csv', multiple=False), html.Span(id="filename")]),
				         style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Feature selection methods", style={"textAlign": "center"}),
			         html.Td(dcc.Dropdown(
				         options=list(fs_methods.keys()),
						 value=list(fs_methods.keys()), placeholder="fs methods",
				         multi=True,
				         id="fs_methods_choosed", ), style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Select features by", style={"textAlign": "center"}),
			         html.Td(dcc.Dropdown(options=["model", "percentage"], id="by", placeholder="select features by"),
			                 style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Percentage", style={"textAlign": "center"}),
			         html.Td(
				         dbc.Input(id="percentage", placeholder="percentage", type="number", min=0, max=100,
				                   style={"border": "0",
				                          "backgroundColor": "#fff",
				                          "outline": "none",
				                          "height": "30px",
				                          "textAlign": "center"}),
				         style={"textAlign": "center"}),
			         html.Td()], id="tr_percentage", style={"display": "none"}),
			html.Tr([html.Th("User-defined feature subsets", style={"textAlign": "center"}),
			         html.Td(dbc.Table([html.Tr([html.Td("Subset 1 : "), html.Td(
				         dbc.Input(id={"type": "user_defined_subset", "index": 0},
				                   placeholder="subset",
				                   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
				                          "height": "30px", "textAlign": "center"}))])], id="user_defined_subsets",
			                           responsive=True)),
			         html.Td(html.Div(
				         [dbc.Button("Add", id="add_user_defined_subset", n_clicks=0, style={"width": "83px"}),
				          dbc.Button("Remove", id="remove_user_defined_subset", n_clicks=0)],
				         style={"display": "flex", "justifyContent": "space-between",
				                "alignItems": "center", "height": "100%"}), style={"height": "100%"})],
			        id="tr_user_defined_subsets"),
			html.Tr([html.Th("Problem type", style={"textAlign": "center"}), html.Td(
				dcc.Dropdown(options=["classification", "regression"], placeholder="problem type",
				             id="problem_type"),
				style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Model for training", style={"textAlign": "center"}),
			         html.Td(
				         dcc.Dropdown(
					         options=list(models["sklearn"].keys()) + list(models["xgboost"].keys()) + list(
						         models["tensorflow"].keys()),
					         placeholder="model for training",
					         id="model_for_training"),
				         style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Model hyperparameters", style={"textAlign": "center"}),
			         html.Td(dbc.Table(id="model_params",
			                           responsive=True)),
			         html.Td(html.Div(
				         [dbc.Button("Add", id="add_model_param", n_clicks=0, style={"width": "83px"}),
				          dbc.Button("Remove", id="remove_model_param", n_clicks=0)],
				         style={"display": "flex", "justifyContent": "space-between",
				                "alignItems": "center", "height": "100%"}), style={"height": "100%"})],
			        id="tr_model_params"),
			html.Tr([html.Th("Init hyperparameters", style={"textAlign": "center"}),
			         html.Td(dbc.Table([html.Tr([html.Td("Input layer : ", style={"width": "70px"}), html.Td(
				         dbc.Input(id={"type": "layer_name", "index": 0}, placeholder="layer name",
				                   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
				                          "height": "30px", "width": "120px", "textAlign": "center"}),
				         style={"width": "150px"}),
			                                     html.Td(dbc.Input(id={"type": "units", "index": 0},
			                                                       placeholder="units",
			                                                       style={"border": "0",
			                                                              "backgroundColor": "#fff",
			                                                              "outline": "none",
			                                                              "height": "30px",
			                                                              "width": "80px", "textAlign": "center"}),
			                                             style={"width": "110px"}), html.Td(
					         dcc.Dropdown(id={"type": "activation", "index": 0},
					                      options=["sigmoid", "relu", "tanh", "elu", "selu", "softmax"],
					                      placeholder="activation",
					                      style={"backgroundColor": "transparent", "textAlign": "center"}),
					         style={"height": "80px"})], style={"verticalAlign": "middle"}),
			                            html.Tr([html.Td("Output layer : ", style={"width": "70px"}), html.Td(
				                            dbc.Input(id={"type": "layer_name", "index": 1}, placeholder="layer name",
				                                      style={"border": "0", "backgroundColor": "#fff",
				                                             "outline": "none",
				                                             "height": "30px", "width": "120px",
				                                             "textAlign": "center"}),
				                            style={"width": "150px"}),
			                                     html.Td(dbc.Input(id={"type": "units", "index": 1},
			                                                       placeholder="units",
			                                                       style={"border": "0",
			                                                              "backgroundColor": "#fff",
			                                                              "outline": "none",
			                                                              "height": "30px",
			                                                              "width": "80px", "textAlign": "center"}),
			                                             style={"width": "110px"}), html.Td(
					                            dcc.Dropdown(id={"type": "activation", "index": 1},
					                                         options=["sigmoid", "relu", "tanh", "elu", "selu",
					                                                  "softmax"],
					                                         placeholder="activation",
					                                         style={"backgroundColor": "transparent",
					                                                "textAlign": "center"}), style={"height": "80px"})],
			                                    style={"verticalAlign": "middle"})],
			                           id="tensorflow_model_init_params", responsive=True)),
			         html.Td(html.Div([dbc.Button("Add", id="add_tensorflow_model_init_param", n_clicks=0,
			                                      style={"width": "83px"}),
			                           dbc.Button("Remove", id="remove_tensorflow_model_init_param",
			                                      n_clicks=0)], style={"height": "100%", "display": "flex",
			                                                           "justifyContent": "space-between",
			                                                           "alignItems": "center"}))],
			        id="tr_tensorflow_model_init_params"),
			html.Tr([html.Th("Compile hyperparameters", style={"textAlign": "center"}),
			         html.Td(dbc.Table([html.Tr([html.Td("Parameter 1 : "), html.Td(
				         dbc.Input(id={"type": "tensorflow_model_compile_param_name", "index": 1},
				                   placeholder="Parameter name",
				                   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
				                          "height": "30px", "textAlign": "center"})), html.Td(
				         dbc.Input(id={"type": "tensorflow_model_compile_param_value", "index": 1},
				                   placeholder="Parameter value",
				                   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
				                          "height": "30px", "textAlign": "center"}))])],
			                           id="tensorflow_model_compile_params",
			                           responsive=True)),
			         html.Td(html.Div([dbc.Button("Add", id="add_tensorflow_model_compile_param",
			                                      n_clicks=0, style={"width": "83px"}),
			                           dbc.Button("Remove", id="remove_tensorflow_model_compile_param",
			                                      n_clicks=0)], style={"height": "100%", "display": "flex",
			                                                           "justifyContent": "space-between",
			                                                           "alignItems": "center"}))],
			        id="tr_tensorflow_model_compile_params"),
			html.Tr([html.Th("Fit hyperparameters", style={"textAlign": "center"}),
			         html.Td(
				         dbc.Table([html.Tr(
					         [html.Td("Epochs : "), html.Td(dbc.Input(id="epochs", placeholder="epochs", type="number",
					                                                  style={"border": "0",
					                                                         "backgroundColor": "#fff",
					                                                         "outline": "none",
					                                                         "height": "30px",
					                                                         "marginLeft": "10px",
					                                                         "textAlign": "center"}))])])),

			         html.Td()], id="tr_tensorflow_model_fit_params"),

			html.Tr([html.Th("Explanation", style={"textAlign": "center"}), html.Td(
				dcc.Dropdown(id="explanation", placeholder="explanation"),
				style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Explainer Parameters", style={"textAlign": "center"}), html.Td(
				dbc.Table(
					[html.Tr([html.Td("discretize continuous : "), dcc.Dropdown(id="discretize_continuous",
					                                                            options=["True", "False"],
					                                                            value="True",
					                                                            placeholder="discretize continuous",
					                                                            style={
						                                                            "backgroundColor": "transparent"})]),
					 html.Tr(
						 [html.Td("num samples : "), dbc.Input(id="num_samples", placeholder="num samples",
						                                       style={"border": "0",
						                                              "backgroundColor": "#fff",
						                                              "outline": "none", "height": "30px",
						                                              "textAlign": "center"},
						                                       type="number")])], responsive=True),
				style={"textAlign": "center"}),
			         html.Td()], id="lime_params"),
			html.Tr([html.Th("Explainer Parameters", style={"textAlign": "center"}), html.Td(
				dbc.Table([html.Tr([html.Td("num background samples : "),
				                    dbc.Input(id="num_background_samples",
				                              placeholder="num background samples",
				                              style={"border": "0", "backgroundColor": "#fff",
				                                     "outline": "none", "height": "30px", "textAlign": "center"},
				                              type="number")])], responsive=True),
				style={"textAlign": "center"}),
			         html.Td()], id="shap_params"),
			html.Tr([html.Th("Explainer Parameters", style={"textAlign": "center"}), html.Td(
				dbc.Table([html.Tr([html.Td("rate : "), dbc.Input(id="rate", placeholder="rate",
				                                                  style={"border": "0",
				                                                         "backgroundColor": "#fff",
				                                                         "outline": "none",
				                                                         "height": "30px", "textAlign": "center"},
				                                                  value=0.25,
				                                                  type="number")]),
				           html.Tr(
					           [html.Td("fvoid : "), dbc.Input(id="coalitional_fvoid", placeholder="fvoid",
					                                           style={"border": "0",
					                                                  "backgroundColor": "#fff",
					                                                  "outline": "none", "height": "30px",
					                                                  "textAlign": "center"},
					                                           type="number")]),
				           html.Tr([html.Td("method : "), dcc.Dropdown(id="method",
				                                                       options=["pca", "spearman", "vif"],
				                                                       value="spearman",
				                                                       placeholder="method",
				                                                       style={
					                                                       "backgroundColor": "transparent"})]),
				           html.Tr([html.Td("reverse : "), dcc.Dropdown(id="reverse",
				                                                        options=["True", "False"],
				                                                        value="False",
				                                                        placeholder="reverse",
				                                                        style={
					                                                        "backgroundColor": "transparent"})]),
				           html.Tr([html.Td("complexity : "), dcc.Dropdown(id="complexity",
				                                                           options=["True", "False"],
				                                                           value="False",
				                                                           placeholder="complexity",
				                                                           style={
					                                                           "backgroundColor": "transparent"})]),
				           html.Tr([html.Td("scaler : "), dcc.Dropdown(id="scaler",
				                                                       options=["True", "False"],
				                                                       value="False",
				                                                       placeholder="scaler",
				                                                       style={
					                                                       "backgroundColor": "transparent"})])],
				          responsive=True),
				style={"textAlign": "center"}),
			         html.Td()], id="coalitional_method_params"),
			html.Tr([html.Th("Explainer Parameters", style={"textAlign": "center"}), html.Td(
				dbc.Table([html.Tr([html.Td("fvoid : "), dbc.Input(id="complete_fvoid", placeholder="fvoid",
				                                                   style={"border": "0",
				                                                          "backgroundColor": "#fff",
				                                                          "outline": "none",
				                                                          "height": "30px",
				                                                          "textAlign":"center"},
				                                                   type="number")])], responsive=True),
				style={"textAlign": "center"}),
			         html.Td()], id="complete_method_params"),
			html.Tr([html.Th("Metrics", style={"textAlign": "center"}),
			         html.Td(dcc.Dropdown(
				         options=create_metrics_options(),  multi=True, placeholder="metrics", id="metrics"),
				         style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Strategy for comparation", style={"textAlign": "center"}), html.Td(
				dcc.Dropdown(options=["pareto", "average by scores", "average by rank", "preference"], placeholder="strategy",
				             id="strategy"),
				style={"textAlign": "center"}),
			         html.Td()]),
			html.Tr([html.Th("Preference", style={"textAlign": "center"}),
			         html.Td(html.Div(id="preference_dropdown_input", style={"backgroundColor": "#f8f9fa"}),
			                 style={"textAlign": "center"}),
			         html.Td()], "preference_display_input", style={"display": "none"}),
			html.Tr([html.Td(html.Div([dbc.Button('Submit', id='submit_button', n_clicks=0),
			                           dbc.Button("Download result file", id="download_button", n_clicks=0,
			                                      style={"display": "none"}),
			                           dcc.Download(id="download_file")],
			                          style={"display": "flex", "justifyContent": "space-evenly"}),
			                 colSpan=3, id="buttons",
			                 )])
		], style={"width": "100%", "borderLeft": "1px #c6c7c8 solid", "borderRight": "1px #c6c7c8 solid",
		          "backgroundColor": "#f8f9fa"},
			color="light", bordered=True, responsive=True, class_name="table"),
		html.Div(id="form_info", style={"width": "100%", "height": "70px"})
	], style={"width": "90%", "margin": "20px auto"}),
	html.Div(
		[
			dcc.Store(id="result_key"),
			html.Div([html.Label(children="Select a strategy : ", style={"display": "inline"}),
			          dcc.Dropdown(["pareto", "average by scores", "average by rank", "preference"],
			                       id="strategy_choosed", placeholder="strategy", style={"textAlign": "center"}),
			          html.Br(),
					  html.Div([html.Label(children="Choose your preference : "),
								html.Div(id="preference_dropdown_result",
										 style={"textAlign": "center"},
										 )],
							   id="preference_display_result", style={"backgroundColor": "#f8f9fa", "style": "none"}
							   ),
			          html.Br(),
			          html.Label(children="Choose metrics to show in the graph : ", id="choose_metrics_label"),
			          html.Div(dbc.Checklist(labelStyle={"marginRight": "80px"}, inline=True, id="metrics_showed", className="mx-2")),
			          html.Div(id="result_part_info"),
			          html.Div(id="result_graph_div", children=dcc.Graph(id="result_graph",
			                                                             figure={
				                                                             'data': [],
				                                                             'layout': {
					                                                             'xaxis': {'visible': False},
					                                                             'yaxis': {'visible': False}
				                                                             }
			                                                             }))],
			         className="col-sm-12 col-lg-5 mx-2"),
			html.Div(children=[
				html.Div(id="table", style={"height": "30%", "marginBottom": "25px"}),
				html.Div(id="summary_plot", style={"width": "100%", "textAlign": "center"})
			], className="col-sm-12 col-lg-6")
		], className="row", id="result_part", style={"display": "none"}),
	html.Footer(
        html.Div([
			html.Ul(
                [
                    html.Li("Haomiao Wang", style={'display': 'inline', 'marginRight': '10px'}),
                    html.Li("Haoran Zhou", style={'display': 'inline', 'marginRight': '10px'}),
					html.Li("Julien Aligon", style={'display': 'inline', 'marginRight': '10px'}),
					html.Li("Paul Monsarrat", style={'display': 'inline', 'marginRight': '10px'}),
                    html.Li("Chantal Soulé-Dupuy", style={'display': 'inline', 'marginRight': '10px'}),
                ],
                style={'listStyleType': 'none', 'padding': '0', 'margin': '0'}
            ),
            html.P("© 2025 RESTORE (Université de Toulouse, INSERM 1301, CNRS 5070, EFS, ENVT) & IRIT (Université Toulouse Capitole, CNRS/UMR 5505)",
				   style={'display': 'inline', 'marginLeft': '20px'}),
        ]),
        style={
            'backgroundColor': 'rgba(0, 0, 0, 0.5)',
            'color': 'white',
            'textAlign': 'center',
            'padding': '10px',
            'bottom': '0',
            'width': '100%',
        }
    )])


@app.callback(
	Output("filename", "children"),
	Input("dataset", "filename"),
	prevent_initial_call=True
)
def update_filename(filename):
	return filename


def determine_problem_type(y: pd.Series):
	unique_values = y.unique()
	num_unique_values = len(unique_values)
	
	if num_unique_values == 2:
		return "classification"
	elif pd.api.types.is_numeric_dtype(y):
		return "regression"
	else:
		raise ValueError("Multiclass classification is not supported in the current version. Please select binary classification or stay tuned for future updates.")

def create_alert(content, color):
	return dbc.Alert(content, color=color)

@app.callback(
	Output("form_info", "children", allow_duplicate=True),
	Output("status", "data", allow_duplicate=True),
	Output("strategy_choosed", "value", allow_duplicate=True),
	Output("preference_dropdown_result", "children", allow_duplicate=True),
	Output("metrics_showed", "options", allow_duplicate=True),
	Output("metrics_showed", "value", allow_duplicate=True),
	Output("result_key", "data", allow_duplicate=True),
	Output("user_defined_subsets", "children", allow_duplicate=True),
	Output("n_user_defined_subsets", "data", allow_duplicate=True),
	Output("model_for_training", "value", allow_duplicate=True),
	Output("problem_type", "value", allow_duplicate=True),  #
	Output("fs_methods_choosed", "value", allow_duplicate=True),
	Output("fs_methods_choosed", "options", allow_duplicate=True),
	Output("by", "value", allow_duplicate=True),
	Output("percentage", "value", allow_duplicate=True),
	Output("explanation", "value", allow_duplicate=True),
	Output("discretize_continuous", "value", allow_duplicate=True),
	Output("num_samples", "value", allow_duplicate=True),
	Output("num_background_samples", "value", allow_duplicate=True),
	Output("rate", "value", allow_duplicate=True),
	Output("coalitional_fvoid", "value", allow_duplicate=True),
    Output("method", "value", allow_duplicate=True),
    Output("reverse", "value", allow_duplicate=True),
    Output("complexity", "value", allow_duplicate=True),
    Output("scaler", "value", allow_duplicate=True),
    Output("complete_fvoid", "value", allow_duplicate=True),
    Output("metrics", "value", allow_duplicate=True),
    Output("strategy", "value", allow_duplicate=True),
	Output("model_params", "children", allow_duplicate=True),
	Output("n_model_params", "data", allow_duplicate=True),
	Output("tensorflow_model_init_params", "children", allow_duplicate=True),
	Output("n_tensorflow_model_init_params", "data", allow_duplicate=True),
	Output("tensorflow_model_compile_params", "children", allow_duplicate=True),
	Output("n_tensorflow_model_compile_params", "data", allow_duplicate=True),
    Output("epochs", "value", allow_duplicate=True),
	Output("preference_dropdown_input", "children", allow_duplicate=True),
	Output("user_consent", "disabled"),
    Input("dataset", "contents"),
    Input("dataset", "filename"),
	State("user_consent", "value"),
    prevent_initial_call=True
)
def set_hyperparam_auto(upload_contents, filename, user_content):
	try:
		if '.csv' in filename:
			# if dataset (csv) upload
			X, y, preprocessed, problem_type = file_to_df(upload_contents, filename)
			form_info, status, hyperparams = get_most_similar_problem(X, y, problem_type, user_content)
			if len(preprocessed) > 0:
				form_info.children = form_info.children + "\nPreprocessing: " + str(preprocessed)[1: -1].replace("'",
																												 "")
				form_info.style = {"white-space": "pre-line"}
			result_key = None
		else:# if result (pickle) upload
				content_type, content_string = upload_contents.split(',')
				decoded = base64.b64decode(content_string)
				data_and_results_pickle = pickle.load(io.BytesIO(decoded))
				redis_connection = redis.StrictRedis(connection_pool=pool)
				result_key = str(uuid.uuid4())
				redis_connection.setex(result_key, 3600, decoded)
				hyperparams = data_and_results_pickle['hyperparameters']
				# key_fullset = str([True] * len(X.columns)).replace(" ", "")
				form_info = create_alert("Hyperparameters have been successfully retrieved from the uploaded file.", "success"),
				status = "hyperparameter_retrieved"
		problem_type = hyperparams["problem_type"]
		user_defined_subsets = hyperparams["user_defined_subsets"]
		user_defined_subsets_input = [html.Tr([html.Td(f"Subset {i+1} : "), html.Td(
					 dbc.Input(id={"type": "user_defined_subset", "index": i},
							   value=subset,
							   placeholder="subset",  # placeholder if no value returned
							   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
									  "height": "30px", "textAlign": "center"}))]) for i, subset in enumerate(user_defined_subsets)]
		n_user_defined_subsets = len(user_defined_subsets)
		model_name_for_training = hyperparams["model_name_for_training"]
		fs_methods_choosed = hyperparams["fs_methods_choosed"]
		by = hyperparams["by"]
		percentage = hyperparams["percentage"]
		explanation = hyperparams["explanation"]
		discretize_continuous = hyperparams["discretize_continuous"]
		num_samples = hyperparams["num_samples"]
		num_background_samples = hyperparams["num_background_samples"]
		rate = hyperparams["rate"]
		coalitional_fvoid = hyperparams["coalitional_fvoid"]
		method = hyperparams["method"]
		reverse = hyperparams["reverse"]
		complexity = hyperparams["complexity"]
		scaler = hyperparams["scaler"]
		complete_fvoid = hyperparams["complete_fvoid"]
		strategy_for_comparation = hyperparams["strategy_for_comparation"]

		preference_order = hyperparams["preference"]
		metrics_choosed = hyperparams["metrics_choosed"]
		preference_dropdown_input = get_preference_dropdown(
			order=metrics_choosed if preference_order in [None, []] else preference_order,
			type_id="preference",
			is_active=True)
		preference_dropdown_result  = get_preference_dropdown(
			order=metrics_choosed if preference_order in [None, []] else preference_order, type_id="preference_result",
			is_active=True)

		model_param_name = hyperparams["model_param_name"]
		model_param_value = hyperparams["model_param_value"]
		model_param_input = [html.Tr([html.Td(f"Parameter {i+1} : "), html.Td(
					 dcc.Dropdown(id={"type": "model_param_name", "index": i},
								  value=param,
								  style={"backgroundColor": "transparent", "textAlign": "center",
										 "width": "500px"}), style={"height": "80px"}), html.Td(
					 dbc.Input(id={"type": "model_param_value", "index": i},
							   value=model_param_value[i],
							   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
									  "height": "30px", "textAlign": "center"}))])for i, param in enumerate(model_param_name)]
		layer_name = hyperparams['layer_name']
		units = hyperparams["units"]
		activation = hyperparams["activation"]
		tensorflow_model_init_params = [html.Tr([html.Td("Input layer : ", style={"width": "70px"}), html.Td(
			dbc.Input(id={"type": "layer_name", "index": 0}, value=layer_name[0],
					  style={"border": "0", "backgroundColor": "#fff", "outline": "none",
							 "height": "30px", "width": "120px", "textAlign": "center"}),
			style={"width": "150px"}),
				  html.Td(dbc.Input(id={"type": "units", "index": 0},
									value=units[0],
									style={"border": "0",
										   "backgroundColor": "#fff",
										   "outline": "none",
										   "height": "30px",
										   "width": "80px", "textAlign": "center"}),
						  style={"width": "110px"}), html.Td(
				dcc.Dropdown(id={"type": "activation", "index": 0},
							 options=["sigmoid", "relu", "tanh", "elu", "selu",
									  "softmax"],
							 value=activation[0],
							 style={"backgroundColor": "transparent", "textAlign": "center"}),
				style={"height": "80px"})], style={"verticalAlign": "middle"})]
		# not same layer numeration logic with when add_or_remove
		added_layers =[html.Tr([html.Td(f"Layer {i} : ", style={"width": "70px"}), html.Td(
		dbc.Input(id={"type": "layer_name", "index": i},
				  value=l_name,
				  style={"border": "0", "backgroundColor": "#fff", "outline": "none",
						 "height": "30px", "width": "120px", "textAlign": "center"}),
		style={"width": "150px"}),
					   html.Td(dbc.Input(id={"type": "units", "index": i},
										 value=units[i],
										 style={"border": "0",
												"backgroundColor": "#fff",
												"outline": "none",
												"height": "30px",
												"width": "80px", "textAlign": "center"}),
							   style={"width": "110px"}), html.Td(
			dcc.Dropdown(id={"type": "activation", "index": i},
						 options=["sigmoid", "relu", "tanh", "elu", "selu",
								  "softmax"],
						 value=activation[i],
						 style={"backgroundColor": "transparent", "textAlign": "center"}),
			style={"height": "80px"})], style={"verticalAlign": "middle"}) for i, l_name in enumerate(layer_name)
					   if i>0 and i<len(layer_name)-1]
		tensorflow_model_init_params.extend(added_layers)
		tensorflow_model_init_params.append(html.Tr([html.Td("Output layer : ", style={"width": "70px"}), html.Td(
			 dbc.Input(id={"type": "layer_name", "index": 1}, value=layer_name[-1],
					   style={"border": "0", "backgroundColor": "#fff",
							  "outline": "none",
							  "height": "30px", "width": "120px",
							  "textAlign": "center"}),
			 style={"width": "150px"}),
				  html.Td(dbc.Input(id={"type": "units", "index": 1},
									value=units[-1],
									style={"border": "0",
										   "backgroundColor": "#fff",
										   "outline": "none",
										   "height": "30px",
										   "width": "80px", "textAlign": "center"}),
						  style={"width": "110px"}), html.Td(
				 dcc.Dropdown(id={"type": "activation", "index": 1},
							  options=["sigmoid", "relu", "tanh", "elu", "selu",
									   "softmax"],
							  value=activation[-1],
							  style={"backgroundColor": "transparent",
									 "textAlign": "center"}), style={"height": "80px"})],
				 style={"verticalAlign": "middle"}))

		tensorflow_model_compile_param_name = hyperparams["tensorflow_model_compile_param_name"]
		tensorflow_model_compile_param_value = hyperparams["tensorflow_model_compile_param_value"]
		tensorflow_model_compile_param_input = [html.Tr([html.Td(f"Parameter {i+1} : "), html.Td(
					 dbc.Input(id={"type": "tensorflow_model_compile_param_name", "index": i},
							   value=param_name,
							   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
									  "height": "30px", "textAlign": "center"})), html.Td(
					 dbc.Input(id={"type": "tensorflow_model_compile_param_value", "index": i},
							   value=tensorflow_model_compile_param_value[i],
							   style={"border": "0", "backgroundColor": "#fff", "outline": "none",
									  "height": "30px", "textAlign": "center"}))]) for i, param_name in enumerate(tensorflow_model_compile_param_name)]
		epochs = hyperparams["epochs"]



		return [form_info, status,
				strategy_for_comparation, preference_dropdown_result,
				(create_metrics_options(metrics_choosed)), (metrics_choosed[0:3]), result_key,
				user_defined_subsets_input, n_user_defined_subsets,
				model_name_for_training, problem_type, fs_methods_choosed, get_fss_by_problem(problem_type),
				by, percentage,
				explanation, discretize_continuous, num_samples, num_background_samples, rate,
				coalitional_fvoid, method, reverse, complexity, scaler, complete_fvoid,
				metrics_choosed, strategy_for_comparation,
				model_param_input, len(model_param_input),
				tensorflow_model_init_params, len(tensorflow_model_init_params) -1,
				tensorflow_model_compile_param_input, len(tensorflow_model_compile_param_input), epochs,
				preference_dropdown_input, True]
	except Exception as e:
		ctx = dash.callback_context
		return [create_alert(str(e), "danger"),
			        "error"] + [dash.no_update for _ in range(len(ctx.outputs_list) - 2)]


def get_most_similar_problem(X, y, problem_type, user_content):
	if user_content:
		new_problem = pd.DataFrame([{"num_rows": X.shape[0], "num_columns": X.shape[1]}])
		with Session() as session:
			session.begin()
			sql = text(
				"SELECT result_key, dataset_name, submit_date, num_rows, num_columns "
				"FROM meta_data WHERE  problem_type = :problem_type")
			rows = session.execute(sql, {'problem_type': problem_type}).fetchall()
			if len(rows) != 0:
				rows_df = pd.DataFrame(rows,
									   columns=["result_key", "dataset_name", "submit_date", "num_rows", "num_columns"])
				rows_df['similarity'] = (
						(rows_df[['num_rows', 'num_columns']] - new_problem[
							['num_rows', 'num_columns']].values) ** 2).sum(
					axis=1)
				sorted_df = rows_df.sort_values(by=['similarity', 'submit_date'], ascending=[True, False])
				latest_result_key = sorted_df.iloc[0]['result_key']

				sql = text(
					"SELECT * "
					"FROM hyperparameter_data WHERE  result_key = :result_key")
				row = session.execute(sql, {'result_key': latest_result_key}).fetchone()
				if row is not None:
					hyperparameters = dict(row._mapping)
					for k in ['user_defined_subsets', 'fs_methods_choosed', 'model_param_name', 'model_param_value',
							  'layer_name', 'units', 'activation',
							  'tensorflow_model_compile_param_name', 'tensorflow_model_compile_param_value',
							  'metrics_choosed', 'preference']: # if type(item) == list
						hyperparameters[k] = eval(hyperparameters[k])
					hyperparameters['user_defined_subsets'] = [''] # do not recommande subsets
					hyperparameters['fs_methods_choosed'] = [f for f in hyperparameters['fs_methods_choosed']
																 if f in get_fss_by_problem(problem_type)]

					return create_alert("Dataset uploaded successfully. You may adjust the recommended hyperparameters as needed.",
										"success"), 'dataset_upload', hyperparameters
			return create_alert("No similar dataset found; using default hyperparameters.",
										"info"), 'dataset_upload', default_hyperparameters(problem_type)
	else:
		return create_alert("You did not enable hyperparameter tuning via meta-learning; using default values.",
							"info"), 'dataset_upload', default_hyperparameters(problem_type)

def get_fss_by_problem(problem_type):
	if problem_type == "classification":
		return list(fs_methods.keys())
	else:
		return [k for k,v in fs_methods.items() if v[problem_type]]

def default_hyperparameters(problem_type):
	metrics_choosed =  ['r2', 'dcor', 'relative influence changes', 'RI', 'retention rate'] if problem_type =="regression" else \
		['accuracy', 'dcor', 'relative influence changes', 'RI', 'retention rate']
	loss = 'mae' if problem_type =="regression" else 'accuracy'

	return {'user_defined_subsets': [],
			'model_name_for_training': 'Random Forest',
			'problem_type': problem_type,
			'fs_methods_choosed': get_fss_by_problem(problem_type),
			'by': 'model', 'percentage': 0.5,
			'explanation': 'TreeSHAP',
			'discretize_continuous': 'True', 'num_samples': 10,
			'num_background_samples': 10, 'rate': None, 'coalitional_fvoid': None, 'method': None,
			'reverse': False, 'complexity': None, 'scaler': None, 'complete_fvoid': None,
			'strategy_for_comparation': 'pareto', 'model_param_name': ['random_state'], 'model_param_value': ['42'],
			'layer_name': ['Dense', 'Dense'], 'units': [64, 1], 'activation': ['relu', 'relu'],
			'tensorflow_model_compile_param_name': ['loss'], 'tensorflow_model_compile_param_value': [loss],
			'epochs': 10, 'metrics_choosed': metrics_choosed,
			'preference': metrics_choosed,
			"metrics_showed": [{'label': 'accuracy (r2)', 'value': 'r2'},
							   {'label': 'explanation discernibility (dcor)', 'value': 'dcor'}],
			}

@app.callback(
	Output("user_defined_subsets", "children", allow_duplicate=True),
	Output("n_user_defined_subsets", "data", allow_duplicate=True),
	Input("add_user_defined_subset", "n_clicks"),
	Input("remove_user_defined_subset", "n_clicks"),
	State("n_user_defined_subsets", "data"),
	State("user_defined_subsets", "children"),
	prevent_initial_call=True
)
def add_or_remove_user_defined_subsets(n_clicks_add, n_clicks_remove, n_user_defined_subsets, user_defined_subsets):
	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = ""
	else:
		button_id = ctx.triggered[0]["prop_id"].split(".")[0]
	
	if button_id == "add_user_defined_subset":
		new_row = html.Tr([html.Td(f"Subset {n_user_defined_subsets + 1} : "), html.Td(
			dbc.Input(id={"type": "user_defined_subset", "index": (n_user_defined_subsets)},
			          placeholder="subset",
			          style={"border": "0", "backgroundColor": "#fff", "outline": "none",
			                 "height": "30px", "textAlign": "center"}))])
		user_defined_subsets.append(new_row)
		return user_defined_subsets, (n_user_defined_subsets + 1)
	elif button_id == "remove_user_defined_subset" and len(user_defined_subsets) > 0:
		user_defined_subsets.pop()
		return user_defined_subsets, (n_user_defined_subsets - 1)
	else:
		return dash.no_update


@app.callback(
	Output({"type": "model_param_name", "index": ALL}, "options"),
	Input("model_for_training", "value"),
	Input("problem_type", "value"),
	State("n_model_params", "data"),
)
def set_model_param_name_options(model_name_for_training, problem_type, n_model_params):
	if model_name_for_training is not None and problem_type is not None:
		for model_type in models.keys():
			if model_name_for_training in models[model_type].keys() and model_type != "tensorflow":
				model_class_for_training = models[model_type][model_name_for_training][problem_type]
				signature = inspect.signature(model_class_for_training.__init__)
				params = signature.parameters
				param_names = list(params.keys())
				param_names.pop(0)
				return [param_names] * n_model_params
		return [[]] * n_model_params
	else:
		return [[]] * n_model_params


@app.callback(
	Output("model_params", "children", allow_duplicate=True),
	Output("n_model_params", "data", allow_duplicate=True),
	Input("add_model_param", "n_clicks"),
	Input("remove_model_param", "n_clicks"),
	State("n_model_params", "data"),
	State("model_params", "children"),
	State("model_for_training", "value"),
	State("problem_type", "value"),
	prevent_initial_call=True
)
def add_or_remove_model_params(n_clicks_add, n_clicks_remove, n_model_params, model_params, model_name_for_training,
                               problem_type):
	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = ""
	else:
		button_id = ctx.triggered[0]["prop_id"].split(".")[0]
	if button_id == "add_model_param":
		for model_type in models.keys():
			if model_name_for_training in models[model_type].keys() and model_type != "tensorflow":
				model_class_for_training = models[model_type][model_name_for_training][problem_type]
				signature = inspect.signature(model_class_for_training.__init__)
				params = signature.parameters
				param_names = list(params.keys())
				param_names.pop(0)
				break
		new_row = html.Tr([html.Td(f"Parameter {n_model_params + 1} : "), html.Td(
			dcc.Dropdown(id={"type": "model_param_name", "index": (n_model_params + 1)},
			             placeholder="Parameter name",
			             options=param_names,
			             style={"backgroundColor": "transparent", "textAlign": "center", "width": "500px"}),
			style={"height": "80px"}), html.Td(
			dbc.Input(id={"type": "model_param_value", "index": (n_model_params + 1)},
			          placeholder="Parameter value",
			          style={"border": "0", "backgroundColor": "#fff", "outline": "none", "height": "30px",
			                 "textAlign": "center"}))])
		model_params.append(new_row)
		return model_params, (n_model_params + 1)
	elif button_id == "remove_model_param" and len(model_params) > 1:
		model_params.pop()
		return model_params, (n_model_params - 1)
	else:
		return dash.no_update


@app.callback(
	Output("tensorflow_model_init_params", "children", allow_duplicate=True),
	Output("n_tensorflow_model_init_params", "data", allow_duplicate=True),
	Input("add_tensorflow_model_init_param", "n_clicks"),
	Input("remove_tensorflow_model_init_param", "n_clicks"),
	State("n_tensorflow_model_init_params", "data"),
	State("tensorflow_model_init_params", "children"),
	prevent_initial_call=True
)
def add_or_remove_tensorflow_model_init_params(n_clicks_add, n_clicks_remove, n_tensorflow_model_init_params,
                                               tensorflow_model_init_params):
	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = ""
	else:
		button_id = ctx.triggered[0]["prop_id"].split(".")[0]
	
	if button_id == "add_tensorflow_model_init_param":
		new_row = html.Tr([html.Td(f"Layer {n_tensorflow_model_init_params} : ", style={"width": "70px"}), html.Td(
			dbc.Input(id={"type": "layer_name", "index": (n_tensorflow_model_init_params + 1)},
			          placeholder="layer name",
			          style={"border": "0", "backgroundColor": "#fff", "outline": "none",
			                 "height": "30px", "width": "120px", "textAlign": "center"}),
			style={"width": "150px"}),
		                   html.Td(dbc.Input(id={"type": "units", "index": (n_tensorflow_model_init_params + 1)},
		                                     placeholder="units",
		                                     style={"border": "0",
		                                            "backgroundColor": "#fff",
		                                            "outline": "none",
		                                            "height": "30px",
		                                            "width": "80px", "textAlign": "center"}),
		                           style={"width": "110px"}), html.Td(
				dcc.Dropdown(id={"type": "activation", "index": (n_tensorflow_model_init_params + 1)},
				             options=["sigmoid", "relu", "tanh", "elu", "selu", "softmax"],
				             placeholder="activation",
				             style={"backgroundColor": "transparent", "textAlign": "center"}),
				style={"height": "80px"})], style={"verticalAlign": "middle"})
		tensorflow_model_init_params.insert(len(tensorflow_model_init_params) - 1, new_row)
		return tensorflow_model_init_params, (n_tensorflow_model_init_params + 1)
	elif button_id == "remove_tensorflow_model_init_param" and len(tensorflow_model_init_params) > 2:
		tensorflow_model_init_params.pop(len(tensorflow_model_init_params) - 2)
		return tensorflow_model_init_params, (n_tensorflow_model_init_params - 1)
	else:
		return dash.no_update


@app.callback(
	Output("tensorflow_model_compile_params", "children", allow_duplicate=True),
	Output("n_tensorflow_model_compile_params", "data", allow_duplicate=True),
	Input("add_tensorflow_model_compile_param", "n_clicks"),
	Input("remove_tensorflow_model_compile_param", "n_clicks"),
	State("n_tensorflow_model_compile_params", "data"),
	State("tensorflow_model_compile_params", "children"),
	prevent_initial_call=True
)
def add_or_remove_tensorflow_model_compile_params(n_clicks_add, n_clicks_remove, n_tensorflow_model_compile_params,
                                                  tensorflow_model_compile_params):
	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = ""
	else:
		button_id = ctx.triggered[0]["prop_id"].split(".")[0]
	if button_id == "add_tensorflow_model_compile_param":
		new_row = html.Tr([html.Td(f"Parameter {n_tensorflow_model_compile_params + 1} : "), html.Td(
			dbc.Input(id={"type": "model_param_name", "index": (n_tensorflow_model_compile_params + 1)},
			          placeholder="Parameter name",
			          style={"border": "0", "backgroundColor": "#fff", "outline": "none", "height": "30px",
			                 "textAlign": "center"})), html.Td(
			dbc.Input(id={"type": "model_param_value", "index": (n_tensorflow_model_compile_params + 1)},
			          placeholder="Parameter value",
			          style={"border": "0", "backgroundColor": "#fff", "outline": "none", "height": "30px",
			                 "textAlign": "center"}))])
		tensorflow_model_compile_params.append(new_row)
		return tensorflow_model_compile_params, (n_tensorflow_model_compile_params + 1)
	elif button_id == "remove_tensorflow_model_compile_param" and len(tensorflow_model_compile_params) > 1:
		tensorflow_model_compile_params.pop()
		return tensorflow_model_compile_params, (n_tensorflow_model_compile_params - 1)
	else:
		return dash.no_update


@app.callback(
	Output("explanation", "options", allow_duplicate=True),
	Input("model_for_training", "value"),
	Input("problem_type", "value"),
	prevent_initial_call='initial_duplicate' 
)
def set_explanation_options(model_name_for_training, problem_type):
	if model_name_for_training is not None and problem_type is not None:
		if model_name_for_training in models["tensorflow"].keys():
			return (["LIME", "KernelSHAP", "DeepSHAP"])
		else:
			if model_name_for_training in models["sklearn"].keys():
				model_class = models["sklearn"][model_name_for_training][problem_type]
			elif model_name_for_training in models["xgboost"].keys():
				model_class = models["xgboost"][model_name_for_training][problem_type]
			else:
				raise Exception(model_name_for_training + " not recognized") 
			
			if model_class in tree_based_models:
				return (["LIME", "KernelSHAP", "TreeSHAP", "Coalitional Method", "Complete Method"])
			else:
				return (["LIME", "KernelSHAP", "Coalitional Method", "Complete Method"])
	else:
		return []


@app.callback(
	Output("tr_model_params", "style"),
	Output("tr_tensorflow_model_init_params", "style"),
	Output("tr_tensorflow_model_compile_params", "style"),
	Output("tr_tensorflow_model_fit_params", "style"),
	Input("model_for_training", "value")
)
def set_model_params_rows_display(model_name_for_training):
	if model_name_for_training in models["sklearn"].keys() or model_name_for_training in models["xgboost"].keys():
		return {"display": "table-row"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
	elif model_name_for_training in models["tensorflow"].keys():
		return {"display": "none"}, {"display": "table-row"}, {"display": "table-row"}, {"display": "table-row"}
	else:
		return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}


@app.callback(
	Output("tr_percentage", "style"),
	Input("by", "value")
)
def set_percentage_display(by):
	if by is not None:
		if by == "model":
			return {"display": "none"}
		else:
			return {"display": "table-row"}
	else:
		return {"display": "none"}


@app.callback(
	Output("lime_params", "style"),
	Output("shap_params", "style"),
	Output("coalitional_method_params", "style"),
	Output("complete_method_params", "style"),
	Input("explanation", "value")
)
def set_explanation_params_rows_display(explanation):
	if explanation is not None:
		if explanation == "LIME":
			return {"display": "table-row"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
		elif "SHAP" in explanation and "Tree" not in explanation:
			return {"display": "none"}, {"display": "table-row"}, {"display": "none"}, {"display": "none"}
		elif explanation == "Coalitional Method":
			return {"display": "none"}, {"display": "none"}, {"display": "table-row"}, {"display": "none"}
		elif explanation == "Complete Method":
			return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "table-row"}
		else:
			return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
	else:
		return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}


@app.callback(
	Output("preference_display_input", "style"),
	Output("preference_dropdown_input", "children", allow_duplicate=True),
	Input("strategy", "value"),
	Input("metrics", "value"),
	Input("result_key", "data"),
	Input("preference_dropdown_input", "children"),
	prevent_initial_call=True
)
def set_preference_display_and_dropdowns(strategy, metrics_choosed, result_key, old_preference_dropdown_input):
	if strategy == "preference":
		if result_key is not None:
			redis_connection = redis.StrictRedis(connection_pool=pool)
			data_and_results_pickle = redis_connection.get(result_key)
			data_and_results = pickle.loads(data_and_results_pickle)
			order = data_and_results["hyperparameters"]["preference"]
		else:
			order = metrics_choosed
		table_body = get_preference_dropdown(order=order, type_id="preference", is_active=(result_key is None))
		return {"display": "table-row"}, table_body
	else:
		return {"display": "none"}, old_preference_dropdown_input


def get_preference_dropdown(order, type_id, is_active=True):
	return [html.Div(
			[html.Div(f"Preference {i + 1} : ", className="col-4 text-end"),
			 html.Div(dcc.Dropdown(id={"type": type_id, "index": i + 1},
								   options=create_metrics_options(
									   order) if is_active else [metric],
								   value=metric,
								   placeholder=f"preference {i + 1}",
								   style={
									   "backgroundColor": "transparent"}),
					  className="col-6")], className="row")
		for i, metric in enumerate(order)]


@app.callback(
	Output("form_info", "children", allow_duplicate=True),
	Output("status", "data", allow_duplicate=True),
	Input("submit_button", "n_clicks"),
	State("dataset", "contents"),
	State("fs_methods_choosed", "value"),
	State("by", "value"),
	State("percentage", "value"),
	State("problem_type", "value"),
	State("model_for_training", "value"),
	State("explanation", "value"),
	State("metrics", "value"),
	State("strategy", "value"),
	[State({"type": "preference", "index": ALL}, "value")],
	
	prevent_initial_call=True
)
def verify_parameters(n_clicks, contents, fs_methods_choosed, by, percentage, problem_type, model_name_for_training,
					  explanation, metrics_choosed, strategy, preference):
	if contents is None:
		return [create_alert("Please upload your dataset.", "warning"), dash.no_update]
	
	if fs_methods_choosed is None or fs_methods_choosed == []:
		return [create_alert("Please choose at least one feature selection method.", "warning"), dash.no_update]
	
	if by == None:
		return [create_alert("Please choose the value of \"Select features by\".", "warning"), dash.no_update]
	
	if by == "percentage" and percentage == None:
		return [create_alert("Please enter the retention rate.", "warning"), dash.no_update]
	
	if problem_type == None:
		return [create_alert("Please choose the problem type.", "warning"), dash.no_update]
	
	if model_name_for_training == None:
		return [create_alert("Please choose the model for training.", "warning"), dash.no_update]
	
	if explanation == None:
		return [create_alert("Please choose the explanation method.", "warning"), dash.no_update]
	
	if len(metrics_choosed) < 2:
		return [create_alert("Please choose at least two metrics.", "warning"), dash.no_update]
	
	if strategy == None:
		return [create_alert("Please choose the strategy for comparison.", "warning"), dash.no_update]
	
	if strategy == "preference" and len(preference) != len(set(preference)):
		return [create_alert("Please do not select duplicate metrics.", "warning"), dash.no_update]
	
	
	return [create_alert("The results are being calculated.", "info"), "calculating"]


@app.callback(
	Output("submit_button", "disabled"),
	Output("add_user_defined_subset", "disabled"),
	Output("remove_user_defined_subset", "disabled"),
	Output("add_model_param", "disabled"),
	Output("remove_model_param", "disabled"),
	Output("add_tensorflow_model_init_param", "disabled"),
	Output("remove_tensorflow_model_init_param", "disabled"),
	Output("add_tensorflow_model_compile_param", "disabled"),
	Output("remove_tensorflow_model_compile_param", "disabled"),
	Output("dataset", "disabled", allow_duplicate=True),
	Output({"type": "user_defined_subset", "index": ALL}, "disabled"),
	Output("model_for_training", "disabled", allow_duplicate=True),
	Output("problem_type", "disabled", allow_duplicate=True),
	Output("fs_methods_choosed", "disabled", allow_duplicate=True),
	Output("by", "disabled", allow_duplicate=True),
	Output("percentage", "disabled", allow_duplicate=True),
	Output("explanation", "disabled", allow_duplicate=True),
	Output("discretize_continuous", "disabled", allow_duplicate=True),
	Output("num_samples", "disabled", allow_duplicate=True),
	Output("num_background_samples", "disabled", allow_duplicate=True),
	Output("rate", "disabled", allow_duplicate=True),
	Output("coalitional_fvoid", "disabled", allow_duplicate=True),
	Output("method", "disabled", allow_duplicate=True),
	Output("reverse", "disabled", allow_duplicate=True),
	Output("complexity", "disabled", allow_duplicate=True),
	Output("scaler", "disabled", allow_duplicate=True),
	Output("complete_fvoid", "disabled", allow_duplicate=True),
	Output("metrics", "disabled", allow_duplicate=True),
	Output("strategy", "disabled", allow_duplicate=True),
	[Output({"type": "model_param_name", "index": ALL}, "disabled")],
	[Output({"type": "model_param_value", "index": ALL}, "disabled")],
	[Output({"type": "layer_name", "index": ALL}, "disabled")],
	[Output({"type": "units", "index": ALL}, "disabled")],
	[Output({"type": "activation", "index": ALL}, "disabled")],
	[Output({"type": "tensorflow_model_compile_param_name", "index": ALL}, "disabled")],
	[Output({"type": "tensorflow_model_compile_param_value", "index": ALL}, "disabled")],
	Output("epochs", "disabled", allow_duplicate=True),
	[Output({"type": "preference", "index": ALL}, "disabled")],
	Output("dataset", "style"),
	Input("status", "data"),
	prevent_initial_call=True
)
def set_disabled(status):
	ctx = dash.callback_context
	if status in ["calculating", "finished"]:
		all_outputs = []
		for i in range(len(ctx.outputs_list) - 1):
			if not isinstance(ctx.outputs_list[i], list):
				all_outputs.append(True)
			else:
				num_outputs = len(ctx.outputs_list[i])
				all_outputs.append([True] * num_outputs)
		all_outputs.append({"cursor": "default"})
		return all_outputs
	elif status in ["not submitted", "error"]:
		all_outputs = []
		for i in range(len(ctx.outputs_list) - 1):
			if not isinstance(ctx.outputs_list[i], list):
				all_outputs.append(False)
			else:
				num_outputs = len(ctx.outputs_list[i])
				all_outputs.append([False] * num_outputs)
		all_outputs.append({"cursor": "pointer"})
		return all_outputs
	else:
		return dash.no_update


@app.callback(
	Output("submit_button", "children"),
	Output("submit_button", "color"),
	Input("status", "data"),
	prevent_initial_call=True
)
def update_button_style(status):
	if status == "calculating":
		return "Calculating...", "info"
	elif status == "finished":
		return "Success", "success"
	elif status == "error":
		return "Submit", "primary"
	else:
		return dash.no_update

def change_dtype(value:str):
	if bool(re.match(r'^-?\d+$', value)):
		return int(value)
	if bool(re.match(r'^-?\d+\.\d+$', value)):
		return float(value)
	if value.lower() in ["true", "false"]:
		return bool(strtobool(value))
	
	return value

@app.callback(
	Output("form_info", "children", allow_duplicate=True),
	Output("status", "data", allow_duplicate=True),
	Output("strategy_choosed", "value", allow_duplicate=True),
	Output("preference_dropdown_result", "children", allow_duplicate=True),
	Output("metrics_showed", "options", allow_duplicate=True),
	Output("metrics_showed", "value", allow_duplicate=True),
	Output("result_key", "data"),
	Output("download_file", "data", allow_duplicate=True),
	Output("result_part", "style", allow_duplicate=True),
	Output("download_button", "style", allow_duplicate=True),
	Input("result_key", "data"),
	Input("submit_button", "n_clicks"),
	Input("status", "data"),
	State("dataset", "contents"),
	State('dataset', 'filename'),
	State({"type": "user_defined_subset", "index": ALL}, "value"),
	State("model_for_training", "value"),
	State("problem_type", "value"),
	State("fs_methods_choosed", "value"),
	State("by", "value"),
	# State("model_for_fs", "value"),
	State("percentage", "value"),
	State("explanation", "value"),
	State("discretize_continuous", "value"),
	State("num_samples", "value"),
	State("num_background_samples", "value"),
	State("rate", "value"),
	State("coalitional_fvoid", "value"),
	State("method", "value"),
	State("reverse", "value"),
	State("complexity", "value"),
	State("scaler", "value"),
	State("complete_fvoid", "value"),
	State("metrics", "value"),
	State("strategy", "value"),
	[State({"type": "model_param_name", "index": ALL}, "value")],
	[State({"type": "model_param_value", "index": ALL}, "value")],
	[State({"type": "layer_name", "index": ALL}, "value")], # index not run since error in children structure, value in a list with position order
	[State({"type": "units", "index": ALL}, "value")],
	[State({"type": "activation", "index": ALL}, "value")],
	[State({"type": "tensorflow_model_compile_param_name", "index": ALL}, "value")],
	[State({"type": "tensorflow_model_compile_param_value", "index": ALL}, "value")],
	[State("epochs", "value")],
	[State({"type": "preference", "index": ALL}, "value")],
	State("preference_dropdown_result", "children"),
	State("user_consent", "value"),
	prevent_initial_call=True
)
def execute_recommendation_system(result_key, n_clicks, status, contents, filename, user_defined_subsets, model_name_for_training,
                                  problem_type,
                                  fs_methods_choosed, by, percentage,
                                  explanation, discretize_continuous, num_samples, num_background_samples, rate,
                                  coalitional_fvoid, method, reverse, complexity, scaler, complete_fvoid,
                                  metrics_choosed,
                                  strategy_for_comparation, model_param_name, model_param_value, layer_name, units,
                                  activation, tensorflow_model_compile_param_name, tensorflow_model_compile_param_value,
                                  epochs, preference, old_preference_dropdown_result, user_consent):
	if result_key is not None : # pickle uploaded
		redis_connection = redis.StrictRedis(connection_pool=pool)
		data_and_results_pickle = redis_connection.get(result_key)
		data_and_results = pickle.loads(data_and_results_pickle)
		return [
			create_alert(
				"The results are displayed below.",
				"success"),
			"finished",
			strategy_for_comparation,
			old_preference_dropdown_result,
			(create_metrics_options(metrics_choosed)),
			(metrics_choosed[0:3]),
			result_key,
			None, #dcc.send_bytes(buffer.getvalue(), "data_and_results.pkl"),
			{"display": "flex"},
			{"display": "None"}
		]

	elif n_clicks > 0 and status == "calculating":
		ctx = dash.callback_context

		# get input hyperparameters
		all_hyper = locals()
		with Session() as session:
			inspector = s_inspect(engine)
			hyper_cols = [col["name"] for col in inspector.get_columns("hyperparameter_data")]
		hyperparams = {k: all_hyper[k] for k in all_hyper.keys() if k in hyper_cols}

		X, y, _, _ = file_to_df(contents, filename)

		user_defined_feature_subsets = []
		for subset in user_defined_subsets:
			if subset is not None and subset.strip() != "":
				list_str = re.split(r'\s*,\s*', subset)
				list_bool = [bool(strtobool(str_value)) for str_value in list_str]
				mask = np.array(list_bool)
				user_defined_feature_subsets.append(mask)
		
		if model_name_for_training is not None:
			for model_type in models.keys():
				if model_name_for_training in models[model_type].keys():
					if model_type in ["sklearn", "xgboost"]:
						model_class_for_training = models[model_type][model_name_for_training][problem_type]
					else:
						model_class_for_training = models[model_type][model_name_for_training]
		
		if problem_type is not None:
			classfication = True if problem_type == "classification" else False
		
		if fs_methods_choosed is not None:
			if type(fs_methods_choosed) is not list:
				fs_methods_choosed = [fs_methods_choosed.replace(" ", "_")]
			else:
				fs_methods_choosed = [fs_method.replace(" ", "_") for fs_method in fs_methods_choosed]
		else:
			fs_methods_choosed = []
		
		lime_params = {}
		shap_params = {}
		coalitional_method_params = {}
		complete_method_params = {}
		if explanation == "LIME":
			lime_params["discretize_continuous"] = bool(
				strtobool(discretize_continuous)) if discretize_continuous is not None else True
			lime_params["num_samples"] = num_samples if num_samples is not None else 5000
		elif "SHAP" in explanation:
			shap_params["num_background_samples"] = num_background_samples
		elif explanation == "Coalitional Method":
			coalitional_method_params["rate"] = rate if rate is not None else 0.25
			coalitional_method_params["fvoid"] = coalitional_fvoid
			coalitional_method_params["method"] = method if method is not None else "spearman"
			coalitional_method_params["reverse"] = bool(strtobool(reverse)) if reverse is not None else False
			coalitional_method_params["complexity"] = bool(strtobool(complexity)) if complexity is not None else False
			coalitional_method_params["scaler"] = bool(strtobool(scaler)) if scaler is not None else False
		elif explanation == "Complete Method":
			complete_method_params["fvoid"] = complete_fvoid
		
		model_kwargs = {}
		tf_model_init_params = {}
		tf_model_compile_params = {}
		tf_model_fit_params = {}
		if model_name_for_training in models["sklearn"].keys():
			for i in range(len(model_param_name)):
				if (model_param_name[i] is not None and model_param_name[i].strip() != "") and (
						model_param_value[i] is not None and model_param_value[i].strip() != ""):
					model_kwargs[model_param_name[i]] = change_dtype(model_param_value[i])
		elif model_name_for_training in models["tensorflow"].keys():
			for i in range(len(layer_name)):
				if i != len(layer_name) - 1:
					if (layer_name[i] is not None and layer_name[i].strip() != "") and (
							units[i] is not None and units[i].strip() != "") and (
							activation[i] is not None and activation[i].strip() != ""):
						tf_model_init_params[i] = {"layer_class": layers[layer_name[i]],
						                           "params": {"units": int(units[i]), "activation": activation[i]}}
				else:
					if (layer_name[i] is not None and layer_name[i].strip() != "") and (
							units[i] is not None and units[i].strip() != ""):
						tf_model_init_params[i] = {"layer_class": layers[layer_name[i]],
						                           "params": {"units": int(units[i])}}
			
			for i in range(len(tensorflow_model_compile_param_name)):
				if (tensorflow_model_compile_param_name[i] is not None and tensorflow_model_compile_param_name[
					i].strip() != "") and (
						tensorflow_model_compile_param_value[i] is not None and tensorflow_model_compile_param_value[
					i].strip() != ""):
					tf_model_compile_params[tensorflow_model_compile_param_name[i]] = change_dtype(tensorflow_model_compile_param_value[i])
			
			tf_model_fit_params = {'epochs': int(epochs) if epochs is not None else None}
		
		metrics_choosed_replaced = [m.replace(" ", "_") for m in metrics_choosed]
		preference_replaced = [p.replace(" ", "_") for p in preference if p is not None]
		
		if model_name_for_training in models["sklearn"].keys() or model_name_for_training in models["xgboost"].keys():
			model_class_for_fs = model_class_for_training
		else:
			model_class_for_fs = RandomForestClassifier if classfication else RandomForestRegressor
		
		try:
			results = recommend_fs_methods_for_one_model(X=X, y=y, model_class_for_training=model_class_for_training,
			                                             classification=classfication,
			                                             user_defined_feature_subsets=user_defined_feature_subsets,
			                                             fs_methods_choosed=fs_methods_choosed,
			                                             by=by, explanation=explanation,
			                                             metrics_choosed=metrics_choosed_replaced,
			                                             strategy_for_comparation=strategy_for_comparation,
			                                             preference=preference_replaced,
			                                             model_class_for_fs=model_class_for_fs,
			                                             percentage=percentage,
			                                             tf_model_init_params=tf_model_init_params,
			                                             tf_model_compile_params=tf_model_compile_params,
			                                             tf_model_fit_params=tf_model_fit_params,
			                                             model_kwargs=model_kwargs, lime_params=lime_params,
			                                             shap_params=shap_params,
			                                             coalitional_method_params=coalitional_method_params,
			                                             complete_method_params=complete_method_params)

			data_and_results = {
				"data": X,
				"targets": y,
				"results": results,
				"hyperparameters" : hyperparams
			}
			
			buffer = io.BytesIO()
			pickle.dump(data_and_results, buffer)
			buffer.seek(0)
			
			data_and_results_pickle = pickle.dumps(data_and_results)
			redis_connection = redis.StrictRedis(connection_pool=pool)
			result_key = str(uuid.uuid4())
			redis_connection.setex(result_key, 3600, data_and_results_pickle)
			hyperparams["result_key"] = result_key # ["result_key"] not in the pickle file
			
			submit_date = datetime.datetime.now()
			submit_date_str = submit_date.strftime("%d/%m/%Y %H:%M:%S")
			num_rows = X.shape[0]
			num_columns = X.shape[1]
			
			fs_method_str = str(fs_methods_choosed)
			user_defined_feature_subsets_str = str(
				[np.array2string(mask, max_line_width=np.inf, separator=",").replace(" ", "") for mask in
				 user_defined_feature_subsets])
			tf_model_hyperparameters = {"init": tf_model_init_params, "compile": tf_model_compile_params,
			                            "fit": tf_model_fit_params}
			
			if model_name_for_training in models["sklearn"].keys() or model_name_for_training in models[
				"xgboost"].keys():
				model_hyperparameters = str(model_kwargs)
			else:
				model_hyperparameters = str(tf_model_hyperparameters)
			
			if explanation == "LIME":
				explainer_parameters = str(lime_params)
			elif "SHAP" in explanation and "Tree" not in explanation:
				explainer_parameters = str(shap_params)
			elif explanation == "Coalitional Method":
				explainer_parameters = str(coalitional_method_params)
			elif explanation == "Complete Method":
				explainer_parameters = str(complete_method_params)
			else:
				explainer_parameters = None
			
			metrics_str = str(metrics_choosed)
			
			preference_str = str(preference) if strategy_for_comparation == "preference" else None

			if user_consent:
				with Session() as session:
					session.begin()
					try:
						sql = text(
							"INSERT INTO meta_data(result_key, dataset_name, submit_date, num_rows, num_columns, problem_type) "
							"VALUES (:result_key, :dataset_name, :submit_date, :num_rows, :num_columns, :problem_type)")
						session.execute(sql,
										{"result_key": result_key, "dataset_name": filename, "submit_date": submit_date_str,
										 "num_rows": num_rows, "num_columns": num_columns, "problem_type": problem_type})
						columns = ', '.join(hyperparams.keys())
						placeholders = ', '.join(f":{key}" for key in hyperparams.keys())
						sql = f'INSERT INTO hyperparameter_data ({columns}) VALUES ({placeholders})'
						values = {k: v if v is None or isinstance(v, (float, int)) else str(v)
							for k, v in hyperparams.items()}

						session.execute(text(sql), values)

						session.commit()
					except Exception as e:
						session.rollback()
						raise Exception(str(e) + ' Please retry.')

			preference_dropdown_result = get_preference_dropdown(
				order=preference if strategy_for_comparation == "preference" else metrics_choosed,
				type_id="preference_result", is_active=True)

			return [
				create_alert(
					"Process finished successfully! The result file will be downloaded automatically. "
					"If there is a problem, you can click the \"Download result file\" button to download it again. "
					"The results will be displayed below.",
					"success"),
				"finished",
				strategy_for_comparation,
				preference_dropdown_result,
				(create_metrics_options(metrics_choosed)),
				(metrics_choosed[0:3]),
				result_key,
				dcc.send_bytes(buffer.getvalue(), "data_and_results.pkl"),
				{"display": "flex"},
				{"display": "inline-block"}
			]
		except Exception as e:
			return [create_alert(str(e), "danger")] + [dash.no_update for _ in range(len(ctx.outputs_list) - 1)]
	else:
		return dash.no_update


@app.callback(
	Output("download_file", "data", allow_duplicate=True),
	Input("download_button", "n_clicks"),
	State("result_key", "data"),
	prevent_initial_call=True
)
def download_result_file(n_clicks, result_key):
	redis_connection = redis.StrictRedis(connection_pool=pool)
	data_and_results_pickle = redis_connection.get(result_key)
	b64_encoded = base64.b64encode(data_and_results_pickle).decode("utf-8")
	download_data = {
		"filename": "data_and_results.pkl",
		"content": b64_encoded,
		"base64": True,
		"type": "application/octet-stream"
	}
	return download_data


@app.callback(
	Output("preference_display_result", "style"),
	Input("strategy_choosed", "value"),
)
def set_preference_div_display(strategy: str):
	return {'display': 'block'} if strategy == "preference" else {'display': 'none'}


@app.callback(
	Output("result_part_info", "children"),
	Output("result_graph_div", "children"),
	Input("strategy_choosed", "value"),
	Input("metrics_showed", "value"),
	[Input({'type': 'preference_result', 'index': ALL}, "value")],
	Input("result_key", "data"),
	State("metrics", "value"),
	State("user_consent", "value"),
	prevent_initial_call=True
)
def create_result_graph(strategy: str, metrics_showed, preference_dropdowns_values, result_key, metrics, user_consent):
	if result_key is not None:
		ctx = dash.callback_context
		if not ctx.triggered:
			element_id = ""
		else:
			element_id = ctx.triggered[0]["prop_id"].split(".")[0]

		if (element_id in ["strategy_choosed", "metrics_showed", "result_key"]) or ("preference_result" in element_id):
			if len(metrics_showed) not in [2, 3]:
				return create_alert("Please choose two or three metrics.", "warning"), dash.no_update

			time = datetime.datetime.now()
			time_str = time.strftime("%d/%m/%Y %H:%M:%S")

			redis_connection = redis.StrictRedis(connection_pool=pool)
			data_and_results_pickle = redis_connection.get(result_key)
			data_and_results = pickle.loads(data_and_results_pickle)
			data = data_and_results["data"]
			results = data_and_results["results"]
			if strategy != "preference":
				if user_consent:
					with Session() as session:
							session.begin()
							try:
								sql = text(
									"INSERT INTO result_part_choices(result_key, time, strategy, metrics_showed) "
									"VALUES (:result_key, :time, :strategy, :metrics_showed)")
								session.execute(sql,
												{"result_key": result_key, "time": time_str, "strategy": strategy,
												 "metrics_showed": str(metrics_showed)
												 })
								session.commit()
							except:
								session.rollback()
				return None, dcc.Graph(id="result_graph",
								 figure=create_graph(data, results["fs_results"], results["metrics_results"],
													 metrics_showed, strategy),
								 style={"height": "800px", "width": "100%"})
			else:
				preference = []
				if preference_dropdowns_values == []:
					preference_dropdowns_values = metrics
				for preference_value in preference_dropdowns_values:
					if preference_value not in preference:
						preference.append(preference_value)
					else:
						return create_alert("Please do not select duplicate metrics in the preference.", "warning"), dash.no_update

				preference_str = str(preference)
				if user_consent:
					with Session() as session:
						session.begin()
						try:
							sql = text(
								"INSERT INTO result_part_choices(result_key, time, strategy, preference, metrics_showed) "
								"VALUES (:result_key, :time, :strategy, :preference, :metrics_showed)")
							session.execute(sql,
											{"result_key": result_key, "time": time_str, "strategy": strategy,
											 "preference": preference_str, "metrics_showed": str(metrics_showed)})
							session.commit()
						except:
							session.rollback()

				return None, dcc.Graph(id="result_graph",
								 figure=create_graph(data, results["fs_results"], results["metrics_results"],
													 metrics_showed, strategy, preference),
								 style={"height": "830px", "width": "100%"})
		else:
			return dash.no_update
	else:
		return [dash.no_update, dash.no_update]


@app.callback(
	[Output("summary_plot", "children"), Output("table", "children")],
	Input("result_key", "data"),
	Input("result_graph", "clickData"),
	Input("metrics_showed", "value"),
	State("strategy_choosed", "value"),
	[State({'type': 'preference_result', 'index': ALL}, "value")],
	State("user_consent", "value"),
	prevent_initial_call=True
)
def show_information_table_and_summary_plot(result_key, click_data, metrics_choosed, strategy_choosed,
											preference_dropdowns_values, user_consent):
	if click_data is not None:
		redis_connection = redis.StrictRedis(connection_pool=pool)
		data_and_results_pickle = redis_connection.get(result_key)
		data_and_results = pickle.loads(data_and_results_pickle)
		data = data_and_results["data"]
		results = data_and_results["results"]
		customdata_mask = click_data["points"][0]["customdata"][0]
		metrics_df = pd.DataFrame(results["metrics_results"])
		matching_row = metrics_df.loc[customdata_mask, :]
		mask = results["fs_results"]["features_selected"][matching_row.name]["mask"]
		subset = list(data.columns[mask])
		subset_str = str(subset).replace("[", "").replace("]", "").replace("'", "")
		fs_methods_list = []
		for method_type in results["fs_results"]["features_selected"][matching_row.name]["methods"].keys():
			fs_methods_list += results["fs_results"]["features_selected"][matching_row.name]["methods"][
				method_type]
		
		fs_methods = str(fs_methods_list).replace(
			"[", "").replace("]", "").replace("'", "").replace("_", " ")
		table = dbc.Table(children=[
			html.Tr(
				html.Th("Detail", colSpan=2, style={"textAlign": "center"})
			),
			html.Tr([html.Th("Subset", style={"textAlign": "center", "vertical-align": "middle"}),
			         html.Td(subset_str, style={"textAlign": "justify"})]),
			html.Tr([html.Th("FS methods", style={"textAlign": "center"}),
			         html.Td(fs_methods, style={"textAlign": "center", "verticalAlign": "middle"})]),
			html.Tr([html.Th("Scores", style={"textAlign": "center", "verticalAlign": "middle"}), html.Td(
				str([f"{m} = {metrics_df.loc[matching_row.name, m.replace(' ', '_')].round(round_digits)}" for m in
				     metrics_choosed]).replace("[", "").replace("]", "").replace("'", ""),
				style={"textAlign": "center", "verticalAlign": "middle"})]),
			html.Tr([html.Th("Copy mask", style={"textAlign": "center"}),
			         html.Td(children=["Click the button to copy the mask of the subset  ",
			                           dcc.Clipboard(id="clipboard",
			                                         content=np.array2string(mask, separator=",",
			                                                                 max_line_width=np.inf).replace(" ",
			                                                                                                ""),
			                                         style={"backgroundColor": "transparent"})],
			                 style={"textAlign": "center", "verticalAlign": "middle"})])
		
		], className="table", bordered=True, responsive=True,
			style={"borderLeft": "1px #c6c7c8 solid", "borderRight": "1px #c6c7c8 solid",
			       "backgroundColor": "#f8f9fa"})
		
		explanation_object = results["explanation_results"][matching_row.name]
		plt.figure()
		shap.summary_plot(explanation_object, explanation_object.data, show=False)
		fig = plt.gcf()
		fig.axes[-1].set_aspect("auto")
		fig.axes[-1].set_box_aspect(50)
		plt.tight_layout()
		
		buffer = io.BytesIO()
		plt.savefig(buffer, format='png')
		plt.close(fig)
		buffer.seek(0)
		
		img_bytes = buffer.read()
		img_base64 = base64.b64encode(img_bytes).decode('ascii')
		img = html.Img(src=f"data:image/png;base64,{img_base64}", style={"height": "100%", "width": "100%"})
		time = datetime.datetime.now()
		time_str = time.strftime("%d/%m/%Y %H:%M:%S")
		
		if strategy_choosed == "preference":
			preference = []
			for preference_value in preference_dropdowns_values:
				if preference_value not in preference:
					preference.append(preference_value)
				else:
					return dash.no_update
		
		preference_str = str(preference) if strategy_choosed == "preference" else None

		if user_consent:
			with Session() as session:
				session.begin()
				try:
					sql = text(
						"INSERT INTO result_part_choices(result_key, time, strategy, preference, metrics_showed, click_data) "
						"VALUES (:result_key, :time, :strategy, :preference, :metrics_showed, :click_data)")
					session.execute(sql,
									{"result_key": result_key, "time": time_str, "strategy": strategy_choosed,
									 "preference": preference_str, "metrics_showed": str(metrics_choosed), "click_data":customdata_mask})
					session.commit()
				except:
					session.rollback()
		return [img, table]
	return dash.no_update


if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=80, debug=False)

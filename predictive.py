from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import streamlit as st  # For compatibility with session_state
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, r2_score)
from sklearn.model_selection import train_test_split

from data_utils import perform_clustering, train_ml_model

# Optional SHAP import with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def st_shap(plot, height: Optional[int] = None) -> html.Iframe:
    """
    Render SHAP plots in Dash.

    Args:
        plot: SHAP plot object.
        height (Optional[int]): Height of the plot in pixels.

    Returns:
        html.Iframe: Iframe containing the SHAP plot.
    """
    try:
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        return html.Iframe(srcDoc=shap_html, style={'height': f'{height}px', 'width': '100%'})
    except Exception as e:
        return html.Div(f"Error rendering SHAP plot: {str(e)}.")

def render_predictive_page(df: pd.DataFrame) -> html.Div:
    """
    Render the predictive analytics page with ML model training, SHAP/LIME visualizations, fairness metrics, and clustering.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        html.Div: Dash layout for the predictive page.
    """
    # Move the SHAP warning here
    if not SHAP_AVAILABLE:
        warning = html.P("SHAP library not installed. Feature importance visualizations will be unavailable.", style={'color': '#FFD700'})
    else:
        warning = None

    if df is None or df.empty:
        return html.Div("No dataset available. Please upload a dataset on the Upload page.")

    # Update progress
    st.session_state.progress["Predictive"] = "In Progress"

    # Model Training Section
    predictive_layout = [
        warning if warning else html.Div(),
        html.H2("Train a Machine Learning Model"),
        dcc.RadioItems(
            id='task-type',
            options=[
                {'label': 'classification', 'value': 'classification'},
                {'label': 'regression', 'value': 'regression'}
            ],
            value='classification',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            inputStyle={'margin-right': '5px'}
        ),
        dcc.Dropdown(
            id='target-col',
            options=[{'label': col, 'value': col} for col in df.columns],
            placeholder="Select the column to predict",
            value=None
        ),
        dcc.Dropdown(
            id='feature-cols',
            options=[{'label': col, 'value': col} for col in df.columns],
            multi=True,
            placeholder="Select columns to use as predictors"
        ),
        dcc.Dropdown(
            id='model-type',
            options=[
                {'label': 'RandomForest', 'value': 'RandomForest'},
                {'label': 'XGBoost', 'value': 'XGBoost'},
                {'label': 'LightGBM', 'value': 'LightGBM'}
            ],
            value='RandomForest',
            placeholder="Choose the ML model to train"
        ),
        html.Button('Train Model', id='train-model-button'),
        html.Div(id='model-training-output'),
    ]

    # SHAP Visualization Section
    if 'model' in st.session_state and SHAP_AVAILABLE and st.session_state.get('explainer') is not None:
        shap_layout = [
            html.H2("Feature Importance (SHAP)"),
            html.H3("Feature Importance Summary"),
            html.Div(id='shap-summary-plot'),
            html.H3("Individual Prediction Explanation"),
            dcc.Slider(
                id='shap-sample-idx',
                min=0,
                max=len(st.session_state['X_test']) - 1,
                value=0,
                step=1,
                marks={0: '0', len(st.session_state['X_test']) - 1: str(len(st.session_state['X_test']) - 1)}
            ),
            html.Div(id='shap-force-plot')
        ]
        predictive_layout.extend(shap_layout)

    # LIME Visualization Section
    if 'model' in st.session_state:
        lime_layout = [
            html.H2("Local Model Interpretability (LIME)"),
            dcc.Slider(
                id='lime-sample-idx',
                min=0,
                max=len(st.session_state['X_test']) - 1,
                value=0,
                step=1,
                marks={0: '0', len(st.session_state['X_test']) - 1: str(len(st.session_state['X_test']) - 1)},
                persistence=True
            ),
            html.H3("LIME Explanation"),
            html.Div(id='lime-explanation-plot')
        ]
        predictive_layout.extend(lime_layout)

    # Clustering Section
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    clustering_layout = [
        html.H2("Clustering Results"),
        dcc.Dropdown(
            id='cluster-cols',
            options=[{'label': col, 'value': col} for col in numeric_cols],
            multi=True,
            placeholder="Select at least two numerical columns for clustering"
        ),
        dcc.Slider(
            id='n-clusters',
            min=2,
            max=10,
            value=3,
            step=1,
            marks={2: '2', 10: '10'},
            persistence=True
        ),
        html.Button('Run Clustering', id='run-clustering-button'),
        html.Div(id='clustering-output')
    ]
    predictive_layout.extend(clustering_layout)

    # Note: Actual visualizations and outputs are handled by callbacks in the main app
    st.session_state.progress["Predictive"] = "Done"

    return html.Div(predictive_layout)

# Note: Dash callbacks should be defined in the main app (app_dash.py) or a separate callbacks.py
# to handle interactions like training models, generating SHAP/LIME plots, and clustering.
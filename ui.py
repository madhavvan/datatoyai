import base64
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import pyarrow.parquet as pq
from dash import Input, Output, State, html, dcc, no_update
from dash.exceptions import PreventUpdate

from data_utils import (
    analyze_time_series, apply_cleaning_operations, calculate_health_score,
    detect_anomalies, extract_column, forecast_time_series,
    generate_synthetic_data, get_cleaning_suggestions, get_insights,
    perform_clustering, suggest_workflow, train_ml_model
)
from predictive import render_predictive_page as render_predictive_page_external

# Set up logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(
        'ui.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Cache expensive operations
def get_cached_suggestions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    cache_key = hash(df.to_string())
    if not hasattr(get_cached_suggestions, 'cache'):
        get_cached_suggestions.cache = {}
    if cache_key not in get_cached_suggestions.cache:
        get_cached_suggestions.cache[cache_key] = get_cleaning_suggestions(df)
    return get_cached_suggestions.cache[cache_key]

def get_download_link(df: pd.DataFrame, filename: str) -> html.A:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return html.A(
        f"Download {filename}",
        href=f"data:file/csv;base64,{b64}",
        download=filename,
        style={'display': 'block', 'margin': '10px 0'}
    )

def profile_dataset(df: pd.DataFrame) -> Dict[str, any]:
    profile = {}
    for col in df.columns:
        col_profile = {}
        col_types = df[col].apply(type).nunique()
        col_profile['mixed_types'] = col_types > 1
        col_profile['type_suggestion'] = f"Convert {col} to {df[col].dtype.name}" if col_types > 1 else None
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            formats = df[col].dropna().apply(lambda x: x.strftime('%Y-%m-%d')).nunique()
            col_profile['inconsistent_formats'] = formats > 1
            col_profile['format_suggestion'] = "Standardize date format to YYYY-MM-DD" if formats > 1 else None
        missing_percentage = df[col].isna().mean() * 100
        col_profile['missing_percentage'] = missing_percentage
        col_profile['missing_suggestion'] = f"Consider filling or dropping {col} (missing {missing_percentage:.2f}%)" if missing_percentage > 10 else None
        profile[col] = col_profile
    return profile

def initialize_session_state(session: dict) -> dict:
    defaults = {
        'df': None,
        'cleaned_df': None,
        'logs': [],
        'suggestions': [],
        'previous_states': [],
        'redo_states': [],
        'chat_history': [],
        'cleaning_history': [],
        'cleaning_templates': {},
        'is_premium': False,
        'ai_suggestions_used': 0,
        'dropped_columns': [],
        'progress': {
            "Upload": "Not Started",
            "Clean": "Not Started",
            "Insights": "Not Started",
            "Visualize": "Not Started",
            "Predictive": "Not Started",
            "Share": "Not Started"
        },
        'cleaned_view_option': "First 10 Rows"
    }
    for key, value in defaults.items():
        if key not in session:
            session[key] = value
    return session

def display_cleaned_dataset(cleaned_df: pd.DataFrame, original_shape: tuple) -> html.Div:
    if cleaned_df is None or cleaned_df.empty:
        return html.Div("No cleaned dataset available to display.", className="text-warning")
    try:
        content = [
            html.H3("Cleaned Dataset"),
            html.P(f"Dataset size: {cleaned_df.shape}"),
            html.Div(
                "Dataset has more than 1000 rows. Displaying first 1000 rows." if len(cleaned_df) > 1000 else "",
                className="text-warning"
            ),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in cleaned_df.columns])
                ),
                html.Tbody([
                    html.Tr([html.Td(cleaned_df.iloc[i][col]) for col in cleaned_df.columns])
                    for i in range(min(len(cleaned_df), 1000))
                ])
            ], className="table table-striped"),
            html.H3("Cleaning Summary"),
            html.P(f"Original Shape: {original_shape}"),
            html.P(f"New Shape: {cleaned_df.shape}"),
            html.P(f"New Health Score: {calculate_health_score(cleaned_df)}/100"),
            html.Ul([html.Li(log) for log in cleaned_df.get('logs', [])]),
            get_download_link(cleaned_df, f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        ]
        return html.Div(content)
    except Exception as e:
        return html.Div(f"Error displaying dataset: {str(e)}", className="text-danger")

def render_upload_page():
    return html.Div([
        html.H1("Upload Your Dataset"),
        html.P("Start your data journey here!", className="welcome"),
        dcc.Upload(
            id="file_uploader",
            children=html.Button("Choose a file (CSV, Excel, JSON, or Parquet)", className="btn btn-primary"),
            accept=".csv,.xlsx,.json,.parquet",
            max_size=50 * 1024 * 1024  # 50MB
        ),
        html.Div(id="upload_output")
    ])

def render_clean_page():
    return html.Div([
        html.H1("Clean Your Dataset"),
        html.Div(id="clean_warning", className="text-warning"),
        dcc.Store(id="clean_form_store"),
        html.H3("Your Progress", id="progress_title"),
        html.Div(id="progress_text"),
        html.H3("Dataset Health"),
        dcc.Graph(id="health_score_bar", style={'width': '100%'}),
        html.P(id="health_score_text"),
        html.H3("Smart Workflow Automation"),
        html.P(html.Span("ℹ️", title="Run an AI-suggested cleaning workflow automatically")),
        dbc.Button("Run Smart Workflow", id="run_smart_workflow_button", color="primary", className="mb-3"),
        html.Div(id="smart_workflow_output"),
        dbc.Form([
            html.H3("Manual Column Dropping"),
            html.P(html.Span("ℹ️", title="Select columns to remove from the dataset")),
            dcc.Dropdown(
                id="columns_to_drop",
                multi=True,
                placeholder="Select columns to drop",
                className="mb-3"
            ),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.H4("Define custom cleaning rules"),
                    html.P(html.Span("ℹ️", title="Create rules like 'if column X > 100, set to NaN'")),
                    dcc.Input(
                        id="num_rules",
                        type="number",
                        min=0,
                        max=10,
                        value=0,
                        placeholder="Number of Custom Rules",
                        className="form-control mb-3"
                    ),
                    html.Div(id="custom_rules_container")
                ], title="Custom Cleaning Rules")
            ], start_collapsed=True, className="mb-3"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.H4("Replace unwanted values"),
                    html.P(html.Span("ℹ️", title="Replace specific values across selected columns")),
                    dcc.Input(
                        id="replace_value",
                        placeholder="Value to Replace",
                        type="text",
                        className="form-control mb-3"
                    ),
                    dcc.RadioItems(
                        id="replace_with",
                        options=[
                            {"label": "NaN", "value": "NaN"},
                            {"label": "?", "value": "?"},
                            {"label": "0", "value": "0"},
                            {"label": "Custom", "value": "Custom"}
                        ],
                        value="NaN",
                        className="mb-3"
                    ),
                    dcc.Input(
                        id="replace_with_custom",
                        placeholder="Custom replacement value",
                        type="text",
                        className="form-control mb-3",
                        style={'display': 'none'}
                    ),
                    dcc.RadioItems(
                        id="replace_scope",
                        options=[
                            {"label": "All columns", "value": "All columns"},
                            {"label": "Numeric columns", "value": "Numeric columns"},
                            {"label": "Categorical columns", "value": "Categorical columns"}
                        ],
                        value="All columns",
                        className="mb-3"
                    )
                ], title="Custom Value Replacement")
            ], start_collapsed=True, className="mb-3"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P(html.Span("ℹ️", title="Convert categorical columns to numerical values")),
                    dcc.Dropdown(
                        id="encode_cols",
                        multi=True,
                        placeholder="Select categorical columns to convert",
                        className="mb-3"
                    ),
                    dcc.RadioItems(
                        id="encode_method",
                        options=[
                            {"label": "Label Encoding", "value": "Label Encoding"},
                            {"label": "One-Hot Encoding", "value": "One-Hot Encoding"}
                        ],
                        value="Label Encoding",
                        className="mb-3"
                    )
                ], title="Convert Categorical to Numerical")
            ], start_collapsed=True, className="mb-3"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P(html.Span("ℹ️", title="Enrich data with external info (e.g., geolocation)")),
                    dcc.Dropdown(
                        id="enrich_col",
                        placeholder="Column to Enrich",
                        className="mb-3"
                    ),
                    dcc.Input(
                        id="enrich_api_key",
                        placeholder="Google API Key (for geolocation)",
                        type="password",
                        className="form-control mb-3"
                    )
                ], title="Smart Data Enrichment")
            ], start_collapsed=True, className="mb-3"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P(html.Span("ℹ️", title="AI-driven suggestions to automate data cleaning")),
                    html.Div(id="ai_suggestions_container")
                ], title="AI Cleaning Suggestions")
            ], start_collapsed=False, className="mb-3"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P(html.Span("ℹ️", title="Detect outliers in numerical columns")),
                    dcc.Dropdown(
                        id="anomaly_cols",
                        multi=True,
                        placeholder="Select numerical columns for anomaly detection",
                        className="mb-3"
                    ),
                    dcc.Slider(
                        id="contamination",
                        min=0.01,
                        max=0.5,
                        step=0.01,
                        value=0.1,
                        marks={0.01: '0.01', 0.5: '0.5'},
                        className="mb-3"
                    ),
                    html.Div(id="anomaly_output")
                ], title="Anomaly Detection")
            ], start_collapsed=True, className="mb-3"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P(html.Span("ℹ️", title="Train a machine learning model and deploy it")),
                    dcc.Dropdown(
                        id="target_col_ml",
                        placeholder="Target Column (to predict)",
                        className="mb-3"
                    ),
                    dcc.Dropdown(
                        id="feature_cols_ml",
                        multi=True,
                        placeholder="Feature Columns",
                        className="mb-3"
                    ),
                    dcc.Checklist(
                        id="train_ml",
                        options=[{"label": "Train and Deploy ML Model", "value": "train"}],
                        value=[],
                        className="mb-3"
                    )
                ], title="One-Click ML Deployment")
            ], start_collapsed=True, className="mb-3"),
            html.Div([
                dbc.Button("Preview Changes", id="preview_button", color="info", className="me-2"),
                dbc.Button("Apply Changes", id="apply_button", color="primary", className="me-2"),
                dbc.Button("Auto-Clean", id="auto_clean_button", color="success")
            ], className="d-flex justify-content-between")
        ], id="cleaning_form"),
        dbc.Accordion([
            dbc.AccordionItem([
                html.H3("Save/Apply Cleaning Templates"),
                html.P(html.Span("ℹ️", title="Save your cleaning configuration as a template to reuse later")),
                dbc.Form([
                    dcc.Input(
                        id="template_name",
                        placeholder="Template Name",
                        type="text",
                        className="form-control mb-3"
                    ),
                    dbc.Button("Save as Template", id="save_template_button", color="primary", className="mb-3")
                ]),
                html.Div(id="save_template_output"),
                dbc.Form([
                    dcc.Dropdown(
                        id="apply_template",
                        placeholder="Apply Saved Template",
                        className="mb-3"
                    ),
                    dbc.Button("Apply Template", id="apply_template_button", color="primary", className="mb-3")
                ]),
                html.Div(id="apply_template_output")
            ], title="Save/Apply Cleaning Templates")
        ], start_collapsed=True, className="mb-3"),
        html.Div([
            dbc.Button("Undo Last Cleaning", id="undo_button", color="warning", className="me-2", disabled=True),
            dbc.Button("Redo Last Cleaning", id="redo_button", color="warning", disabled=True)
        ], className="d-flex justify-content-between mb-3"),
        dbc.Accordion([
            dbc.AccordionItem([
                html.H3("Cleaning History"),
                html.Div(id="cleaning_history")
            ], title="Cleaning History")
        ], start_collapsed=True, className="mb-3"),
        dbc.Accordion([
            dbc.AccordionItem([
                html.H3("Export to Tableau"),
                html.P(html.Span("ℹ️", title="Export your cleaned dataset as a CSV file for use in Tableau")),
                dbc.Button("Export Cleaned Dataset for Tableau", id="export_tableau_button", color="primary", className="mb-3"),
                html.Div(id="export_tableau_output")
            ], title="Export to Tableau")
        ], start_collapsed=True)
    ])

def render_insights_page():
    return html.Div([
        html.H1("Insights Dashboard"),
        html.Div(id="insights_warning", className="text-warning"),
        html.H3("Key Insights"),
        html.Div(id="insights_output")
    ])

def render_predictive_page():
    return render_predictive_page_external()
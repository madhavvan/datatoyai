import base64
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyarrow.parquet as pq  # For Parquet file support
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import streamlit as st  # For compatibility with session_state in data_utils
from data_utils import (
    analyze_time_series, apply_cleaning_operations, calculate_health_score,
    chat_with_gpt, detect_anomalies, extract_column, forecast_time_series,
    generate_synthetic_data, get_cleaning_suggestions, get_insights,
    perform_clustering, suggest_workflow, train_ml_model
)
from predictive import render_predictive_page as render_predictive_page_external

# Set up logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to INFO for production
if not logger.handlers:  # Avoid adding handlers multiple times
    handler = RotatingFileHandler(
        'ui.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3)  # 5MB per file, keep 3 backups
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Cache expensive operations
def get_cached_suggestions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    return get_cleaning_suggestions(df)

def get_download_link(df: pd.DataFrame, filename: str) -> html.A:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return html.A(f"Download {filename}", href=f"data:file/csv;base64,{b64}", download=filename)

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

def initialize_session_state() -> Dict:
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
        'cleaned_view_option': "First 10 Rows"  # New: Persist view option
    }
    return defaults

def display_cleaned_dataset(cleaned_df: pd.DataFrame) -> html.Div:
    if cleaned_df is None or cleaned_df.empty:
        return html.Div("No cleaned dataset available to display.")
    try:
        content = [
            html.H2("Cleaned Dataset"),
            html.P(f"Dataset size: {cleaned_df.shape}")
        ]
        if len(cleaned_df) > 1000:
            content.append(html.P(f"Dataset has {len(cleaned_df)} rows. Displaying first 1000 rows to avoid performance issues."))
            content.append(dash_table.DataTable(
                data=cleaned_df.head(1000).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in cleaned_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ))
        else:
            content.append(dash_table.DataTable(
                data=cleaned_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in cleaned_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ))
        content.extend([
            html.H2("Cleaning Summary"),
            html.P(f"Original Shape: {st.session_state.df.shape}"),
            html.P(f"New Shape: {cleaned_df.shape}"),
            html.P(f"New Health Score: {calculate_health_score(cleaned_df)}/100"),
            html.Ul([html.Li(log) for log in st.session_state.logs]),
            get_download_link(cleaned_df, f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        ])
        return html.Div(content)
    except Exception as e:
        return html.Div(f"Error displaying dataset: {str(e)}")

def render_upload_page() -> html.Div:
    st.title("Upload Your Dataset")
    st.markdown("<p class='welcome'>Start your data journey here!</p>", unsafe_allow_html=True)

    initialize_session_state()
    st.session_state.progress["Upload"] = "In Progress"

    upload_layout = [
        html.H1("Upload Your Dataset"),
        html.P("Start your data journey here!", className='welcome'),
        dcc.Upload(
            id='file-uploader',
            children=html.Button('Choose a file (CSV, Excel, JSON, or Parquet)'),
            multiple=False,
            accept='.csv,.xlsx,.json,.parquet'
        ),
        html.Div(id='upload-output')
    ]

    if st.session_state.df is not None:
        score = calculate_health_score(st.session_state.df)
        upload_layout.extend([
            html.H2("Original Dataset Preview (First 10 Rows)"),
            dash_table.DataTable(
                data=st.session_state.df.head(10).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in st.session_state.df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ),
            html.H2("Basic Metadata"),
            html.P(f"Rows: {st.session_state.df.shape[0]}"),
            html.P(f"Columns: {st.session_state.df.shape[1]}"),
            html.P(f"Missing Values: {st.session_state.df.isna().sum().sum()}"),
            dcc.Graph(
                figure={
                    'data': [{'x': [score], 'y': ['Health Score'], 'type': 'bar', 'marker': {'color': '#1E90FF'}}],
                    'layout': {'yaxis': {'range': [0, 100]}}
                }
            ),
            html.P(f"Dataset Health Score: {score}/100"),
            html.P("This is the original dataset. Cleaning operations are applied to a working copy.", style={'color': '#1E90FF'}),
            html.P("Uploading a new file will overwrite the current dataset and reset all cleaning operations. Proceed with caution!", style={'color': '#FFD700'})
        ])

    return html.Div(upload_layout)

def render_clean_page() -> html.Div:
    st.title("Clean Your Dataset")
    if 'df' not in st.session_state or st.session_state.df is None:
        return html.Div("Please upload a dataset first on the Upload page.")

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    available_columns = [col for col in df.columns if col not in st.session_state.dropped_columns]

    if not available_columns:
        return html.Div("No columns available for cleaning. Please upload a new dataset.")

    st.session_state.progress["Clean"] = "In Progress"

    progress_text = "".join(
        f"{emoji} {step}: {status}\n"
        for step, status in st.session_state.progress.items()
        for emoji in ["✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"]
    )

    if not st.session_state.suggestions or id(st.session_state.cleaned_df) != id(df):
        st.session_state.suggestions = get_cached_suggestions(df[available_columns])

    score = calculate_health_score(df)
    clean_layout = [
        html.H1("Clean Your Dataset"),
        html.H2("Your Progress"),
        html.P(progress_text),
        html.H2("Dataset Health"),
        dcc.Graph(
            figure={
                'data': [{'x': [score], 'y': ['Health Score'], 'type': 'bar', 'marker': {'color': '#1E90FF'}}],
                'layout': {'yaxis': {'range': [0, 100]}}
            }
        ),
        html.P(f"Current Health Score: {score}/100"),
        html.H2("Smart Workflow Automation"),
        html.P("ℹ️ Run an AI-suggested cleaning workflow automatically", style={'font-size': '14px'}),
        html.Button("Run Smart Workflow", id="run-smart-workflow-button"),
        html.Div(id='smart-workflow-output'),
        html.H2("Manual Column Dropping"),
        html.P("ℹ️ Select columns to remove from the dataset", style={'font-size': '14px'}),
        dcc.Dropdown(
            id='columns-to-drop',
            options=[{'label': col, 'value': col} for col in available_columns],
            multi=True,
            placeholder="Select columns to drop"
        ),
        html.H2("Custom Cleaning Rules"),
        html.P("ℹ️ Create rules like 'if column X > 100, set to NaN'", style={'font-size': '14px'}),
        dcc.Input(id='num-rules', type='number', value=0, min=0, max=10, step=1),
        html.Div(id='custom-rules-container'),
        html.H2("Custom Value Replacement"),
        html.P("ℹ️ Replace specific values across selected columns", style={'font-size': '14px'}),
        dcc.Input(
            id='replace-value',
            placeholder="Enter the value you want to replace (e.g., ?, 999, Unknown)",
            type='text',
            value=""
        ),
        dcc.RadioItems(
            id='replace-with',
            options=[
                {'label': 'NaN', 'value': 'NaN'},
                {'label': '?', 'value': '?'},
                {'label': '0', 'value': '0'},
                {'label': 'Custom', 'value': 'Custom'}
            ],
            value='NaN',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        dcc.Input(
            id='replace-with-custom',
            placeholder="Custom replacement value",
            type='text',
            value=""
        ),
        dcc.RadioItems(
            id='replace-scope',
            options=[
                {'label': 'All columns', 'value': 'All columns'},
                {'label': 'Numeric columns', 'value': 'Numeric columns'},
                {'label': 'Categorical columns', 'value': 'Categorical columns'}
            ],
            value='All columns',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        html.H2("Convert Categorical to Numerical"),
        html.P("ℹ️ Convert categorical columns to numerical values", style={'font-size': '14px'}),
        dcc.Dropdown(
            id='encode-cols',
            options=[{'label': col, 'value': col} for col in df.select_dtypes(include=['object', 'category']).columns if col in available_columns],
            multi=True,
            placeholder="Select categorical columns to convert"
        ),
        dcc.RadioItems(
            id='encode-method',
            options=[
                {'label': 'Label Encoding', 'value': 'Label Encoding'},
                {'label': 'One-Hot Encoding', 'value': 'One-Hot Encoding'}
            ],
            value='Label Encoding',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        html.H2("Smart Data Enrichment"),
        html.P("ℹ️ Enrich data with external info (e.g., geolocation)", style={'font-size': '14px'}),
        dcc.Dropdown(
            id='enrich-col',
            options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in available_columns],
            value='None',
            placeholder="Select a column to enrich"
        ),
        dcc.Input(
            id='enrich-api-key',
            placeholder="Google API Key (for geolocation)",
            type='password',
            value=""
        ),
        html.H2("AI Cleaning Suggestions"),
        html.P("ℹ️ AI-driven suggestions to automate data cleaning", style={'font-size': '14px'}),
        html.Div(id='ai-suggestions-container'),
        html.H2("Anomaly Detection"),
        html.P("ℹ️ Detect outliers in numerical columns", style={'font-size': '14px'}),
        dcc.Dropdown(
            id='anomaly-cols',
            options=[{'label': col, 'value': col} for col in df.select_dtypes(include=['int64', 'float64']).columns if col in available_columns],
            multi=True,
            placeholder="Select numerical columns for anomaly detection"
        ),
        dcc.Slider(
            id='contamination',
            min=0.01,
            max=0.5,
            value=0.1,
            step=0.01,
            marks={0.01: '0.01', 0.5: '0.5'}
        ),
        html.H2("One-Click ML Deployment"),
        html.P("ℹ️ Train a machine learning model and deploy it", style={'font-size': '14px'}),
        dcc.Dropdown(
            id='target-col',
            options=[{'label': col, 'value': col} for col in available_columns],
            placeholder="Select target column"
        ),
        dcc.Dropdown(
            id='feature-cols',
            options=[{'label': col, 'value': col} for col in available_columns],
            multi=True,
            placeholder="Select feature columns"
        ),
        dcc.Checklist(
            id='train-ml',
            options=[{'label': 'Train and Deploy ML Model', 'value': True}],
            value=[]
        ),
        html.Div([
            html.Button('Preview Changes', id='preview-button', style={'margin': '10px'}),
            html.Button('Apply Changes', id='apply-button', style={'margin': '10px'}),
            html.Button('Auto-Clean', id='auto-clean-button', style={'margin': '10px'})
        ], style={'display': 'flex', 'justify-content': 'center'}),
        html.H2("Save/Apply Cleaning Templates"),
        html.P("ℹ️ Save your cleaning configuration as a template to reuse later", style={'font-size': '14px'}),
        dcc.Input(id='template-name', placeholder="Template Name", type='text', value=""),
        html.Button('Save as Template', id='save-template-button'),
        dcc.Dropdown(
            id='apply-template',
            options=[{'label': 'None', 'value': 'None'}] + [{'label': name, 'value': name} for name in st.session_state.cleaning_templates.keys()],
            value='None',
            placeholder="Apply Saved Template"
        ),
        html.Button('Apply Template', id='apply-template-button'),
        html.H2("Cleaning History"),
        html.Div(id='cleaning-history-container'),
        html.H2("Export to Tableau"),
        html.P("ℹ️ Export your cleaned dataset as a CSV file for use in Tableau", style={'font-size': '14px'}),
        html.Button('Export Cleaned Dataset for Tableau', id='export-tableau-button'),
        html.Div(id='cleaning-output')
    ]

    return html.Div(clean_layout)

def render_insights_page() -> html.Div:
    st.title("Insights Dashboard")
    if 'df' not in st.session_state or st.session_state.df is None:
        return html.Div("Please upload a dataset first on the Upload page.")

    st.session_state.progress["Insights"] = "In Progress"

    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    available_columns = [col for col in df.columns if col not in st.session_state.dropped_columns]

    try:
        insights = get_insights(df[available_columns])
        st.session_state.progress["Insights"] = "Done"
        return html.Div([
            html.H1("Insights Dashboard"),
            html.H2("Key Insights"),
            html.Ul([html.Li(insight) for insight in insights])
        ])
    except Exception as e:
        st.session_state.progress["Insights"] = "Failed"
        return html.Div(f"Error generating insights: {str(e)}")

def render_predictive_page(df: pd.DataFrame) -> html.Div:
    st.title("Predictive Analytics")
    if 'df' not in st.session_state or st.session_state.df is None:
        return html.Div("Please upload a dataset first on the Upload page.")

    st.session_state.progress["Predictive"] = "In Progress"

    available_columns = [col for col in df.columns if col not in st.session_state.dropped_columns]
    df = df[available_columns]

    time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    predictive_layout = [
        html.H1("Predictive Analytics"),
        html.H2("Predictive Dashboard"),
        render_predictive_page_external(df),
        html.H2("Generate Synthetic Data"),
        dcc.Dropdown(
            id='task-type',
            options=[
                {'label': 'classification', 'value': 'classification'},
                {'label': 'regression', 'value': 'regression'}
            ],
            value='classification',
            placeholder="Select task type"
        ),
        html.Button('Generate Synthetic Data', id='generate-synthetic-button'),
        html.H2("Time Series Forecasting"),
        html.P("No datetime columns found for time series forecasting." if not time_cols else ""),
        dcc.Dropdown(
            id='forecast-col',
            options=[{'label': col, 'value': col} for col in time_cols],
            placeholder="Select time series column"
        ),
        dcc.Slider(
            id='forecast-periods',
            min=1,
            max=30,
            value=5,
            step=1,
            marks={1: '1', 30: '30'}
        ),
        dcc.Dropdown(
            id='forecast-freq',
            options=[
                {'label': 'Daily', 'value': 'D'},
                {'label': 'Monthly', 'value': 'M'},
                {'label': 'Yearly', 'value': 'Y'}
            ],
            value='D',
            placeholder="Select frequency"
        ),
        html.Button('Forecast', id='forecast-button'),
        html.H2("Time Series Decomposition"),
        html.P("No datetime columns found for time series decomposition." if not time_cols else ""),
        dcc.Dropdown(
            id='decompose-col',
            options=[{'label': col, 'value': col} for col in time_cols],
            placeholder="Select column for decomposition"
        ),
        dcc.Slider(
            id='decompose-period',
            min=1,
            max=30,
            value=12,
            step=1,
            marks={1: '1', 30: '30'}
        ),
        html.Button('Decompose Time Series', id='decompose-button'),
        html.H2("Clustering"),
        dcc.Dropdown(
            id='cluster-cols',
            options=[{'label': col, 'value': col} for col in numeric_cols],
            multi=True,
            placeholder="Select columns for clustering"
        ),
        dcc.Slider(
            id='n-clusters',
            min=2,
            max=10,
            value=3,
            step=1,
            marks={2: '2', 10: '10'}
        ),
        html.Button('Perform Clustering', id='ui-perform-clustering-button'),
        html.Div(id='predictive-output')
    ]

    return html.Div(predictive_layout)

# Note: Dash callbacks will be defined in app_dash.py or a separate callbacks.py
# to handle interactions, as they need to integrate with the main app.
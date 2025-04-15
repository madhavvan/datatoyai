import base64
import io
import logging
import os
import pickle
import uuid
from logging.handlers import RotatingFileHandler
from typing import Optional

import bcrypt
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import psycopg2
import pyarrow.parquet as pq
import requests
from authlib.integrations.requests_client import OAuth2Session
from dash import Input, Output, State, html, dcc, no_update
from dash.exceptions import PreventUpdate
from psycopg2 import sql
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud
import numpy as np
import lime.lime_tabular

from data_utils import (
    AI_AVAILABLE, chat_with_gpt, suggest_workflow, apply_cleaning_operations,
    calculate_health_score, get_cleaning_suggestions, get_insights,
    forecast_time_series, perform_clustering, train_ml_model
)
from ui import render_clean_page, render_insights_page, render_predictive_page, render_upload_page
from visualizations import render_visualization_page
from predictive import SHAP_AVAILABLE

# Initialize Dash app with a modern Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],  # DARKLY for dark theme; switchable to BOOTSTRAP for light
    suppress_callback_exceptions=True,
    title="Data Toy",
    update_title="Loading..."
)

# Set up logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(
        'app.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("AZURE_APP_URL", "https://datatoyai.com")
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]

# Initialize session storage
def initialize_session():
    return {
        'chat_history': [],
        'theme': 'dark',
        'authenticated': False,
        'username': None,
        'page': 'Login',
        'progress': {
            "Upload": "Not Started",
            "Clean": "Not Started",
            "Insights": "Not Started",
            "Visualize": "Not Started",
            "Predictive": "Not Started",
            "Share": "Not Started"
        },
        'user_info': None,
        'session_token': None,
        'oauth_state': None,
        'df': None,
        'cleaned_df': None,
        'filtered_df': None,
        'logs': [],
        'suggestions': [],
        'previous_states': [],
        'redo_states': [],
        'cleaning_history': [],
        'cleaning_templates': {},
        'is_premium': False,
        'ai_suggestions_used': 0,
        'dropped_columns': [],
        'cleaned_view_option': "First 10 Rows",
        'clustering_labels': None,
        'cluster_cols': [],
        'dashboard_charts': [],
        'dashboard_filters': {},
        'model': None,
        'explainer': None,
        'shap_values': None,
        'X_test': None,
        'feature_cols': [],
        'task_type': None
    }

# Database connection
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            sslmode="require"
        )
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to initialize database due to connection failure")
        return
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, email TEXT, name TEXT, password BYTEA, google_id TEXT, profile_picture TEXT)''')
    c.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sessions')")
    table_exists = c.fetchone()[0]
    if table_exists:
        c.execute("SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'sessions' AND column_name = 'session_token')")
        session_token_exists = c.fetchone()[0]
        if not session_token_exists:
            c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
            logger.debug("Added session_token column to sessions table")
    else:
        c.execute('''CREATE TABLE sessions
                     (username TEXT PRIMARY KEY, session_token TEXT, session_data BYTEA)''')
        logger.debug("Created sessions table with session_token column")
    conn.commit()
    conn.close()

init_db()

def restore_session(session_token):
    logger.debug("Starting restore_session")
    if session_token:
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in restore_session")
            return None
        c = conn.cursor()
        try:
            c.execute(
                "SELECT username, session_data FROM sessions WHERE session_token = %s",
                (session_token,)
            )
            result = c.fetchone()
            logger.debug(f"Database query result: {result}")
            if result:
                username, session_data = result
                session_data = pickle.loads(session_data)
                return session_data
            else:
                logger.debug("No session found for the given session token")
        except Exception as e:
            logger.error(f"Error in restore_session: {str(e)}")
        finally:
            conn.close()
    return None

def save_auth_state(session_data, username):
    logger.debug("Starting save_auth_state")
    session_token = session_data.get('session_token') or str(uuid.uuid4())
    session_data['session_token'] = session_token
    session_blob = pickle.dumps(session_data)
    conn = get_db_connection()
    if conn is None:
        logger.debug("Failed to connect to database in save_auth_state")
        return
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO sessions (username, session_token, session_data) VALUES (%s, %s, %s) ON CONFLICT (username) DO UPDATE SET session_token = %s, session_data = %s",
            (username, session_token, session_blob, session_token, session_blob)
        )
        conn.commit()
        logger.info("Session state saved successfully")
    except Exception as e:
        logger.error(f"Error in save_auth_state: {str(e)}")
    finally:
        conn.close()
    return session_token

def add_user(username: str, email: str, name: str, password: str = None, google_id: str = None, profile_picture: str = None):
    hashed_password = None if password is None else bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db_connection()
    if conn is None:
        return False
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, name, password, google_id, profile_picture) VALUES (%s, %s, %s, %s, %s, %s)",
            (username, email, name, hashed_password if hashed_password is None else psycopg2.Binary(hashed_password), google_id, profile_picture)
        )
        conn.commit()
    except psycopg2.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True

def verify_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    if conn is None:
        return False
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0]:
        stored_password = result[0]
        if stored_password is None:
            return False
        if isinstance(stored_password, memoryview):
            stored_password = stored_password.tobytes()
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False

def get_user_by_google_id(google_id: str):
    conn = get_db_connection()
    if conn is None:
        return None
    c = conn.cursor()
    c.execute(
        "SELECT username, email, name, profile_picture FROM users WHERE google_id = %s",
        (google_id,)
    )
    result = c.fetchone()
    conn.close()
    return result

def load_session(username):
    conn = get_db_connection()
    if conn is None:
        return None
    c = conn.cursor()
    c.execute("SELECT session_data FROM sessions WHERE username = %s", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return pickle.loads(result[0])
    return None

# CSS for theming
def get_css(theme: str = "dark"):
    return """
        body {
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
        }
        .dark-theme {
            background: linear-gradient(to bottom right, #1C2526, #2A3B47);
            color: #FFFFFF;
        }
        .light-theme {
            background: linear-gradient(to bottom right, #F0F4F8, #D9E2EC);
            color: #000000;
        }
        .dark-theme .card {
            background: #2A3B47;
            color: #FFFFFF;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .light-theme .card {
            background: #FFFFFF;
            color: #000000;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .dark-theme h1, .dark-theme h2, .dark-theme h3 {
            color: #1E90FF;
        }
        .light-theme h1, .light-theme h2, .light-theme h3 {
            color: #0066CC;
        }
        .google-login-button {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #FFFFFF;
            color: #757575;
            border: 1px solid #DADCE0;
            border-radius: 4px;
            padding: 10px 20px;
            font-size: 16px;
            font-family: 'Roboto', sans-serif;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
            width: 100%;
            box-sizing: border-box;
            margin: 10px auto;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        .google-login-button:hover {
            background-color: #F8FAFC;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .google-login-button img {
            width: 20px;
            height: 20px;
            margin-right: 10px;
        }
        .welcome {
            font-size: 16px;
            color: #1E90FF;
            font-style: italic;
        }
        .tagline {
            font-size: 16px;
            color: #1E90FF;
            font-style: italic;
        }
    """

# Layout components
def render_custom_header(page_title: str):
    return html.Div([
        html.H1(page_title, style={'marginTop': '20px'}),
        html.Hr(style={'border': '1px solid #FFD700'})
    ])

def get_google_auth_url():
    client = OAuth2Session(
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        redirect_uri=GOOGLE_REDIRECT_URI,
        scope=SCOPES)
    auth_url, state = client.create_authorization_url(GOOGLE_AUTH_URL)
    return auth_url, state

def handle_google_callback(code, state, stored_state):
    if state != stored_state:
        return None, "Invalid OAuth state"
    try:
        client = OAuth2Session(
            GOOGLE_CLIENT_ID,
            GOOGLE_CLIENT_SECRET,
            redirect_uri=GOOGLE_REDIRECT_URI,
            state=state)
        token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        user_info = requests.get(
            GOOGLE_USERINFO_URL, headers={'Authorization': f"Bearer {token['access_token']}"}).json()
        if 'error' in user_info:
            return None, f"Google OAuth error: {user_info['error']}"
        return user_info, None
    except Exception as e:
        return None, f"Error during Google OAuth callback: {str(e)}"

def login_layout(theme):
    return html.Div([
        html.Link(href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap", rel="stylesheet"),
        html.Style(get_css(theme)),
        html.Div([
            html.H1("Welcome to Data Toy AI", className="text-center mb-4"),
            dcc.Input(
                id="username_input",
                placeholder="Enter your username",
                type="text",
                className="form-control mb-3",
                style={'width': '100%'}
            ),
            dcc.Input(
                id="password_input",
                placeholder="Enter your password",
                type="password",
                className="form-control mb-3",
                style={'width': '100%'}
            ),
            dbc.Button("Login", id="login_button", color="primary", className="w-100 mb-3"),
            html.A(
                html.Div([
                    html.Img(src="https://developers.google.com/identity/images/g-logo.png", style={'width': '20px', 'height': '20px', 'marginRight': '10px'}),
                    html.Span("Sign in with Google")
                ], className="google-login-button"),
                href="", id="google_login_link", className="w-100 mb-3"
            ),
            dbc.Button("Sign Up", id="signup_button", color="secondary", className="w-100"),
            html.Div(id="login_error", className="text-danger text-center mt-3")
        ], className="card mx-auto mt-5", style={'maxWidth symmetrize your code to make it look cleaner and more readable.'400px'})
    ], className=f"{theme}-theme")

def signup_layout(theme):
    return html.Div([
        html.Link(href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap", rel="stylesheet"),
        html.Style(get_css(theme)),
        html.Div([
            html.H1("Sign Up for Data Toy AI", className="text-center mb-4"),
            dcc.Input(
                id="new_username_input",
                placeholder="Choose a username",
                type="text",
                className="form-control mb-3",
                style={'width': '100%'}
            ),
            dcc.Input(
                id="new_email_input",
                placeholder="Enter your email",
                type="email",
                className="form-control mb-3",
                style={'width': '100%'}
            ),
            dcc.Input(
                id="new_name_input",
                placeholder="Enter your name",
                type="text",
                className="form-control mb-3",
                style={'width': '100%'}
            ),
            dcc.Input(
                id="new_password_input",
                placeholder="Choose a password",
                type="password",
                className="form-control mb-3",
                style={'width': '100%'}
            ),
            dbc.Button("Register", id="register_button", color="primary", className="w-100 mb-3"),
            dbc.Button("Back to Login", id="back_to_login_button", color="secondary", className="w-100"),
            html.Div(id="signup_error", className="text-danger text-center mt-3"),
            html.Div(id="signup_success", className="text-success text-center mt-3")
        ], className="card mx-auto mt-5", style={'maxWidth': '400px'})
    ], className=f"{theme}-theme")

def sidebar_layout(session):
    progress_text = [
        html.P(f"{emoji} {step}: {status}")
        for step, status in session.get('progress', initialize_session()['progress']).items()
        for emoji in ["✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"]
    ]
    chat_history = [
        html.Div([
            html.Strong(f"{message['role'].capitalize()}: "),
            message['content']
        ], className="mb-2")
        for message in session.get('chat_history', [])
    ]
    user_info = session.get('user_info', {})
    return dbc.Offcanvas([
        html.Img(src="/assets/datatoy_logo.png", style={'width': '100%'}, onError=lambda: html.P("**Data Toy** (Logo not found)")),
        html.Hr(),
        html.Div([
            html.Img(src=user_info.get('picture'), style={'width': '100px'}) if user_info.get('picture') else None,
            html.P(f"Welcome, {user_info.get('name', session.get('username'))}")
        ]),
        html.H4("Navigation"),
        html.P("Transform your data with AI magic.", className="tagline"),
        dcc.RadioItems(
            id="sidebar_page",
            options=[
                {"label": page, "value": page}
                for page in ["Upload", "Clean", "Insights", "Visualize", "Predictive", "Share"]
            ],
            value=session.get('page', 'Upload'),
            className="mb-3"
        ),
        html.Hr(),
        html.H5("Theme"),
        dcc.Dropdown(
            id="theme_select",
            options=[
                {"label": "Dark", "value": "dark"},
                {"label": "Light", "value": "light"}
            ],
            value=session.get('theme', 'dark'),
            clearable=False,
            className="mb-3"
        ),
        html.Hr(),
        html.H5("Your Progress"),
        html.Div(progress_text),
        html.Hr() if AI_AVAILABLE else html.Div("⚠️ AI features are disabled.", className="text-danger"),
        html.H5("AI Data Assistant"),
        html.Div(chat_history, className="mb-3"),
        dcc.Input(id="chat_input", placeholder="Ask Data Toy", type="text", className="form-control mb-3"),
        dbc.Button("Send", id="chat_submit", color="primary", className="w-100 mb-3"),
        html.Hr(),
        html.H5("Feedback"),
        html.A("Share your feedback", href="https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog"),
        html.H5("Join Our Community"),
        html.A("Join our Discord", href="https://discord.gg/your-invite-link"),
        html.H5("Upgrade to Premium"),
        html.A("Upgrade Now", href="https://stripe.com/your-checkout-link"),
        html.Hr(),
        html.Div("Running in DEV_MODE: Unlimited AI suggestions enabled.", className="text-info") if os.getenv("DEV_MODE") == "true" else None,
        dbc.Button("Logout", id="logout_button", color="danger", className="w-100")
    ], id="sidebar", is_open=True, title="Data Toy", placement="start")

def main_layout():
    return html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="session_store", data=initialize_session()),
        html.Div(id="sidebar_container"),
        html.Div(id="page_content", className="container-fluid", style={'marginLeft': '250px'}),
        dcc.Download(id="download")
    ])

app.layout = main_layout


@app.callback(
    Output("google_login_link", "href"),
    Output("session_store", "data", allow_duplicate=True),
    Input("url", "pathname"),
    State("session_store", "data"),
    prevent_initial_call=True
)
def update_google_auth_url(pathname, session):
    auth_url, state = get_google_auth_url()
    session['oauth_state'] = state
    return auth_url, session

@app.callback(
    Output("sidebar_container", "children"),
    Output("page_content", "children"),
    Output("session_store", "data", allow_duplicate=True),
    Output("url", "search"),
    Output("viz_config_controls", "children", allow_duplicate=True),
    Output("filter_controls", "children"),
    Output("viz_output", "children"),
    Output("dashboard_output", "children"),
    Output("global_filter_col", "options"),
    Output("target_col", "options"),
    Output("feature_cols", "options"),
    Output("cluster_cols", "options"),
    Output("train_output", "children"),
    Output("shap_output", "children"),
    Output("lime_output", "children"),
    Output("clustering_output", "children"),
    Output("lime_sample_idx", "max"),
    Output("download", "data", allow_duplicate=True),
    Output("columns_to_drop", "options"),
    Output("encode_cols", "options"),
    Output("enrich_col", "options"),
    Output("anomaly_cols", "options"),
    Output("target_col_ml", "options"),
    Output("feature_cols_ml", "options"),
    Output("custom_rules_container", "children"),
    Output("smart_workflow_output", "children"),
    Output("save_template_output", "children"),
    Output("apply_template_output", "children"),
    Output("cleaning_history", "children"),
    Output("export_tableau_output", "children"),
    Output("undo_button", "disabled"),
    Output("redo_button", "disabled"),
    Output("ai_suggestions_container", "children"),
    Output("apply_template", "options"),
    Output("progress_text", "children"),
    Output("health_score_bar", "figure"),
    Output("health_score_text", "children"),
    Output("insights_output", "children"),
    Input("url", "search"),
    Input("login_button", "n_clicks"),
    Input("signup_button", "n_clicks"),
    Input("register_button", "n_clicks"),
    Input("back_to_login_button", "n_clicks"),
    Input("sidebar_page", "value"),
    Input("theme_select", "value"),
    Input("logout_button", "n_clicks"),
    Input("chat_submit", "n_clicks"),
    Input("file_uploader", "contents"),
    Input("run_smart_workflow_button", "n_clicks"),
    Input("submit_viz", "n_clicks"),
    Input("global_filter_col", "value"),
    Input({"type": "remove_chart", "index": dash.dependencies.ALL}, "n_clicks"),
    Input("train_model", "n_clicks"),
    Input("run_clustering", "n_clicks"),
    Input("lime_sample_idx", "value"),
    Input("export_viz", "n_clicks"),
    Input("preview_button", "n_clicks"),
    Input("apply_button", "n_clicks"),
    Input("auto_clean_button", "n_clicks"),
    Input("save_template_button", "n_clicks"),
    Input("apply_template_button", "n_clicks"),
    Input("undo_button", "n_clicks"),
    Input("redo_button", "n_clicks"),
    Input("export_tableau_button", "n_clicks"),
    Input("num_rules", "value"),
    Input("filter_range", "value"),
    Input("filter_date", "start_date"),
    Input("filter_date", "end_date"),
    Input("filter_values", "value"),
    State("username_input", "value"),
    State("password_input", "value"),
    State("new_username_input", "value"),
    State("new_email_input", "value"),
    State("new_name_input", "value"),
    State("new_password_input", "value"),
    State("chat_input", "value"),
    State("file_uploader", "filename"),
    State("viz_type", "value"),
    State("chart_title", "value"),
    State("add_to_dashboard", "value"),
    State("task_type", "value"),
    State("target_col", "value"),
    State("feature_cols", "value"),
    State("model_type", "value"),
    State("cluster_cols", "value"),
    State("n_clusters", "value"),
    State("x_col", "value"),
    State("y_col", "value"),
    State("hue_col", "value"),
    State("z_col", "value"),
    State("size_col", "value"),
    State("color_col", "value"),
    State("lat_col", "value"),
    State("lon_col", "value"),
    State("periods", "value"),
    State("freq", "value"),
    State("geo_col", "value"),
    State("value_col", "value"),
    State("path_cols", "value"),
    State("values_col", "value"),
    State("selected_cols", "value"),
    State("source_col", "value"),
    State("target_col", "value"),
    State("weight_col", "value"),
    State("time_col", "value"),
    State("event_col", "value"),
    State("start_col", "value"),
    State("end_col", "value"),
    State("task_col", "value"),
    State("date_col", "value"),
    State("group_col", "value"),
    State("text_col", "value"),
    State("max_value", "value"),
    State("stages_col", "value"),
    State("measure_col", "value"),
    State("export_format", "value"),
    State("columns_to_drop", "value"),
    State("replace_value", "value"),
    State("replace_with", "value"),
    State("replace_with_custom", "value"),
    State("replace_scope", "value"),
    State("encode_cols", "value"),
    State("encode_method", "value"),
    State("enrich_col", "value"),
    State("enrich_api_key", "value"),
    State("anomaly_cols", "value"),
    State("contamination", "value"),
    State("target_col_ml", "value"),
    State("feature_cols_ml", "value"),
    State("train_ml", "value"),
    State("template_name", "value"),
    State("apply_template", "value"),
    State({"type": "suggestion", "index": dash.dependencies.ALL}, "value"),
    State({"type": "special_chars_opt", "index": dash.dependencies.ALL}, "value"),
    State({"type": "fill_opt", "index": dash.dependencies.ALL}, "value"),
    State({"type": "outlier_opt", "index": dash.dependencies.ALL}, "value"),
    State({"type": "rule_col", "index": dash.dependencies.ALL}, "value"),
    State({"type": "rule_cond", "index": dash.dependencies.ALL}, "value"),
    State({"type": "rule_threshold", "index": dash.dependencies.ALL}, "value"),
    State({"type": "rule_action", "index": dash.dependencies.ALL}, "value"),
    State({"type": "rule_action_value", "index": dash.dependencies.ALL}, "value"),
    State("session_store", "data"),
    prevent_initial_call=True
)
def update_app(
    search, login_clicks, signup_clicks, register_clicks, back_clicks, sidebar_page,
    theme_select, logout_clicks, chat_submit, file_contents, workflow_clicks, submit_viz,
    filter_col, remove_chart_clicks, train_clicks, cluster_clicks, lime_sample_idx,
    export_viz_clicks, preview_clicks, apply_clicks, auto_clean_clicks, save_template_clicks,
    apply_template_clicks, undo_clicks, redo_clicks, export_tableau_clicks, num_rules,
    filter_range, filter_date_start, filter_date_end, filter_values,
    username, password, new_username, new_email, new_name, new_password, chat_input,
    filename, viz_type, chart_title, add_to_dashboard, task_type, target_col, feature_cols,
    model_type, cluster_cols, n_clusters, x_col, y_col, hue_col, z_col, size_col, color_col,
    lat_col, lon_col, periods, freq, geo_col, value_col, path_cols, values_col, selected_cols,
    source_col, target_col, weight_col, time_col, event_col, start_col, end_col, task_col,
    date_col, group_col, text_col, max_value, stages_col, measure_col, export_format,
    columns_to_drop, replace_value, replace_with, replace_with_custom, replace_scope,
    encode_cols, encode_method, enrich_col, enrich_api_key, anomaly_cols, contamination,
    target_col_ml, feature_cols_ml, train_ml, template_name, apply_template,
    suggestion_values, special_chars_opts, fill_opts, outlier_opts, rule_cols, rule_conds,
    rule_thresholds, rule_actions, rule_action_values, session
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    session = session or initialize_session()
    page_titles = {
        "Upload": "Upload Your Dataset",
        "Clean": "Clean Your Dataset",
        "Insights": "Insights Dashboard",
        "Visualize": "Visualize Your Dataset",
        "Predictive": "Predictive Analytics",
        "Share": "Share Your Work"
    }

    # Initialize outputs
    sidebar = None
    content = html.Div()
    config_controls = []
    filter_controls = []
    viz_output = html.Div()
    dashboard_output = html.Div()
    filter_options = [{"label": "None", "value": "None"}]
    target_options = []
    feature_options = []
    cluster_options = []
    train_output = html.Div()
    shap_output = html.Div("SHAP library not installed.", className="text-warning") if not SHAP_AVAILABLE else html.Div()
    lime_output = html.Div()
    clustering_output = html.Div()
    lime_max = 0
    download_data = None
    columns_drop_options = []
    encode_cols_options = []
    enrich_col_options = [{"label": "None", "value": "None"}]
    anomaly_cols_options = []
    target_col_ml_options = []
    feature_cols_ml_options = []
    custom_rules_container = []
    smart_workflow_output = html.Div()
    save_template_output = html.Div()
    apply_template_output = html.Div()
    cleaning_history = html.Div()
    export_tableau_output = html.Div()
    undo_disabled = True
    redo_disabled = True
    ai_suggestions_container = html.Div()
    template_options = [{"label": "None", "value": "None"}] + [
        {"label": name, "value": name} for name in session.get('cleaning_templates', {}).keys()
    ]
    progress_text = []
    health_score_figure = {'data': [], 'layout': {}}
    health_score_text = ""
    insights_output = html.Div()

    df = session.get('filtered_df') or session.get('cleaned_df') or session.get('df')
    if df is not None:
        target_options = [{"label": col, "value": col} for col in df.columns]
        feature_options = [{"label": col, "value": col} for col in df.columns]
        cluster_options = [{"label": col, "value": col} for col in df.select_dtypes(include=['int64', 'float64']).columns]
        columns_drop_options = [{"label": col, "value": col} for col in df.columns if col not in session.get('dropped_columns', [])]
        encode_cols_options = [{"label": col, "value": col} for col in df.select_dtypes(include=['object', 'category']).columns]
        enrich_col_options.extend([{"label": col, "value": col} for col in df.columns])
        anomaly_cols_options = [{"label": col, "value": col} for col in df.select_dtypes(include=['int64', 'float64']).columns]
        target_col_ml_options = [{"label": col, "value": col} for col in df.columns]
        feature_cols_ml_options = [{"label": col, "value": col} for col in df.columns]
        if session.get('X_test') is not None:
            lime_max = len(session['X_test']) - 1
        if session.get('previous_states'):
            undo_disabled = False
        if session.get('redo_states'):
            redo_disabled = False
        progress_text = [
            html.P(f"{emoji} {step}: {status}")
            for step, status in session['progress'].items()
            for emoji in ["✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"]
        ]
        health_score = calculate_health_score(df)
        health_score_figure = {
            'data': [{'x': [0, health_score], 'y': [''], 'type': 'bar', 'orientation': 'h'}],
            'layout': {'xaxis': {'range': [0, 100]}}
        }
        health_score_text = f"Current Health Score: {health_score}/100"

    # Handle Google OAuth callback
    if search and 'code' in search:
        code = search.split('code=')[1].split('&')[0]
        state = search.split('state=')[1].split('&')[0] if 'state=' in search else None
        user_info, error = handle_google_callback(code, state, session.get('oauth_state'))
        if user_info:
            google_id = user_info['sub']
            user = get_user_by_google_id(google_id)
            if user:
                username, email, name, profile_picture = user
            else:
                username = user_info['email'].split('@')[0]
                email = user_info['email']
                name = user_info['name']
                profile_picture = user_info.get('picture')
                add_user(username, email, name, google_id=google_id, profile_picture=profile_picture)
            session.update({
                'authenticated': True,
                'username': username,
                'user_info': user_info,
                'page': 'Upload'
            })
            session_token = save_auth_state(session, username)
            loaded_session = load_session(username)
            if loaded_session:
                session.update({k: v for k, v in loaded_session.items() if k not in ['authenticated', 'username', 'user_info', 'session_token', 'page']})
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Upload']),
                    render_upload_page()
                ]),
                session,
                f"?session_token={session_token}",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        else:
            session['page'] = 'Login'
            return (
                None,
                html.Div([login_layout(session['theme']), html.Div(error, className="text-danger text-center")]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )

    # Handle login
    elif trigger_id == "login_button" and login_clicks:
        if verify_user(username, password):
            session.update({
                'authenticated': True,
                'username': username,
                'user_info': None,
                'page': 'Upload'
            })
            session_token = save_auth_state(session, username)
            loaded_session = load_session(username)
            if loaded_session:
                session.update({k: v for k, v in loaded_session.items() if k not in ['authenticated', 'username', 'user_info', 'session_token', 'page']})
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Upload']),
                    render_upload_page()
                ]),
                session,
                f"?session_token={session_token}",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        else:
            session['page'] = 'Login'
            return (
                None,
                html.Div([login_layout(session['theme']), html.Div("Incorrect username or password", className="text-danger text-center")]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )

    # Handle signup
    elif trigger_id == "signup_button" and signup_clicks:
        session['page'] = 'Sign Up'
        return (
            None,
            signup_layout(session['theme']),
            session,
            "",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle register
    elif trigger_id == "register_button" and register_clicks:
        if add_user(new_username, new_email, new_name, new_password):
            session['page'] = 'Login'
            return (
                None,
                html.Div([login_layout(session['theme']), html.Div("Registration successful! Please log in.", className="text-success text-center")]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        else:
            return (
                None,
                html.Div([signup_layout(session['theme']), html.Div("Username already exists.", className="text-danger text-center")]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )

    # Handle back to login
    elif trigger_id == "back_to_login_button" and back_clicks:
        session['page'] = 'Login'
        return (
            None,
            login_layout(session['theme']),
            session,
            "",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle sidebar navigation
    elif trigger_id == "sidebar_page" and sidebar_page:
        session['page'] = sidebar_page
        session_token = save_auth_state(session, session.get('username'))
        content = html.Div([
            render_custom_header(page_titles.get(sidebar_page, "Data Toy"))
        ])
        if sidebar_page == "Upload":
            content.children.append(render_upload_page())
        elif sidebar_page == "Clean":
            if not df:
                content.children.append(html.Div("Please upload a dataset first.", className="text-warning"))
            else:
                content.children.append(render_clean_page())
                progress_text = [
                    html.P(f"{emoji} {step}: {status}")
                    for step, status in session['progress'].items()
                    for emoji in ["✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"]
                ]
                health_score = calculate_health_score(df)
                health_score_figure = {
                    'data': [{'x': [0, health_score], 'y': [''], 'type': 'bar', 'orientation': 'h'}],
                    'layout': {'xaxis': {'range': [0, 100]}}
                }
                health_score_text = f"Current Health Score: {health_score}/100"
                suggestions = get_cached_suggestions(df[[col for col in df.columns if col not in session.get('dropped_columns', [])]])
                ai_suggestions_container = html.Ul([
                    html.Li([
                        dcc.Checklist(
                            id={"type": "suggestion", "index": idx},
                            options=[{"label": sug, "value": sug}],
                            value=[],
                            style={'display': 'inline-block'}
                        ),
                        html.Span(f" - {exp}", style={'marginLeft': '10px'}),
                        html.Div([
                            dcc.RadioItems(
                                id={"type": "special_chars_opt", "index": idx},
                                options=[
                                    {"label": "Drop them", "value": "Drop them"},
                                    {"label": "Replace with underscores", "value": "Replace with underscores"}
                                ],
                                value="Drop them",
                                className="mb-2"
                            ) if "Handle special characters" in sug else
                            dcc.RadioItems(
                                id={"type": "fill_opt", "index": idx},
                                options=[
                                    {"label": "Mean", "value": "mean"},
                                    {"label": "Median", "value": "median"},
                                    {"label": "Mode", "value": "mode"}
                                ],
                                value="mode",
                                className="mb-2"
                            ) if "Fill missing values" in sug and extract_column(sug) in df.columns and df[extract_column(sug)].dtype in ['int64', 'float64'] else
                            dcc.RadioItems(
                                id={"type": "outlier_opt", "index": idx},
                                options=[
                                    {"label": "Remove", "value": "Remove"},
                                    {"label": "Cap at bounds", "value": "Cap at bounds"}
                                ],
                                value="Remove",
                                className="mb-2"
                            ) if "Handle outliers" in sug and extract_column(sug) in df.columns else None
                        ])
                    ])
                    for idx, (sug, exp) in enumerate(suggestions)
                ])
                cleaning_history = html.Ul([
                    html.Li([
                        html.Strong(f"{entry['timestamp']}:"),
                        html.Ul([html.Li(log) for log in entry['logs']])
                    ])
                    for entry in session.get('cleaning_history', [])
                ])
        elif sidebar_page == "Insights":
            if not df:
                content.children.append(html.Div("Please upload a dataset first.", className="text-warning"))
            else:
                session['progress'] = session['progress'] | {'Insights': 'In Progress'}
                content.children.append(render_insights_page())
                insights = get_insights(df[[col for col in df.columns if col not in session.get('dropped_columns', [])]])
                insights_output = html.Ul([html.Li(insight) for insight in insights])
                session['progress'] = session['progress'] | {'Insights': 'Done'}
        elif sidebar_page == "Visualize":
            if not df:
                content.children.append(html.Div("Please upload a dataset first.", className="text-warning"))
            else:
                session['progress'] = session['progress'] | {'Visualize': 'In Progress'}
                content.children.append(render_visualization_page())
                filter_options.extend([{"label": col, "value": col} for col in df.columns])
        elif sidebar_page == "Predictive":
            if not df:
                content.children.append(html.Div("Please upload a dataset first.", className="text-warning"))
            else:
                session['progress'] = session['progress'] | {'Predictive': 'In Progress'}
                content.children.append(render_predictive_page())
        elif sidebar_page == "Share":
            content.children.append(html.Div("Sharing features coming soon!"))
        sidebar = sidebar_layout(session) if session.get('authenticated') else None
        return (
            sidebar,
            content,
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle theme selection
    elif trigger_id == "theme_select" and theme_select:
        session['theme'] = theme_select
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            no_update,
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle logout
    elif trigger_id == "logout_button" and logout_clicks:
        conn = get_db_connection()
        if conn:
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE username = %s", (session.get('username'),))
            conn.commit()
            conn.close()
        session = initialize_session()
        return (
            None,
            login_layout(session['theme']),
            session,
            "",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle chat submit
    elif trigger_id == "chat_submit" and chat_submit and chat_input:
        if not df:
            return (
                sidebar_layout(session),
                html.Div("Please upload a dataset first.", className="text-warning"),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        response = chat_with_gpt(df, chat_input, max_tokens=100, username=session.get('username', 'anonymous'))
        session['chat_history'].append({"role": "user", "content": chat_input})
        session['chat_history'].append({"role": "assistant", "content": response})
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            no_update,
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle file upload
    elif trigger_id == "file_uploader" and file_contents:
        try:
            content_type, content_string = file_contents.split(',')
            decoded = base64.b64decode(content_string)
            if filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif filename.endswith('.json'):
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
            elif filename.endswith('.parquet'):
                df = pq.read_table(io.BytesIO(decoded)).to_pandas()
            else:
                df = pd.read_excel(io.BytesIO(decoded))
            if df.empty:
                return (
                    sidebar_layout(session),
                    html.Div([
                        render_custom_header(page_titles['Upload']),
                        render_upload_page(),
                        html.Div("Uploaded dataset is empty.", className="text-danger")
                    ]),
                    session,
                    "",
                    config_controls,
                    filter_controls,
                    viz_output,
                    dashboard_output,
                    filter_options,
                    target_options,
                    feature_options,
                    cluster_options,
                    train_output,
                    shap_output,
                    lime_output,
                    clustering_output,
                    lime_max,
                    download_data,
                    columns_drop_options,
                    encode_cols_options,
                    enrich_col_options,
                    anomaly_cols_options,
                    target_col_ml_options,
                    feature_cols_ml_options,
                    custom_rules_container,
                    smart_workflow_output,
                    save_template_output,
                    apply_template_output,
                    cleaning_history,
                    export_tableau_output,
                    undo_disabled,
                    redo_disabled,
                    ai_suggestions_container,
                    template_options,
                    progress_text,
                    health_score_figure,
                    health_score_text,
                    insights_output
                )
            from ui import profile_dataset
            profile = profile_dataset(df)
            session.update({
                'df': df,
                'filtered_df': df,
                'cleaned_df': None,
                'logs': [],
                'suggestions': [],
                'previous_states': [],
                'redo_states': [],
                'chat_history': [],
                'cleaning_history': [],
                'cleaning_templates': {},
                'ai_suggestions_used': 0,
                'dropped_columns': [],
                'clustering_labels': None,
                'cluster_cols': [],
                'dashboard_charts': [],
                'dashboard_filters': {},
                'model': None,
                'explainer': None,
                'shap_values': None,
                'X_test': None,
                'feature_cols': [],
                'task_type': None,
                'progress': session['progress'] | {'Upload': 'Done'}
            })
            session_token = save_auth_state(session, session.get('username'))
            table_rows = [
                html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
                for i in range(min(len(df), 10))
            ]
            filter_options.extend([{"label": col, "value": col} for col in df.columns])
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Upload']),
                    render_upload_page(),
                    html.H3("Original Dataset Preview (First 10 Rows)"),
                    html.Table([
                        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                        html.Tbody(table_rows)
                    ], className="table table-striped"),
                    html.H3("Basic Metadata"),
                    html.P(f"Rows: {df.shape[0]}"),
                    html.P(f"Columns: {df.shape[1]}"),
                    html.P(f"Missing Values: {df.isna().sum().sum()}"),
                    dcc.Graph(
                        figure={
                            'data': [{'x': [0, calculate_health_score(df)], 'y': [''], 'type': 'bar', 'orientation': 'h'}],
                            'layout': {'xaxis': {'range': [0, 100]}}
                        },
                        style={'height': '50px'}
                    ),
                    html.P(f"Dataset Health Score: {calculate_health_score(df)}/100"),
                    html.P("This is the original dataset. Cleaning operations are applied to a working copy."),
                    html.P("Uploading a new file will overwrite the current dataset and reset all cleaning operations.", className="text-warning"),
                    html.H3("Dataset Profile"),
                    html.Ul([
                        html.Li([
                            html.Strong(col),
                            html.Ul([
                                html.Li(f"Mixed Types: {info['mixed_types']} - {info['type_suggestion']}") if info['mixed_types'] else None,
                                html.Li(f"Inconsistent Formats: {info.get('inconsistent_formats')} - {info.get('format_suggestion')}") if info.get('inconsistent_formats') else None,
                                html.Li(f"Missing: {info['missing_percentage']:.2f}% - {info['missing_suggestion']}") if info['missing_percentage'] > 10 else None
                            ])
                        ])
                        for col, info in profile.items() if any(v for v in info.values() if v)
                    ])
                ]),
                session,
                f"?session_token={session_token}",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        except Exception as e:
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Upload']),
                    render_upload_page(),
                    html.Div(f"Error loading file: {str(e)}", className="text-danger")
                ]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )

    # Handle Smart Workflow
    elif trigger_id == "run_smart_workflow_button" and workflow_clicks:
        if not df:
            smart_workflow_output = html.Div("Please upload a dataset first.", className="text-warning")
        else:
            try:
                workflow = suggest_workflow(df[[col for col in df.columns if col not in session.get('dropped_columns', [])]])
                cleaned_df, logs = apply_cleaning_operations(
                    df, [], [], {}, "", "", "", [], "", auto_clean=True
                )
                session.update({
                    'cleaned_df': cleaned_df,
                    'logs': logs,
                    'previous_states': session.get('previous_states', []) + [(df.copy(), session.get('logs', []).copy())],
                    'cleaning_history': session.get('cleaning_history', []) + [{
                        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": logs + ["Executed Smart Workflow"]
                    }],
                    'suggestions': get_cached_suggestions(cleaned_df[[col for col in cleaned_df.columns if col not in session.get('dropped_columns', [])]]),
                    'progress': session['progress'] | {'Clean': 'Done'}
                })
                smart_workflow_output = html.Div([
                    html.H4("Suggested Workflow:"),
                    html.Ul([html.Li(step) for step in workflow]),
                    display_cleaned_dataset(cleaned_df, df.shape),
                    html.P("Smart Workflow executed successfully!", className="text-success")
                ])
                undo_disabled = False
            except Exception as e:
                smart_workflow_output = html.Div(f"Error executing smart workflow: {str(e)}", className="text-danger")
                session['progress'] = session['progress'] | {'Clean': 'Failed'}
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page(),
                smart_workflow_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle custom rules
    elif trigger_id == "num_rules" and num_rules is not None:
        available_columns = [col for col in df.columns if col not in session.get('dropped_columns', [])] if df is not None else []
        custom_rules_container = [
            html.Div([
                html.H5(f"Rule {i + 1}"),
                dcc.Dropdown(
                    id={"type": "rule_col", "index": i},
                    options=[{"label": col, "value": col} for col in available_columns],
                    placeholder="Select column",
                    className="mb-2"
                ),
                dcc.Dropdown(
                    id={"type": "rule_cond", "index": i},
                    options=[
                        {"label": "Greater than", "value": "greater than"},
                        {"label": "Less than", "value": "less than"},
                        {"label": "Equal to", "value": "equal to"}
                    ],
                    placeholder="Condition",
                    className="mb-2"
                ),
                dcc.Input(
                    id={"type": "rule_threshold", "index": i},
                    type="number",
                    value=0.0,
                    placeholder="Threshold",
                    className="form-control mb-2"
                ),
                dcc.Dropdown(
                    id={"type": "rule_action", "index": i},
                    options=[
                        {"label": "Set to NaN", "value": "Set to NaN"},
                        {"label": "Set to Value", "value": "Set to Value"}
                    ],
                    placeholder="Action",
                    className="mb-2"
                ),
                dcc.Input(
                    id={"type": "rule_action_value", "index": i},
                    type="number",
                    value=0.0,
                    placeholder="Action Value",
                    className="form-control mb-2",
                    style={'display': 'none'}
                )
            ])
            for i in range(int(num_rules or 0))
        ]
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles.get(session.get('page', 'Clean'))),
                render_clean_page()
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle cleaning form (preview, apply, auto-clean)
    elif trigger_id in ["preview_button", "apply_button", "auto_clean_button"] and (preview_clicks or apply_clicks or auto_clean_clicks):
        if not df:
            smart_workflow_output = html.Div("Please upload a dataset first.", className="text-warning")
        else:
            try:
                selected_suggestions = [(sug, get_cached_suggestions(df)[idx][1]) for idx, sug_list in enumerate(suggestion_values) for sug in sug_list]
                options = {}
                for idx, sug in enumerate([sug for sug_list in suggestion_values for sug in sug_list]):
                    if "Handle special characters" in sug:
                        options["special_chars"] = special_chars_opts[idx] if idx < len(special_chars_opts) else "Drop them"
                    elif "Fill missing values" in sug:
                        col = extract_column(sug)
                        if col:
                            options[f"fill_{col}"] = fill_opts[idx] if idx < len(fill_opts) else "mode"
                    elif "Handle outliers" in sug:
                        col = extract_column(sug)
                        if col:
                            options[f"outlier_{col}"] = outlier_opts[idx] if idx < len(outlier_opts) else "Remove"
                custom_rules = [
                    {
                        "column": rule_cols[i],
                        "condition": rule_conds[i],
                        "threshold": rule_thresholds[i],
                        "action": rule_actions[i],
                        "action_value": rule_action_values[i] if rule_actions[i] == "Set to Value" else None
                    }
                    for i in range(min(len(rule_cols), len(rule_conds), len(rule_thresholds), len(rule_actions)))
                    if rule_cols[i] and rule_conds[i] and rule_thresholds[i] is not None and rule_actions[i]
                ]
                replace_with_value = replace_with_custom if replace_with == "Custom" else replace_with
                operations_selected = (
                    selected_suggestions or columns_to_drop or
                    (replace_value and replace_with_value) or encode_cols or
                    (enrich_col != "None" and enrich_api_key) or auto_clean_clicks or
                    (train_ml and target_col_ml and feature_cols_ml) or custom_rules
                )
                if not operations_selected:
                    smart_workflow_output = html.Div("Please select at least one cleaning operation.", className="text-warning")
                else:
                    cleaned_df, logs = apply_cleaning_operations(
                        df, selected_suggestions, columns_to_drop or [], options,
                        replace_value or "", replace_with_value or "NaN",
                        replace_scope or "All columns", encode_cols or [], encode_method or "Label Encoding",
                        auto_clean=auto_clean_clicks, enrich_col=enrich_col if enrich_col != "None" else None,
                        enrich_api_key=enrich_api_key, train_ml=bool(train_ml), target_col=target_col_ml,
                        feature_cols=feature_cols_ml or []
                    )
                    for rule in custom_rules:
                        col = rule["column"]
                        condition = rule["condition"]
                        threshold = rule["threshold"]
                        action = rule["action"]
                        action_value = rule["action_value"]
                        if col in cleaned_df.columns:
                            if condition == "greater than":
                                mask = cleaned_df[col] > threshold
                            elif condition == "less than":
                                mask = cleaned_df[col] < threshold
                            else:
                                mask = cleaned_df[col] == threshold
                            if action == "Set to NaN":
                                cleaned_df.loc[mask, col] = pd.NA
                            else:
                                cleaned_df.loc[mask, col] = action_value
                            logs.append(
                                f"Applied custom rule on {col}: {condition} {threshold}, {action} "
                                f"{'NaN' if action == 'Set to NaN' else action_value}"
                            )
                    if preview_clicks:
                        smart_workflow_output = html.Div([
                            html.H3("Preview of Changes"),
                            html.H4("Before"),
                            html.Table([
                                html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                                html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(len(df), 10))])
                            ], className="table table-striped"),
                            html.H4("After"),
                            html.Table([
                                html.Thead(html.Tr([html.Th(col) for col in cleaned_df.columns])),
                                html.Tbody([html.Tr([html.Td(cleaned_df.iloc[i][col]) for col in cleaned_df.columns]) for i in range(min(len(cleaned_df), 10))])
                            ], className="table table-striped"),
                            html.H4("Preview Logs"),
                            html.Ul([html.Li(log) for log in logs])
                        ])
                    if apply_clicks or auto_clean_clicks:
                        session.update({
                            'cleaned_df': cleaned_df,
                            'logs': logs,
                            'previous_states': session.get('previous_states', []) + [(df.copy(), session.get('logs', []).copy())],
                            'redo_states': [],
                            'dropped_columns': session.get('dropped_columns', []) + (columns_to_drop or []),
                            'cleaning_history': session.get('cleaning_history', []) + [{
                                "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "logs": logs
                            }],
                            'suggestions': get_cached_suggestions(cleaned_df[[col for col in cleaned_df.columns if col not in session.get('dropped_columns', []) + (columns_to_drop or [])]]),
                            'progress': session['progress'] | {'Clean': 'Done'}
                        })
                        smart_workflow_output = html.Div([
                            display_cleaned_dataset(cleaned_df, df.shape),
                            html.P("Changes applied successfully!", className="text-success")
                        ])
                        undo_disabled = False
                        columns_drop_options = [{"label": col, "value": col} for col in cleaned_df.columns if col not in session.get('dropped_columns', [])]
                        encode_cols_options = [{"label": col, "value": col} for col in cleaned_df.select_dtypes(include=['object', 'category']).columns]
                        anomaly_cols_options = [{"label": col, "value": col} for col in cleaned_df.select_dtypes(include=['int64', 'float64']).columns]
                        target_col_ml_options = [{"label": col, "value": col} for col in cleaned_df.columns]
                        feature_cols_ml_options = [{"label": col, "value": col} for col in cleaned_df.columns]
            except Exception as e:
                smart_workflow_output = html.Div(f"Error processing cleaning operations: {str(e)}", className="text-danger")
                session['progress'] = session['progress'] | {'Clean': 'Failed'}
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page(),
                smart_workflow_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle save template
    elif trigger_id == "save_template_button" and save_template_clicks:
        if not template_name:
            save_template_output = html.Div("Please enter a template name.", className="text-warning")
        else:
            try:
                selected_suggestions = [(sug, get_cached_suggestions(df)[idx][1]) for idx, sug_list in enumerate(suggestion_values) for sug in sug_list]
                options = {}
                for idx, sug in enumerate([sug for sug_list in suggestion_values for sug in sug_list]):
                    if "Handle special characters" in sug:
                        options["special_chars"] = special_chars_opts[idx] if idx < len(special_chars_opts) else "Drop them"
                    elif "Fill missing values" in sug:
                        col = extract_column(sug)
                        if col:
                            options[f"fill_{col}"] = fill_opts[idx] if idx < len(fill_opts) else "mode"
                    elif "Handle outliers" in sug:
                        col = extract_column(sug)
                        if col:
                            options[f"outlier_{col}"] = outlier_opts[idx] if idx < len(outlier_opts) else "Remove"
                custom_rules = [
                    {
                        "column": rule_cols[i],
                        "condition": rule_conds[i],
                        "threshold": rule_thresholds[i],
                        "action": rule_actions[i],
                        "action_value": rule_action_values[i] if rule_actions[i] == "Set to Value" else None
                    }
                    for i in range(min(len(rule_cols), len(rule_conds), len(rule_thresholds), len(rule_actions)))
                    if rule_cols[i] and rule_conds[i] and rule_thresholds[i] is not None and rule_actions[i]
                ]
                template = {
                    "columns_to_drop": columns_to_drop or [],
                    "selected_suggestions": selected_suggestions,
                    "options": options,
                    "replace_value": replace_value or "",
                    "replace_with": replace_with_custom if replace_with == "Custom" else replace_with or "NaN",
                    "replace_scope": replace_scope or "All columns",
                    "encode_cols": encode_cols or [],
                    "encode_method": encode_method or "Label Encoding",
                    "enrich_col": enrich_col if enrich_col != "None" else None,
                    "train_ml": bool(train_ml),
                    "target_col": target_col_ml,
                    "feature_cols": feature_cols_ml or [],
                    "custom_rules": custom_rules
                }
                session['cleaning_templates'] = session.get('cleaning_templates', {})
                session['cleaning_templates'][template_name] = {k: v for k, v in template.items() if k != "enrich_api_key"}
                save_template_output = html.P(f"Saved template '{template_name}'", className="text-success")
                template_options = [{"label": "None", "value": "None"}] + [
                    {"label": name, "value": name} for name in session.get('cleaning_templates', {}).keys()
                ]
            except Exception as e:
                save_template_output = html.Div(f"Error saving template: {str(e)}", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page()
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle apply template
    elif trigger_id == "apply_template_button" and apply_template_clicks:
        if not apply_template or apply_template == "None":
            apply_template_output = html.Div("Please select a template.", className="text-warning")
        else:
            try:
                template = session.get('cleaning_templates', {}).get(apply_template, {})
                cleaned_df, logs = apply_cleaning_operations(
                    df,
                    selected_suggestions=template.get("selected_suggestions", []),
                    columns_to_drop=template.get("columns_to_drop", []),
                    options=template.get("options", {}),
                    replace_value=template.get("replace_value", ""),
                    replace_with=template.get("replace_with", "NaN"),
                    replace_scope=template.get("replace_scope", "All columns"),
                    encode_cols=template.get("encode_cols", []),
                    encode_method=template.get("encode_method", "Label Encoding"),
                    auto_clean=False,
                    enrich_col=template.get("enrich_col", None),
                    enrich_api_key=enrich_api_key,
                    train_ml=template.get("train_ml", False),
                    target_col=template.get("target_col", None),
                    feature_cols=template.get("feature_cols", [])
                )
                custom_rules = template.get("custom_rules", [])
                for rule in custom_rules:
                    col = rule["column"]
                    condition = rule["condition"]
                    threshold = rule["threshold"]
                    action = rule["action"]
                    action_value = rule["action_value"]
                    if col in cleaned_df.columns:
                        if condition == "greater than":
                            mask = cleaned_df[col] > threshold
                        elif condition == "less than":
                            mask = cleaned_df[col] < threshold
                        else:
                            mask = cleaned_df[col] == threshold
                        if action == "Set to NaN":
                            cleaned_df.loc[mask, col] = pd.NA
                        else:
                            cleaned_df.loc[mask, col] = action_value
                        logs.append(
                            f"Applied custom rule from template on {col}: {condition} {threshold}, {action} "
                            f"{'NaN' if action == 'Set to NaN' else action_value}"
                        )
                session.update({
                    'cleaned_df': cleaned_df,
                    'logs': logs,
                    'previous_states': session.get('previous_states', []) + [(df.copy(), session.get('logs', []).copy())],
                    'redo_states': [],
                    'dropped_columns': session.get('dropped_columns', []) + template.get("columns_to_drop", []),
                    'cleaning_history': session.get('cleaning_history', []) + [{
                        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": logs + [f"Applied template '{apply_template}'"]
                    }],
                    'suggestions': get_cached_suggestions(cleaned_df[[col for col in cleaned_df.columns if col not in session.get('dropped_columns', []) + template.get("columns_to_drop", [])]]),
                    'progress': session['progress'] | {'Clean': 'Done'}
                })
                apply_template_output = html.Div([
                    display_cleaned_dataset(cleaned_df, df.shape),
                    html.P(f"Applied template '{apply_template}'", className="text-success")
                ])
                undo_disabled = False
                columns_drop_options = [{"label": col, "value": col} for col in cleaned_df.columns if col not in session.get('dropped_columns', [])]
                encode_cols_options = [{"label": col, "value": col} for col in cleaned_df.select_dtypes(include=['object', 'category']).columns]
                anomaly_cols_options = [{"label": col, "value": col} for col in cleaned_df.select_dtypes(include=['int64', 'float64']).columns]
                target_col_ml_options = [{"label": col, "value": col} for col in cleaned_df.columns]
                feature_cols_ml_options = [{"label": col, "value": col} for col in cleaned_df.columns]
            except Exception as e:
                apply_template_output = html.Div(f"Error applying template: {str(e)}", className="text-danger")
                session['progress'] = session['progress'] | {'Clean': 'Failed'}
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page(),
                apply_template_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle undo
    elif trigger_id == "undo_button" and undo_clicks:
        if not session.get('previous_states'):
            smart_workflow_output = html.Div("No previous state to undo.", className="text-warning")
        else:
            try:
                current_state = (session.get('cleaned_df').copy(), session.get('logs', []).copy())
                session['redo_states'] = session.get('redo_states', []) + [current_state]
                if len(session['redo_states']) > 5:
                    session['redo_states'].pop(0)
                previous_df, previous_logs = session['previous_states'].pop()
                session.update({
                    'cleaned_df': previous_df,
                    'logs': previous_logs,
                    'suggestions': get_cached_suggestions(previous_df[[col for col in previous_df.columns if col not in session.get('dropped_columns', [])]]),
                    'cleaning_history': session.get('cleaning_history', []) + [{
                        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": ["Undid last cleaning operation"]
                    }]
                })
                smart_workflow_output = display_cleaned_dataset(previous_df, df.shape)
                undo_disabled = len(session['previous_states']) == 0
                redo_disabled = False
                columns_drop_options = [{"label": col, "value": col} for col in previous_df.columns if col not in session.get('dropped_columns', [])]
                encode_cols_options = [{"label": col, "value": col} for col in previous_df.select_dtypes(include=['object', 'category']).columns]
                anomaly_cols_options = [{"label": col, "value": col} for col in previous_df.select_dtypes(include=['int64', 'float64']).columns]
                target_col_ml_options = [{"label": col, "value": col} for col in previous_df.columns]
                feature_cols_ml_options = [{"label": col, "value": col} for col in previous_df.columns]
            except Exception as e:
                smart_workflow_output = html.Div(f"Error undoing changes: {str(e)}", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page(),
                smart_workflow_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle redo
    elif trigger_id == "redo_button" and redo_clicks:
        if not session.get('redo_states'):
            smart_workflow_output = html.Div("No state to redo.", className="text-warning")
        else:
            try:
                session['previous_states'] = session.get('previous_states', []) + [(session.get('cleaned_df').copy(), session.get('logs', []).copy())]
                if len(session['previous_states']) > 5:
                    session['previous_states'].pop(0)
                redo_df, redo_logs = session['redo_states'].pop()
                session.update({
                    'cleaned_df': redo_df,
                    'logs': redo_logs,
                    'suggestions': get_cached_suggestions(redo_df[[col for col in redo_df.columns if col not in session.get('dropped_columns', [])]]),
                    'cleaning_history': session.get('cleaning_history', []) + [{
                        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "logs": ["Redid last cleaning operation"]
                    }]
                })
                smart_workflow_output = display_cleaned_dataset(redo_df, df.shape)
                redo_disabled = len(session['redo_states']) == 0
                undo_disabled = False
                columns_drop_options = [{"label": col, "value": col} for col in redo_df.columns if col not in session.get('dropped_columns', [])]
                encode_cols_options = [{"label": col, "value": col} for col in redo_df.select_dtypes(include=['object', 'category']).columns]
                anomaly_cols_options = [{"label": col, "value": col} for col in redo_df.select_dtypes(include=['int64', 'float64']).columns]
                target_col_ml_options = [{"label": col, "value": col} for col in redo_df.columns]
                feature_cols_ml_options = [{"label": col, "value": col} for col in redo_df.columns]
            except Exception as e:
                smart_workflow_output = html.Div(f"Error redoing changes: {str(e)}", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page(),
                smart_workflow_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle export to Tableau
    elif trigger_id == "export_tableau_button" and export_tableau_clicks:
        if not session.get('cleaned_df'):
            export_tableau_output = html.Div("No cleaned dataset to export.", className="text-warning")
        else:
            try:
                cleaned_df = session['cleaned_df']
                csv = cleaned_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                download_data = dcc.send_data_frame(cleaned_df.to_csv, f"tableau_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                export_tableau_output = html.P("Exported dataset for Tableau.", className="text-success")
            except Exception as e:
                export_tableau_output = html.Div(f"Error exporting dataset: {str(e)}", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Clean']),
                render_clean_page(),
                export_tableau_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle visualization controls
    elif trigger_id == "global_filter_col" and filter_col:
        if not df:
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Visualize']),
                    render_visualization_page(),
                    html.Div("Please upload a dataset first.", className="text-warning")
                ]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        if filter_col == "None":
            session['filtered_df'] = df
            session['dashboard_filters'] = {}
            filter_controls = []
        else:
            col_type = df[filter_col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                if pd.isna(min_val) or pd.isna(max_val):
                    filter_controls = [html.P(f"Column {filter_col} contains missing values.", className="text-warning")]
                elif min_val == max_val:
                    filter_controls = [html.P(f"Column {filter_col} has identical values ({min_val}).", className="text-warning")]
                else:
                    filter_controls = [
                        dcc.RangeSlider(
                            id="filter_range",
                            min=min_val,
                            max=max_val,
                            value=[min_val, max_val],
                            step=(max_val - min_val) / 100,
                            marks={min_val: str(min_val), max_val: str(max_val)},
                            className="mb-3"
                        )
                    ]
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                min_date, max_date = df[filter_col].min(), df[filter_col].max()
                if pd.isna(min_date) or pd.isna(max_date):
                    filter_controls = [html.P(f"Column {filter_col} contains missing values.", className="text-warning")]
                elif min_date == max_date:
                    filter_controls = [html.P(f"Column {filter_col} has identical dates ({min_date}).", className="text-warning")]
                else:
                    filter_controls = [
                        dcc.DatePickerRange(
                            id="filter_date",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            className="mb-3"
                        )
                    ]
            else:
                unique_vals = df[filter_col].dropna().unique().tolist()
                if len(unique_vals) == 1:
                    filter_controls = [html.P(f"Column {filter_col} has a single value ({unique_vals[0]}).", className="text-warning")]
                else:
                    filter_controls = [
                        dcc.Dropdown(
                            id="filter_values",
                            options=[{"label": val, "value": val} for val in unique_vals],
                            value=unique_vals,
                            multi=True,
                            className="mb-3"
                        )
                    ]
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles.get(session.get('page', 'Visualize'))),
                render_visualization_page()
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle visualization generation
    elif trigger_id == "submit_viz" and submit_viz:
        if not df:
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Visualize']),
                    render_visualization_page(),
                    html.Div("Please upload a dataset first.", className="text-warning")
                ]),
                session,
                "",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        session['progress'] = session['progress'] | {'Visualize': 'In Progress'}
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_cols = df.columns.tolist()
        time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        object_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Configure visualization controls based on viz_type
        config_controls = []
        if viz_type in ["Bar", "Scatter", "Line", "3D Scatter", "Bubble Chart"]:
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="X-Axis Column", options=[{"label": col, "value": col} for col in all_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Y-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2"),
                dcc.Dropdown(id="hue_col", placeholder="Group By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=hue_col, className="mb-2")
            ])
            if viz_type == "3D Scatter":
                config_controls.append(dcc.Dropdown(id="z_col", placeholder="Z-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=z_col, className="mb-2"))
            if viz_type == "Bubble Chart":
                config_controls.append(dcc.Dropdown(id="size_col", placeholder="Size By", options=[{"label": col, "value": col} for col in numeric_cols], value=size_col, className="mb-2"))
        elif viz_type == "Histogram":
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="Column", options=[{"label": col, "value": col} for col in numeric_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="hue_col", placeholder="Group By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=hue_col, className="mb-2")
            ])
        elif viz_type in ["Box", "Violin", "Strip Plot", "Swarm Plot"]:
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="X-Axis Column (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Y-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2"),
                dcc.Dropdown(id="hue_col", placeholder="Group By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=hue_col, className="mb-2")
            ])
        elif viz_type == "Heatmap (Correlation)":
            config_controls.append(
                dcc.Dropdown(id="selected_cols", placeholder="Columns for Correlation", options=[{"label": col, "value": col} for col in numeric_cols], value=selected_cols, multi=True, className="mb-2")
            )
        elif viz_type == "Pie":
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="Categories", options=[{"label": col, "value": col} for col in all_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Values", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2")
            ])
        elif viz_type == "Time Series Forecast":
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="Time Column", options=[{"label": col, "value": col} for col in time_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Value Column", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2"),
                dcc.Slider(id="periods", min=1, max=30, step=1, value=periods or 5, marks={1: '1', 30: '30'}, className="mb-2"),
                dcc.Dropdown(id="freq", options=[{"label": f, "value": f} for f in ["D", "M", "Y"]], value=freq or "D", className="mb-2")
            ])
        elif viz_type == "Geospatial Map":
            config_controls.extend([
                dcc.Dropdown(id="lat_col", placeholder="Latitude Column", options=[{"label": col, "value": col} for col in numeric_cols], value=lat_col, className="mb-2"),
                dcc.Dropdown(id="lon_col", placeholder="Longitude Column", options=[{"label": col, "value": col} for col in numeric_cols], value=lon_col, className="mb-2"),
                dcc.Dropdown(id="size_col", placeholder="Size By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in numeric_cols], value=size_col, className="mb-2"),
                dcc.Dropdown(id="color_col", placeholder="Color By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=color_col, className="mb-2")
            ])
        elif viz_type == "Choropleth Map":
            config_controls.extend([
                dcc.Dropdown(id="geo_col", placeholder="Geographic Column", options=[{"label": col, "value": col} for col in all_cols], value=geo_col, className="mb-2"),
                dcc.Dropdown(id="value_col", placeholder="Values", options=[{"label": col, "value": col} for col in numeric_cols], value=value_col, className="mb-2")
            ])
        elif viz_type == "Heatmap (Geospatial)":
            config_controls.extend([
                dcc.Dropdown(id="lat_col", placeholder="Latitude Column", options=[{"label": col, "value": col} for col in numeric_cols], value=lat_col, className="mb-2"),
                dcc.Dropdown(id="lon_col", placeholder="Longitude Column", options=[{"label": col, "value": col} for col in numeric_cols], value=lon_col, className="mb-2")
            ])
        elif viz_type == "Area Chart":
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="Time Column", options=[{"label": col, "value": col} for col in time_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Y-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2"),
                dcc.Dropdown(id="hue_col", placeholder="Group By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=hue_col, className="mb-2")
            ])
        elif viz_type in ["Density Plot", "ECDF Plot"]:
            config_controls.append(
                dcc.Dropdown(id="x_col", placeholder="Column", options=[{"label": col, "value": col} for col in numeric_cols], value=x_col, className="mb-2")
            )
        elif viz_type in ["Treemap", "Sunburst Chart"]:
            config_controls.extend([
                dcc.Dropdown(id="path_cols", placeholder="Hierarchy", options=[{"label": col, "value": col} for col in all_cols], value=path_cols, multi=True, className="mb-2"),
                dcc.Dropdown(id="values_col", placeholder="Values", options=[{"label": col, "value": col} for col in numeric_cols], value=values_col, className="mb-2")
            ])
        elif viz_type == "Dendrogram":
            config_controls.append(
                dcc.Dropdown(id="selected_cols", placeholder="Select numerical columns", options=[{"label": col, "value": col} for col in numeric_cols], value=selected_cols, multi=True, className="mb-2")
            )
        elif viz_type == "Network Graph":
            config_controls.extend([
                dcc.Dropdown(id="source_col", placeholder="Source Node", options=[{"label": col, "value": col} for col in all_cols], value=source_col, className="mb-2"),
                dcc.Dropdown(id="target_col", placeholder="Target Node", options=[{"label": col, "value": col} for col in all_cols], value=target_col, className="mb-2"),
                dcc.Dropdown(id="weight_col", placeholder="Weight (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in numeric_cols], value=weight_col, className="mb-2")
            ])
        elif viz_type == "Timeline":
            config_controls.extend([
                dcc.Dropdown(id="time_col", placeholder="Time Column", options=[{"label": col, "value": col} for col in time_cols], value=time_col, className="mb-2"),
                dcc.Dropdown(id="event_col", placeholder="Event Column", options=[{"label": col, "value": col} for col in all_cols], value=event_col, className="mb-2")
            ])
        elif viz_type == "Gantt Chart":
            config_controls.extend([
                dcc.Dropdown(id="start_col", placeholder="Start Time", options=[{"label": col, "value": col} for col in time_cols], value=start_col, className="mb-2"),
                dcc.Dropdown(id="end_col", placeholder="End Time", options=[{"label": col, "value": col} for col in time_cols], value=end_col, className="mb-2"),
                dcc.Dropdown(id="task_col", placeholder="Task", options=[{"label": col, "value": col} for col in all_cols], value=task_col, className="mb-2")
            ])
        elif viz_type == "Calendar Heatmap":
            config_controls.extend([
                dcc.Dropdown(id="date_col", placeholder="Date Column", options=[{"label": col, "value": col} for col in time_cols], value=date_col, className="mb-2"),
                dcc.Dropdown(id="value_col", placeholder="Values", options=[{"label": col, "value": col} for col in numeric_cols], value=value_col, className="mb-2")
            ])
        elif viz_type == "Parallel Coordinates":
            config_controls.extend([
                dcc.Dropdown(id="selected_cols", placeholder="Select numerical columns", options=[{"label": col, "value": col} for col in numeric_cols], value=selected_cols, multi=True, className="mb-2"),
                dcc.Dropdown(id="color_col", placeholder="Color By (Optional)", options=[{"label": "None", "value": "None"}] + [{"label": col, "value": col} for col in all_cols], value=color_col, className="mb-2")
            ])
        elif viz_type == "Radar Chart":
            config_controls.extend([
                dcc.Dropdown(id="selected_cols", placeholder="Select numerical columns", options=[{"label": col, "value": col} for col in numeric_cols], value=selected_cols, multi=True, className="mb-2"),
                dcc.Dropdown(id="group_col", placeholder="Group", options=[{"label": col, "value": col} for col in all_cols], value=group_col, className="mb-2")
            ])
        elif viz_type == "Surface Plot":
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="X-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Y-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2"),
                dcc.Dropdown(id="z_col", placeholder="Z-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=z_col, className="mb-2")
            ])
        elif viz_type == "Word Cloud":
            config_controls.append(
                dcc.Dropdown(id="text_col", placeholder="Text Column", options=[{"label": col, "value": col} for col in object_cols], value=text_col, className="mb-2")
            )
        elif viz_type == "Gauge Chart":
            config_controls.extend([
                dcc.Dropdown(id="value_col", placeholder="Value", options=[{"label": col, "value": col} for col in numeric_cols], value=value_col, className="mb-2"),
                dcc.Input(id="max_value", placeholder="Max Value", type="number", value=max_value, className="form-control mb-2")
            ])
        elif viz_type == "Funnel Chart":
            config_controls.extend([
                dcc.Dropdown(id="stages_col", placeholder="Stages", options=[{"label": col, "value": col} for col in all_cols], value=stages_col, className="mb-2"),
                dcc.Dropdown(id="values_col", placeholder="Values", options=[{"label": col, "value": col} for col in numeric_cols], value=values_col, className="mb-2")
            ])
        elif viz_type == "Sankey Diagram":
            config_controls.extend([
                dcc.Dropdown(id="source_col", placeholder="Source", options=[{"label": col, "value": col} for col in all_cols], value=source_col, className="mb-2"),
                dcc.Dropdown(id="target_col", placeholder="Target", options=[{"label": col, "value": col} for col in all_cols], value=target_col, className="mb-2"),
                dcc.Dropdown(id="value_col", placeholder="Value", options=[{"label": col, "value": col} for col in numeric_cols], value=value_col, className="mb-2")
            ])
        elif viz_type == "Waterfall Chart":
            config_controls.extend([
                dcc.Dropdown(id="measure_col", placeholder="Measure", options=[{"label": col, "value": col} for col in all_cols], value=measure_col, className="mb-2"),
                dcc.Dropdown(id="x_col", placeholder="X-Axis (categories)", options=[{"label": col, "value": col} for col in all_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Y-Axis (values)", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2")
            ])
        elif viz_type == "Pair Plot":
            config_controls.append(
                dcc.Dropdown(id="selected_cols", placeholder="Select numerical columns", options=[{"label": col, "value": col} for col in numeric_cols], value=selected_cols, multi=True, className="mb-2")
            )
        elif viz_type == "Joint Plot":
            config_controls.extend([
                dcc.Dropdown(id="x_col", placeholder="X-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=x_col, className="mb-2"),
                dcc.Dropdown(id="y_col", placeholder="Y-Axis Column", options=[{"label": col, "value": col} for col in numeric_cols], value=y_col, className="mb-2")
            ])
        elif viz_type == "Clustering":
            config_controls.extend([
                dcc.Dropdown(id="cluster_cols", placeholder="Select columns for clustering", options=[{"label": col, "value": col} for col in numeric_cols], value=cluster_cols, multi=True, className="mb-2"),
                dcc.Slider(id="n_clusters", min=2, max=10, step=1, value=n_clusters or 3, marks={i: str(i) for i in range(2, 11)}, className="mb-2")
            ])

        try:
            if len(df) > 1000:
                df = df.sample(min(1000, len(df)), random_state=42)
                viz_output = html.P(f"Dataset sampled to {len(df)} rows for performance.", className="text-info")
            fig = None
            is_wordcloud = False
            is_jointplot = False
            is_clustering = False

            # Generate visualization based on viz_type
            if viz_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Line":
                fig = px.line(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Box":
                fig = px.box(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Violin":
                fig = px.violin(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Heatmap (Correlation)":
                if not selected_cols or len(selected_cols) < 2:
                    viz_output = html.Div("Select at least two numerical columns.", className="text-danger")
                else:
                    corr = df[selected_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title=chart_title)
            elif viz_type == "Pie":
                fig = px.pie(df, names=x_col, values=y_col, title=chart_title)
            elif viz_type == "Time Series Forecast":
                if not time_cols:
                    viz_output = html.Div("No datetime columns available.", className="text-danger")
                else:
                    forecast_df = forecast_time_series(df, y_col, periods, time_col=x_col, freq=freq)
                    historical = df[[x_col, y_col]].copy()
                    historical['Type'] = 'Historical'
                    forecast_df = forecast_df.reset_index().rename(columns={'index': x_col, y_col: y_col})
                    forecast_df['Type'] = 'Forecast'
                    combined_df = pd.concat([historical, forecast_df], ignore_index=True)
                    fig = px.line(combined_df, x=x_col, y=y_col, color='Type', title=chart_title)
                    fig.add_vline(x=df[x_col].iloc[-1], line_dash="dash", line_color="red")
            elif viz_type == "3D Scatter":
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Geospatial Map":
                fig = px.scatter_mapbox(
                    df, lat=lat_col, lon=lon_col, size=None if size_col == "None" else size_col,
                    color=None if color_col == "None" else color_col, title=chart_title, zoom=3
                )
                fig.update_layout(mapbox_style="open-street-map")
            elif viz_type == "Choropleth Map":
                if not geo_col or not value_col:
                    viz_output = html.Div("Please select both geographic and value columns.", className="text-danger")
                else:
                    fig = px.choropleth(df, locations=geo_col, locationmode="country names", color=value_col, title=chart_title)
            elif viz_type == "Heatmap (Geospatial)":
                fig = px.density_mapbox(
                    df, lat=lat_col, lon=lon_col, radius=10,
                    center=dict(lat=df[lat_col].mean(), lon=df[lon_col].mean()),
                    zoom=5, mapbox_style="open-street-map", title=chart_title
                )
            elif viz_type == "Area Chart":
                if not time_cols:
                    viz_output = html.Div("No datetime columns available.", className="text-danger")
                else:
                    fig = px.area(df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Strip Plot":
                fig = px.strip(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Swarm Plot":
                fig = px.strip(df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=chart_title)
                fig.update_traces(jitter=1)
            elif viz_type == "Density Plot":
                fig = px.density_contour(df, x=x_col, title=chart_title)
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            elif viz_type == "ECDF Plot":
                x = df[x_col].dropna()
                ecdf = np.arange(1, len(x) + 1) / len(x)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.sort(x), y=ecdf, mode='lines', name='ECDF'))
                fig.update_layout(title=chart_title, xaxis_title=x_col, yaxis_title="Cumulative Probability")
            elif viz_type == "Treemap":
                if not path_cols:
                    viz_output = html.Div("Select at least one column for the hierarchy.", className="text-danger")
                else:
                    fig = px.treemap(df, path=path_cols, values=values_col, title=chart_title)
            elif viz_type == "Sunburst Chart":
                if not path_cols:
                    viz_output = html.Div("Select at least one column for the hierarchy.", className="text-danger")
                else:
                    fig = px.sunburst(df, path=path_cols, values=values_col, title=chart_title)
            elif viz_type == "Dendrogram":
                if not selected_cols or len(selected_cols) < 2:
                    viz_output = html.Div("Select at least two numerical columns.", className="text-danger")
                else:
                    X = df[selected_cols].dropna()
                    Z = linkage(X, method='ward')
                    fig = go.Figure()
                    dendro = dendrogram(Z, no_plot=True)
                    fig.add_trace(go.Scatter(x=dendro['icoord'][0], y=dendro['dcoord'][0], mode='lines', line=dict(color='white')))
                    for i in range(1, len(dendro['icoord'])):
                        fig.add_trace(go.Scatter(x=dendro['icoord'][i], y=dendro['dcoord'][i], mode='lines', line=dict(color='white'), showlegend=False))
                    fig.update_layout(title=chart_title, xaxis_title="Sample Index", yaxis_title="Distance")
            elif viz_type == "Network Graph":
                G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, edge_attr=None if weight_col == "None" else weight_col)
                pos = nx.spring_layout(G)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none'))
                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='white'), text=list(G.nodes()), hoverinfo='text'))
                fig.update_layout(title=chart_title, showlegend=False)
            elif viz_type == "Timeline":
                fig = px.scatter(df, x=time_col, y=[0] * len(df), text=event_col, title=chart_title)
                fig.update_traces(textposition="top center")
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            elif viz_type == "Gantt Chart":
                fig = px.timeline(df, x_start=start_col, x_end=end_col, y=task_col, title=chart_title)
            elif viz_type == "Calendar Heatmap":
                data = df.groupby(date_col)[value_col].sum().reset_index()
                fig = px.density_heatmap(data, x=data[date_col].dt.day, y=data[date_col].dt.month, z=value_col, title=chart_title)
            elif viz_type == "Parallel Coordinates":
                if not selected_cols or len(selected_cols) < 2:
                    viz_output = html.Div("Select at least two numerical columns.", className="text-danger")
                else:
                    fig = px.parallel_coordinates(df, dimensions=selected_cols, color=None if color_col == "None" else color_col, title=chart_title)
            elif viz_type == "Radar Chart":
                if not selected_cols or len(selected_cols) < 2:
                    viz_output = html.Div("Select at least two numerical columns.", className="text-danger")
                else:
                    grouped = df.groupby(group_col)[selected_cols].mean().reset_index()
                    fig = go.Figure()
                    for _, row in grouped.iterrows():
                        fig.add_trace(go.Scatterpolar(r=[row[col] for col in selected_cols], theta=selected_cols, fill='toself', name=row[group_col]))
                    fig.update_layout(title=chart_title)
            elif viz_type == "Bubble Chart":
                fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=None if hue_col == "None" else hue_col, title=chart_title)
            elif viz_type == "Surface Plot":
                data = df.pivot_table(index=x_col, columns=y_col, values=z_col).fillna(0)
                fig = go.Figure(data=[go.Surface(z=data.values, x=data.columns, y=data.index)])
                fig.update_layout(title=chart_title, scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))
            elif viz_type == "Word Cloud":
                text = " ".join(df[text_col].dropna().astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                plt.figure(figsize=(10, 5), facecolor='black')
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                plt.close()
                is_wordcloud = True
                viz_output = html.Img(src=f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}")
            elif viz_type == "Gauge Chart":
                value = df[value_col].mean()
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': chart_title},
                    gauge={'axis': {'range': [0, max_value or df[value_col].max() * 1.2]}, 'bar': {'color': "white"}}
                ))
            elif viz_type == "Funnel Chart":
                fig = px.funnel(df, x=values_col, y=stages_col, title=chart_title)
            elif viz_type == "Sankey Diagram":
                label_list = list(set(df[source_col].tolist() + df[target_col].tolist()))
                label_dict = {label: idx for idx, label in enumerate(label_list)}
                source = df[source_col].map(label_dict)
                target = df[target_col].map(label_dict)
                value = df[value_col]
                fig = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label_list),
                    link=dict(source=source, target=target, value=value)
                )])
                fig.update_layout(title=chart_title)
            elif viz_type == "Waterfall Chart":
                fig = go.Figure(go.Waterfall(x=df[x_col], measure=df[measure_col], y=df[y_col], textposition="auto"))
                fig.update_layout(title=chart_title)
            elif viz_type == "Pair Plot":
                if not selected_cols or len(selected_cols) < 2:
                    viz_output = html.Div("Select at least two numerical columns.", className="text-danger")
                else:
                    fig = px.scatter_matrix(df, dimensions=selected_cols, title=chart_title)
            elif viz_type == "Joint Plot":
                joint_plot = sns.jointplot(data=df, x=x_col, y=y_col, kind="scatter")
                buf = io.BytesIO()
                joint_plot.savefig(buf, format="png", bbox_inches="tight")
                plt.close()
                is_jointplot = True
                viz_output = html.Img(src=f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}")
            elif viz_type == "Clustering":
                if not cluster_cols or len(cluster_cols) < 2:
                    viz_output = html.Div("Select at least two numerical columns.", className="text-danger")
                else:
                    labels = perform_clustering(df, cluster_cols, n_clusters)
                    session['clustering_labels'] = labels
                    session['cluster_cols'] = cluster_cols
                    df['Cluster'] = labels
                    session['filtered_df'] = df
                    is_clustering = True
                    viz_output = html.Div([
                        dcc.Graph(figure=px.scatter(
                            df, x=cluster_cols[0], y=cluster_cols[1], color=labels.astype(str),
                            labels={'color': 'Cluster'}, title="Clustering Results (2D Scatter Plot)"
                        )) if len(cluster_cols) >= 2 else None,
                        dcc.Graph(figure=px.scatter_3d(
                            df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color=labels.astype(str),
                            labels={'color': 'Cluster'}, title="Clustering Results (3D Scatter Plot)"
                        )) if len(cluster_cols) >= 3 else None,
                        dcc.Graph(figure=px.bar(
                            x=pd.Series(labels).value_counts().sort_index().index.astype(str),
                            y=pd.Series(labels).value_counts().sort_index().values,
                            labels={'x': 'Cluster', 'y': 'Number of Points'},
                            title="Cluster Distribution"
                        )),
                        html.H4("Dataset with Cluster Labels"),
                        html.Table([
                            html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                            html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(len(df), 10))])
                        ], className="table table-striped")
                    ])

            # Render visualization output
            if not is_wordcloud and not is_jointplot and not is_clustering:
                if fig:
                    viz_output = html.Div([
                        dcc.Graph(figure=fig, style={'width': '100%'}),
                        html.H4("Export Visualization"),
                        dcc.Dropdown(
                            id="export_format",
                            options=[{"label": fmt, "value": fmt} for fmt in ["PNG", "SVG", "PDF"]],
                            value="PNG",
                            className="mb-2"
                        ),
                        dbc.Button("Export", id="export_viz", color="secondary", className="mb-3"),
                        html.H4("Suggested Follow-Up Visualizations"),
                        html.Ul([
                            html.Li(f"{sug}: {reason}") for sug, reason in [
                                (suggest_visualization(df)[0], suggest_visualization(df)[1]),
                                ("Heatmap (Correlation)", "Explore correlations.") if viz_type == "Scatter" and len(numeric_cols) >= 2 else None,
                                ("Pie Chart", "Visualize categorical distribution.") if viz_type == "Bar" and len(object_cols) > 0 else None,
                                ("Time Series Forecast", "Predict future values.") if viz_type == "Line" and time_cols else None
                            ] if sug
                        ]),
                        html.P("Visualization generated successfully!", className="text-success")
                    ])
                    if add_to_dashboard:
                        chart_config = {
                            "type": viz_type,
                            "title": chart_title,
                            "x_col": x_col,
                            "y_col": y_col,
                            "hue_col": hue_col,
                            "z_col": z_col,
                            "lat_col": lat_col,
                            "lon_col": lon_col,
                            "size_col": size_col,
                            "color_col": color_col,
                            "periods": periods,
                            "freq": freq,
                            "geo_col": geo_col,
                            "value_col": value_col,
                            "path_cols": path_cols,
                            "values_col": values_col,
                            "selected_cols": selected_cols,
                            "source_col": source_col,
                            "target_col": target_col,
                            "weight_col": weight_col,
                            "time_col": time_col,
                            "event_col": event_col,
                            "start_col": start_col,
                            "end_col": end_col,
                            "task_col": task_col,
                            "date_col": date_col,
                            "group_col": group_col,
                            "text_col": text_col,
                            "max_value": max_value,
                            "stages_col": stages_col,
                            "measure_col": measure_col
                        }
                        session['dashboard_charts'] = session.get('dashboard_charts', []) + [chart_config]
                        viz_output.children.append(html.P("Chart added to dashboard!", className="text-success"))

            # Update dashboard output
            dashboard_charts = session.get('dashboard_charts', [])
            dashboard_output = html.Div([
                html.H4("Dashboard"),
                html.P("No charts added to dashboard yet.", className="text-info") if not dashboard_charts else html.Div([
                    html.Div([
                        html.H5(f"Chart {i + 1}: {chart['title']}"),
                        dcc.Graph(figure=px.bar(df, x=chart['x_col'], y=chart['y_col'], title=chart['title']) if chart['type'] == "Bar" else
                                  px.scatter(df, x=chart['x_col'], y=chart['y_col'], title=chart['title']) if chart['type'] == "Scatter" else
                                  go.Figure()),
                        dbc.Button(f"Remove Chart {i + 1}", id={"type": "remove_chart", "index": i}, color="danger", className="mb-2")
                    ]) for i, chart in enumerate(dashboard_charts)
                ])
            ])

            session['progress'] = session['progress'] | {'Visualize': 'Done'}
            session_token = save_auth_state(session, session.get('username'))
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Visualize']),
                    render_visualization_page(),
                    viz_output,
                    dashboard_output
                ]),
                session,
                f"?session_token={session_token}",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )
        except Exception as e:
            session['progress'] = session['progress'] | {'Visualize': 'Failed'}
            session_token = save_auth_state(session, session.get('username'))
            return (
                sidebar_layout(session),
                html.Div([
                    render_custom_header(page_titles['Visualize']),
                    render_visualization_page(),
                    html.Div(f"Error generating visualization: {str(e)}", className="text-danger")
                ]),
                session,
                f"?session_token={session_token}",
                config_controls,
                filter_controls,
                viz_output,
                dashboard_output,
                filter_options,
                target_options,
                feature_options,
                cluster_options,
                train_output,
                shap_output,
                lime_output,
                clustering_output,
                lime_max,
                download_data,
                columns_drop_options,
                encode_cols_options,
                enrich_col_options,
                anomaly_cols_options,
                target_col_ml_options,
                feature_cols_ml_options,
                custom_rules_container,
                smart_workflow_output,
                save_template_output,
                apply_template_output,
                cleaning_history,
                export_tableau_output,
                undo_disabled,
                redo_disabled,
                ai_suggestions_container,
                template_options,
                progress_text,
                health_score_figure,
                health_score_text,
                insights_output
            )

    # Handle dashboard chart removal
    elif "remove_chart" in trigger_id:
        idx = [i for i, n in enumerate(remove_chart_clicks) if n][0]
        session['dashboard_charts'].pop(idx)
        session_token = save_auth_state(session, session.get('username'))
        dashboard_charts = session.get('dashboard_charts', [])
        dashboard_output = html.Div([
            html.H4("Dashboard"),
            html.P("No charts added to dashboard yet.", className="text-info") if not dashboard_charts else html.Div([
                html.Div([
                    html.H5(f"Chart {i + 1}: {chart['title']}"),
                    dcc.Graph(figure=px.bar(df, x=chart['x_col'], y=chart['y_col'], title=chart['title']) if chart['type'] == "Bar" else
                              px.scatter(df, x=chart['x_col'], y=chart['y_col'], title=chart['title']) if chart['type'] == "Scatter" else
                              go.Figure()),
                    dbc.Button(f"Remove Chart {i + 1}", id={"type": "remove_chart", "index": i}, color="danger", className="mb-2")
                ]) for i, chart in enumerate(dashboard_charts)
            ])
        ])
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Visualize']),
                render_visualization_page(),
                dashboard_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle model training
    elif trigger_id == "train_model" and train_clicks:
        if not df or not target_col or not feature_cols:
            train_output = html.Div("Please select a target column and at least one feature column.", className="text-warning")
        else:
            try:
                model, score, explainer, shap_values, X_test = train_ml_model(
                    df, target_col, feature_cols, task_type, model_type=model_type
                )
                if model is None:
                    train_output = html.Div("Model training failed.", className="text-danger")
                else:
                    session.update({
                        'model': model,
                        'explainer': explainer,
                        'shap_values': shap_values,
                        'X_test': X_test,
                        'feature_cols': feature_cols,
                        'task_type': task_type,
                        'progress': session['progress'] | {'Predictive': 'Done'}
                    })
                    train_output = html.Div([
                        html.P(f"Model trained successfully! {task_type.capitalize()} score: {score:.2f}", className="text-success"),
                        html.H4("Fairness Metrics") if task_type == "classification" else None,
                        html.Div([
                            dcc.Graph(figure=px.imshow(
                                confusion_matrix(df.loc[X_test.index, target_col], model.predict(X_test)),
                                text_auto=True,
                                title="Confusion Matrix"
                            )),
                            html.H5("Classification Report"),
                            html.Pre(classification_report(df.loc[X_test.index, target_col], model.predict(X_test))),
                            html.H5("Demographic Parity (Prediction Rates by Gender)") if "gender" in df.columns else None,
                            html.P(str(pd.DataFrame({
                                'prediction': model.predict(X_test),
                                'gender': df.loc[X_test.index, "gender"]
                            }).groupby('gender')['prediction'].mean())) if "gender" in df.columns else None,
                            html.P("Potential fairness issue: Prediction rates differ significantly across groups.", className="text-warning")
                            if "gender" in df.columns and abs(pd.DataFrame({
                                'prediction': model.predict(X_test),
                                'gender': df.loc[X_test.index, "gender"]
                            }).groupby('gender')['prediction'].mean().diff().iloc[-1]) > 0.1 else None
                        ]) if task_type == "classification" else None
                    ])
                    lime_max = len(X_test) - 1
                    if SHAP_AVAILABLE and explainer:
                        shap_df = pd.DataFrame(shap_values, columns=feature_cols)
                        mean_shap = np.abs(shap_df).mean().sort_values(ascending=False)
                        shap_output = dcc.Graph(figure=px.bar(
                            x=mean_shap.values,
                            y=mean_shap.index,
                            orientation='h',
                            labels={'x': 'Mean |SHAP Value|', 'y': 'Feature'},
                            title="Feature Importance (Mean |SHAP Value|)",
                            color=mean_shap.values,
                            color_continuous_scale='Viridis'
                        ))
            except Exception as e:
                train_output = html.Div(f"Error during model training: {str(e)}.", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Predictive']),
                render_predictive_page(),
                train_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle clustering
    elif trigger_id == "run_clustering" and cluster_clicks:
        if not df or len(cluster_cols or []) < 2:
            clustering_output = html.Div("Please select at least two columns for clustering.", className="text-warning")
        else:
            try:
                labels = perform_clustering(df, cluster_cols, n_clusters)
                df['Cluster'] = labels
                session.update({
                    'cleaned_df': df,
                    'clustering_labels': labels,
                    'cluster_cols': cluster_cols,
                    'progress': session['progress'] | {'Predictive': 'Done'}
                })
                clustering_output = html.Div([
                    html.P("Clustering completed successfully!", className="text-success"),
                    dcc.Graph(figure=px.scatter(
                        df, x=cluster_cols[0], y=cluster_cols[1], color=labels.astype(str),
                        labels={'color': 'Cluster'}, title="Clustering Results (2D Scatter Plot)"
                    )) if len(cluster_cols) >= 2 else None,
                    dcc.Graph(figure=px.scatter_3d(
                        df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color=labels.astype(str),
                        labels={'color': 'Cluster'}, title="Clustering Results (3D Scatter Plot)"
                    )) if len(cluster_cols) >= 3 else None,
                    dcc.Graph(figure=px.bar(
                        x=pd.Series(labels).value_counts().sort_index().index.astype(str),
                        y=pd.Series(labels).value_counts().sort_index().values,
                        labels={'x': 'Cluster', 'y': 'Number of Points'},
                        title="Cluster Distribution"
                    ))
                ])
            except Exception as e:
                clustering_output = html.Div(f"Error performing clustering: {str(e)}.", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Predictive']),
                render_predictive_page(),
                clustering_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle LIME explanations
    elif trigger_id == "lime_sample_idx" and lime_sample_idx is not None:
        if not session.get('model') or not session.get('X_test'):
            lime_output = html.Div("No model trained yet.", className="text-warning")
        else:
            try:
                X_test = session['X_test']
                feature_cols = session['feature_cols']
                model = session['model']
                task_type = session['task_type']
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_test.values,
                    feature_names=feature_cols,
                    class_names=[str(i) for i in range(len(np.unique(df[target_col]))) if task_type == "classification"],
                    mode="classification" if task_type == "classification" else "regression"
                )
                instance = X_test.iloc[lime_sample_idx].values
                if task_type == "classification":
                    exp = lime_explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_cols))
                else:
                    exp = lime_explainer.explain_instance(instance, lambda x: model.predict(x).reshape(-1), num_features=len(feature_cols))
                fig, ax = plt.subplots()
                exp.as_pyplot_figure()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                lime_output = html.Img(src=f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}")
            except Exception as e:
                lime_output = html.Div(f"Error generating LIME explanations: {str(e)}.", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Predictive']),
                render_predictive_page()
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle visualization export
    elif trigger_id == "export_viz" and export_viz_clicks:
        if not fig:
            download_data = None
        else:
            try:
                buf = io.BytesIO()
                fig.write_image(buf, format=export_format.lower())
                download_data = dcc.send_bytes(buf.getvalue(), f"visualization.{export_format.lower()}")
            except ImportError:
                viz_output = html.Div("Kaleido library missing.", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Visualize']),
                render_visualization_page(),
                viz_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    # Handle filter application (numeric, datetime, categorical)
    elif trigger_id in ["filter_range", "filter_date", "filter_values"]:
        if not df or not filter_col or filter_col == "None":
            viz_output = html.Div("No filter applied.", className="text-warning")
        else:
            try:
                filtered_df = df.copy()
                if filter_range and trigger_id == "filter_range":
                    filtered_df = filtered_df[
                        (filtered_df[filter_col] >= filter_range[0]) &
                        (filtered_df[filter_col] <= filter_range[1])
                    ]
                    session['dashboard_filters'][filter_col] = filter_range
                elif filter_date_start and filter_date_end and trigger_id == "filter_date":
                    filtered_df = filtered_df[
                        (filtered_df[filter_col] >= pd.to_datetime(filter_date_start)) &
                        (filtered_df[filter_col] <= pd.to_datetime(filter_date_end))
                    ]
                    session['dashboard_filters'][filter_col] = (filter_date_start, filter_date_end)
                elif filter_values and trigger_id == "filter_values":
                    filtered_df = filtered_df[filtered_df[filter_col].isin(filter_values)]
                    session['dashboard_filters'][filter_col] = filter_values
                if filtered_df.empty:
                    viz_output = html.Div("Filters resulted in an empty dataset.", className="text-warning")
                else:
                    session['filtered_df'] = filtered_df
                    viz_output = html.P(f"Filtered dataset: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
            except Exception as e:
                viz_output = html.Div(f"Error applying filter: {str(e)}", className="text-danger")
        session_token = save_auth_state(session, session.get('username'))
        return (
            sidebar_layout(session),
            html.Div([
                render_custom_header(page_titles['Visualize']),
                render_visualization_page(),
                viz_output
            ]),
            session,
            f"?session_token={session_token}",
            config_controls,
            filter_controls,
            viz_output,
            dashboard_output,
            filter_options,
            target_options,
            feature_options,
            cluster_options,
            train_output,
            shap_output,
            lime_output,
            clustering_output,
            lime_max,
            download_data,
            columns_drop_options,
            encode_cols_options,
            enrich_col_options,
            anomaly_cols_options,
            target_col_ml_options,
            feature_cols_ml_options,
            custom_rules_container,
            smart_workflow_output,
            save_template_output,
            apply_template_output,
            cleaning_history,
            export_tableau_output,
            undo_disabled,
            redo_disabled,
            ai_suggestions_container,
            template_options,
            progress_text,
            health_score_figure,
            health_score_text,
            insights_output
        )

    raise PreventUpdate
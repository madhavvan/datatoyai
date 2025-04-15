import logging
import os
import pickle
import uuid
import base64
import io
import pandas as pd
import numpy as np
import bcrypt
import psycopg2
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
from dash import Dash, html, dcc, Input, Output, State, callback, no_update, dash_table
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
from authlib.integrations.requests_client import OAuth2Session
import matplotlib.pyplot as plt
from io import BytesIO
import pyarrow.parquet as pq
from data_utils_dash import (
    AI_AVAILABLE, chat_with_gpt, get_cleaning_suggestions, get_insights,
    apply_cleaning_operations, calculate_health_score, forecast_time_series,
    perform_clustering, generate_synthetic_data, detect_anomalies,
    train_ml_model, suggest_visualization
)
from ui_dash import render_upload_page, render_clean_page, render_insights_page, render_predictive_page
from visualizations_dash import render_visualization_page
from predictive_dash import st_shap

# Initialize Dash app
app = Dash(__name__, external_stylesheets=['/assets/styles.css'], title="Data Toy")
server = app.server  # For Azure deployment

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=3)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://your-azure-app-url.azurewebsites.net/callback")
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = ["openid", "email", "profile"]

# Session state
session_state = {
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
    'logs': [],
    'suggestions': [],
    'previous_states': [],
    'redo_states': [],
    'cleaning_history': [],
    'cleaning_templates': {},
    'is_premium': False,
    'ai_suggestions_used': 0,
    'dropped_columns': [],
    'dashboard_charts': [],
    'dashboard_filters': {},
    'filtered_df': None,
    'clustering_labels': None,
    'cluster_cols': [],
    'model': None,
    'explainer': None,
    'shap_values': None,
    'X_test': None,
    'feature_cols': [],
    'task_type': 'classification'
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
    c.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sessions')")
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
            return
        c = conn.cursor()
        try:
            c.execute("SELECT username, session_data FROM sessions WHERE session_token = %s", (session_token,))
            result = c.fetchone()
            if result:
                username, session_data = result
                session_data = pickle.loads(session_data)
                session_state.update({
                    'authenticated': session_data.get('authenticated', False),
                    'username': username,
                    'user_info': session_data.get('user_info', None),
                    'session_token': session_token,
                    'page': session_data.get('page', "Upload"),
                    'df': session_data.get('df'),
                    'cleaned_df': session_data.get('cleaned_df'),
                    'logs': session_data.get('logs', []),
                    'suggestions': session_data.get('suggestions', []),
                    'previous_states': session_data.get('previous_states', []),
                    'redo_states': session_data.get('redo_states', []),
                    'chat_history': session_data.get('chat_history', []),
                    'cleaning_history': session_data.get('cleaning_history', []),
                    'cleaning_templates': session_data.get('cleaning_templates', {}),
                    'is_premium': session_data.get('is_premium', False),
                    'ai_suggestions_used': session_data.get('ai_suggestions_used', 0),
                    'dropped_columns': session_data.get('dropped_columns', []),
                    'progress': session_data.get('progress', session_state['progress']),
                    'dashboard_charts': session_data.get('dashboard_charts', []),
                    'dashboard_filters': session_data.get('dashboard_filters', {}),
                    'filtered_df': session_data.get('filtered_df'),
                    'clustering_labels': session_data.get('clustering_labels'),
                    'cluster_cols': session_data.get('cluster_cols', []),
                    'model': session_data.get('model'),
                    'explainer': session_data.get('explainer'),
                    'shap_values': session_data.get('shap_values'),
                    'X_test': session_data.get('X_test'),
                    'feature_cols': session_data.get('feature_cols', []),
                    'task_type': session_data.get('task_type', 'classification')
                })
                logger.info(f"Session restored for user {username}")
            else:
                logger.debug("No session found for the given session token")
        except Exception as e:
            logger.error(f"Error in restore_session: {str(e)}")
        finally:
            conn.close()

def save_auth_state():
    if session_state['username']:
        logger.debug("Starting save_auth_state")
        if not session_state['session_token']:
            session_state['session_token'] = str(uuid.uuid4())
            logger.debug(f"Generated new session token: {session_state['session_token']}")
        session_data = {
            'authenticated': session_state['authenticated'],
            'username': session_state['username'],
            'user_info': session_state['user_info'],
            'page': session_state['page'],
            'df': session_state.get('df'),
            'cleaned_df': session_state.get('cleaned_df'),
            'logs': session_state.get('logs'),
            'suggestions': session_state.get('suggestions'),
            'previous_states': session_state.get('previous_states'),
            'redo_states': session_state.get('redo_states'),
            'chat_history': session_state.get('chat_history'),
            'cleaning_history': session_state.get('cleaning_history'),
            'cleaning_templates': session_state.get('cleaning_templates'),
            'is_premium': session_state.get('is_premium'),
            'ai_suggestions_used': session_state.get('ai_suggestions_used'),
            'dropped_columns': session_state.get('dropped_columns'),
            'progress': session_state.get('progress'),
            'dashboard_charts': session_state.get('dashboard_charts'),
            'dashboard_filters': session_state.get('dashboard_filters'),
            'filtered_df': session_state.get('filtered_df'),
            'clustering_labels': session_state.get('clustering_labels'),
            'cluster_cols': session_state.get('cluster_cols'),
            'model': session_state.get('model'),
            'explainer': session_state.get('explainer'),
            'shap_values': session_state.get('shap_values'),
            'X_test': session_state.get('X_test'),
            'feature_cols': session_state.get('feature_cols'),
            'task_type': session_state.get('task_type')
        }
        session_blob = pickle.dumps(session_data)
        conn = get_db_connection()
        if conn is None:
            logger.debug("Failed to connect to database in save_auth_state")
            return
        c = conn.cursor()
        try:
            c.execute("INSERT INTO sessions (username, session_token, session_data) VALUES (%s, %s, %s) ON CONFLICT (username) DO UPDATE SET session_token = %s, session_data = %s",
                      (session_state['username'], session_state['session_token'], session_blob, session_state['session_token'], session_blob))
            conn.commit()
            logger.info("Session state saved successfully")
        except Exception as e:
            logger.error(f"Error in save_auth_state: {str(e)}")
        finally:
            conn.close()

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
    c.execute("SELECT username, email, name, profile_picture FROM users WHERE google_id = %s", (google_id,))
    result = c.fetchone()
    conn.close()
    return result

def save_session(username):
    save_auth_state()

def load_session(username):
    conn = get_db_connection()
    if conn is None:
        return
    c = conn.cursor()
    c.execute("SELECT session_data FROM sessions WHERE username = %s", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        session_data = pickle.loads(result[0])
        for key, value in session_data.items():
            if key not in ['authenticated', 'username', 'user_info', 'session_token', 'page']:
                session_state[key] = value

def load_css(theme: str = "dark") -> str:
    css = """
    body {
        font-family: 'Roboto', sans-serif !important;
        margin: 0;
        padding: 0;
    }

    .dark-theme {
        background: linear-gradient(to bottom right, #1C2526, #2A3B47) !important;
        color: #FFFFFF !important;
    }

    .dark-theme h1 {
        color: #1E90FF !important;
        font-family: 'Roboto', sans-serif !important;
    }

    .dark-theme h2, .dark-theme h3 {
        color: #FFD700 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    .dark-theme button {
        background-color: #1E90FF !important;
        color: white !important;
        border-radius: 5px !important;
        transition: background-color 0.3s !important;
        font-family: 'Roboto', sans-serif !important;
        border: none !important;
        padding: 10px 20px !important;
    }

    .dark-theme button:hover {
        background-color: #FFD700 !important;
        color: #1C2526 !important;
    }

    .dark-theme .dash-table-container {
        background-color: #2A3B47 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    .dark-theme input, .dark-theme select, .dark-theme textarea {
        background-color: #2A3B47 !important;
        color: #FFFFFF !important;
        border: 1px solid #1E90FF !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }

    .dark-theme .Select-control, .dark-theme .Select-menu-outer {
        background-color: #2A3B47 !important;
        color: #FFFFFF !important;
        border: 1px solid #1E90FF !important;
    }

    .light-theme {
        background: linear-gradient(to bottom right, #F0F4F8, #D9E2EC) !important;
        color: #000000 !important;
    }

    .light-theme h1 {
        color: #0066CC !important;
        font-family: 'Roboto', sans-serif !important;
    }

    .light-theme h2, .light-theme h3 {
        color: #CC9900 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    .light-theme button {
        background-color: #0066CC !important;
        color: white !important;
        border-radius: 5px !important;
        transition: background-color 0.3s !important;
        font-family: 'Roboto', sans-serif !important;
        border: none !important;
        padding: 10px 20px !important;
    }

    .light-theme button:hover {
        background-color: #CC9900 !important;
        color: #FFFFFF !important;
    }

    .light-theme .dash-table-container {
        background-color: #F0F4F8 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    .light-theme input, .light-theme select, .light-theme textarea {
        background-color: #F0F4F8 !important;
        color: #000000 !important;
        border: 1px solid #0066CC !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }

    .light-theme .Select-control, .light-theme .Select-menu-outer {
        background-color: #F0F4F8 !important;
        color: #000000 !important;
        border: 1px solid #0066CC !important;
    }

    .google-login-button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #FFFFFF !important;
        color: #757575 !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        font-size: 16px !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease, box-shadow 0.3s ease !important;
        width: 100% !important;
        box-sizing: border-box !important;
        margin: 10px auto !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    }

    .google-login-button:hover {
        background-color: #F8FAFC !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    .google-login-button img {
        width: 20px !important;
        height: 20px !important;
        margin-right: 10px !important;
    }

    .google-login-button span {
        color: #757575 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    a.google-login-button {
        text-decoration: none !important;
    }

    .login-card {
        background: #2A3B47;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        max-width: 400px;
        margin: 100px auto;
    }

    .light-theme .login-card {
        background: #FFFFFF;
    }

    .sidebar {
        width: 250px;
        padding: 20px;
        background: #1C2526;
        height: 100vh;
        position: fixed;
    }

    .light-theme .sidebar {
        background: #D9E2EC;
    }

    .content {
        margin-left: 270px;
        padding: 20px;
    }
    """
    if not os.path.exists('assets'):
        os.makedirs('assets')
    with open('assets/styles.css', 'w') as f:
        f.write(css)
    return css

def render_custom_header(page_title: str) -> html.Div:
    return html.Div([
        html.H1(page_title, style={'margin-top': '20px'}),
        html.Hr(style={'border': '1px solid #FFD700'})
    ])

def get_google_auth_url():
    client = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, redirect_uri=GOOGLE_REDIRECT_URI, scope=SCOPES)
    auth_url, state = client.create_authorization_url(GOOGLE_AUTH_URL)
    session_state['oauth_state'] = state
    return auth_url

def handle_google_callback(code):
    try:
        client = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, redirect_uri=GOOGLE_REDIRECT_URI, state=session_state.get('oauth_state'))
        token = client.fetch_token(GOOGLE_TOKEN_URL, code=code)
        user_info = requests.get(GOOGLE_USERINFO_URL, headers={'Authorization': f"Bearer {token['access_token']}"}).json()
        if 'error' in user_info:
            logger.error(f"Google OAuth error: {user_info['error']}")
            return None
        return user_info
    except Exception as e:
        logger.error(f"Error during Google OAuth callback: {str(e)}")
        return None

load_css(session_state['theme'])

def setup_sidebar(logo_path: str = "images/datatoy_logo.png") -> html.Div:
    try:
        logo = html.Img(src=logo_path, style={'width': '100%'})
    except Exception:
        logo = html.Div("**Data Toy** (Logo not found)")
        logger.warning(f"Logo file '{logo_path}' not found.")

    sidebar_content = [
        logo,
        html.Hr(),
        html.H3("Navigation"),
        html.P("Transform your data with AI magic.", className='tagline'),
        dcc.RadioItems(
            id='sidebar-page',
            options=[
                {'label': 'Upload', 'value': 'Upload'},
                {'label': 'Clean', 'value': 'Clean'},
                {'label': 'Insights', 'value': 'Insights'},
                {'label': 'Visualize', 'value': 'Visualize'},
                {'label': 'Predictive', 'value': 'Predictive'},
                {'label': 'Share', 'value': 'Share'}
            ],
            value=session_state['page'],
            style={'color': '#FFFFFF' if session_state['theme'] == 'dark' else '#000000'}
        ),
        html.H4("Theme"),
        dcc.Dropdown(
            id='theme-select',
            options=[
                {'label': 'Dark', 'value': 'dark'},
                {'label': 'Light', 'value': 'light'}
            ],
            value=session_state['theme'],
            clearable=False
        ),
        html.H4("Your Progress"),
        html.P(id='progress-text'),
        html.P("⚠️ AI features are disabled. Please configure an OPENAI_API_KEY." if not AI_AVAILABLE else "", id='ai-warning'),
        html.H4("AI Data Assistant"),
        html.Div(id='chat-container'),
        dcc.Textarea(id='chat-input', placeholder='Ask Data Toy', style={'width': '100%'}),
        html.Button('Send', id='chat-submit'),
        html.Hr(),
        html.H4("Feedback"),
        html.A("Share your feedback", href="https://docs.google.com/forms/d/e/1FAIpQLScpUFM0Y5_i5LJDM-HZEZEtOHbLHy4Vp-ek_-819MRZo7Q9rQ/viewform?usp=dialog"),
        html.H4("Join Our Community"),
        html.A("Join our Discord", href="https://discord.gg/your-invite-link"),
        html.H4("Upgrade to Premium"),
        html.A("Upgrade Now", href="https://stripe.com/your-checkout-link"),
        html.P("Running in DEV_MODE: Unlimited AI suggestions enabled." if os.getenv("DEV_MODE") == "true" else "", id='dev-mode'),
        html.Button('Logout', id='logout-button')
    ]
    return html.Div(sidebar_content, className='sidebar')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session-store', data=session_state),
    html.Div(id='page-content', className=f"{session_state['theme']}-theme")
])

@app.callback(
    Output('session-store', 'data', allow_duplicate=True),
    Input('url', 'search'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_query_params(search, stored_state):
    global session_state
    session_state.update(stored_state)
    session_token = None
    if 'session_token=' in search:
        session_token = search.split('session_token=')[1].split('&')[0]
        restore_session(session_token)
    return session_state

@app.callback(
    Output('page-content', 'children'),
    Output('session-store', 'data'),
    Output('progress-text', 'children'),
    Output('chat-container', 'children'),
    Output('ai-warning', 'children'),
    Output('dev-mode', 'children'),
    Input('url', 'pathname'),
    Input('sidebar-page', 'value'),
    Input('theme-select', 'value'),
    Input('logout-button', 'n_clicks'),
    Input('chat-submit', 'n_clicks'),
    State('chat-input', 'value'),
    State('session-store', 'data')
)
def update_page(pathname, sidebar_page, theme, logout_clicks, chat_clicks, chat_input, stored_state):
    global session_state
    session_state.update(stored_state)

    if logout_clicks:
        conn = get_db_connection()
        if conn:
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE username = %s", (session_state['username'],))
            conn.commit()
            conn.close()
        session_state.update({
            'authenticated': False,
            'username': None,
            'user_info': None,
            'session_token': None,
            'page': 'Login',
            'df': None,
            'cleaned_df': None,
            'logs': [],
            'suggestions': [],
            'previous_states': [],
            'redo_states': [],
            'cleaning_history': [],
            'cleaning_templates': {},
            'is_premium': False,
            'ai_suggestions_used': 0,
            'dropped_columns': [],
            'dashboard_charts': [],
            'dashboard_filters': {},
            'filtered_df': None,
            'clustering_labels': None,
            'cluster_cols': [],
            'model': None,
            'explainer': None,
            'shap_values': None,
            'X_test': None,
            'feature_cols': [],
            'task_type': 'classification'
        })
        return html.Div(id='login-page'), session_state, "", [], "", ""

    if theme and theme != session_state['theme']:
        session_state['theme'] = theme
        save_auth_state()

    if sidebar_page and sidebar_page != session_state['page']:
        session_state['page'] = sidebar_page
        save_auth_state()

    if chat_clicks and chat_input:
        df = session_state.get('cleaned_df') or session_state.get('df')
        if df is not None:
            session_state['chat_history'].append({"role": "user", "content": chat_input})
            try:
                response = chat_with_gpt(df, chat_input, max_tokens=100)
                session_state['chat_history'].append({"role": "assistant", "content": response})
                save_auth_state()
            except Exception as e:
                logger.error(f"Chat error: {str(e)}")
                session_state['chat_history'].append({"role": "assistant", "content": f"Error: {str(e)}"})
        else:
            session_state['chat_history'].append({"role": "assistant", "content": "Please upload a dataset first to use the AI assistant."})

    if not session_state['authenticated']:
        login_layout = html.Div([
            html.H1("Welcome to Data Toy AI", style={'text-align': 'center', 'margin-bottom': '20px', 'font-size': '24px'}),
            dcc.Input(id='username-input', placeholder='Enter your username', type='text', style={'width': '100%', 'margin': '10px 0'}),
            dcc.Input(id='password-input', placeholder='Enter your password', type='password', style={'width': '100%', 'margin': '10px 0'}),
            html.Button('Login', id='login-button'),
            html.A(
                html.Button([
                    html.Img(src="https://developers.google.com/identity/images/g-logo.png", style={'width': '20px', 'margin-right': '10px'}),
                    html.Span("Sign in with Google")
                ], className='google-login-button'),
                href=get_google_auth_url()
            ),
            html.Button('Sign Up', id='signup-button'),
            html.Div(id='login-error')
        ], className='login-card')
        progress_text = ""
        chat_container = []
        ai_warning = "⚠️ AI features are disabled. Please configure an OPENAI_API_KEY." if not AI_AVAILABLE else ""
        dev_mode = "Running in DEV_MODE: Unlimited AI suggestions enabled." if os.getenv("DEV_MODE") == "true" else ""
        return login_layout, session_state, progress_text, chat_container, ai_warning, dev_mode

    if session_state['page'] == 'Sign Up':
        signup_layout = html.Div([
            html.H1("Sign Up for Data Toy AI", style={'text-align': 'center', 'margin-bottom': '20px', 'font-size': '24px'}),
            dcc.Input(id='new-username-input', placeholder='Choose a username', type='text', style={'width': '100%', 'margin': '10px 0'}),
            dcc.Input(id='new-email-input', placeholder='Enter your email', type='email', style={'width': '100%', 'margin': '10px 0'}),
            dcc.Input(id='new-name-input', placeholder='Enter your name', type='text', style={'width': '100%', 'margin': '10px 0'}),
            dcc.Input(id='new-password-input', placeholder='Choose a password', type='password', style={'width': '100%', 'margin': '10px 0'}),
            html.Button('Register', id='register-button'),
            html.Button('Back to Login', id='back-to-login-button'),
            html.Div(id='signup-error')
        ], className='login-card')
        progress_text = ""
        chat_container = []
        ai_warning = "⚠️ AI features are disabled. Please configure an OPENAI_API_KEY." if not AI_AVAILABLE else ""
        dev_mode = "Running in DEV_MODE: Unlimited AI suggestions enabled." if os.getenv("DEV_MODE") == "true" else ""
        return signup_layout, session_state, progress_text, chat_container, ai_warning, dev_mode

    sidebar = setup_sidebar()
    content = html.Div([
        render_custom_header({
            "Upload": "Upload Your Dataset",
            "Clean": "Clean Your Dataset",
            "Insights": "Insights Dashboard",
            "Visualize": "Visualize Your Dataset",
            "Predictive": "Predictive Analytics",
            "Share": "Share Your Work"
        }.get(session_state['page'], "Data Toy")),
        html.Div(id=f"{session_state['page']}-content")
    ], className='content')

    progress_text = "".join(
        f"{emoji} {step}: {status}\n"
        for step, status in session_state['progress'].items()
        for emoji in ["✅" if status == "Done" else "🟡" if status == "In Progress" else "⬜"]
    )
    chat_container = [
        html.Div(
            f"{message['role'].capitalize()}: {message['content']}",
            style={'background': '#2A3B47' if session_state['theme'] == 'dark' else '#F0F4F8', 'padding': '10px', 'margin': '5px', 'border-radius': '5px'}
        )
        for message in session_state['chat_history']
    ]
    ai_warning = "⚠️ AI features are disabled. Please configure an OPENAI_API_KEY." if not AI_AVAILABLE else ""
    dev_mode = "Running in DEV_MODE: Unlimited AI suggestions enabled." if os.getenv("DEV_MODE") == "true" else ""

    return html.Div([sidebar, content], className=f"{session_state['theme']}-theme"), session_state, progress_text, chat_container, ai_warning, dev_mode

@app.callback(
    Output('login-error', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('login-button', 'n_clicks'),
    Input('signup-button', 'n_clicks'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_login(login_clicks, signup_clicks, username, password, stored_state):
    global session_state
    session_state.update(stored_state)
    if login_clicks:
        if verify_user(username, password):
            session_state.update({
                'authenticated': True,
                'username': username,
                'page': 'Upload',
                'user_info': None
            })
            load_session(username)
            save_auth_state()
            return None, session_state
        return "Incorrect username or password", no_update
    if signup_clicks:
        session_state['page'] = 'Sign Up'
        save_auth_state()
        return None, session_state
    raise PreventUpdate

@app.callback(
    Output('signup-error', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('register-button', 'n_clicks'),
    Input('back-to-login-button', 'n_clicks'),
    State('new-username-input', 'value'),
    State('new-email-input', 'value'),
    State('new-name-input', 'value'),
    State('new-password-input', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_signup(register_clicks, back_clicks, username, email, name, password, stored_state):
    global session_state
    session_state.update(stored_state)
    if register_clicks:
        if add_user(username, email, name, password):
            session_state['page'] = 'Login'
            save_auth_state()
            return "Registration successful! Please log in.", session_state
        return "Username already exists. Please choose a different username.", no_update
    if back_clicks:
        session_state['page'] = 'Login'
        save_auth_state()
        return None, session_state
    raise PreventUpdate

@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Output('session-store', 'data', allow_duplicate=True),
    Input('url', 'search'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_google_callback(search, stored_state):
    global session_state
    session_state.update(stored_state)
    if 'code=' in search:
        code = search.split('code=')[1].split('&')[0]
        user_info = handle_google_callback(code)
        if user_info is None:
            login_layout = html.Div([
                html.H1("Welcome to Data Toy AI"),
                html.P("Failed to authenticate with Google. Please try again."),
                dcc.Input(id='username-input', placeholder='Enter your username', type='text'),
                dcc.Input(id='password-input', placeholder='Enter your password', type='password'),
                html.Button('Login', id='login-button'),
                html.A(
                    html.Button([
                        html.Img(src="https://developers.google.com/identity/images/g-logo.png", style={'width': '20px', 'margin-right': '10px'}),
                        html.Span("Sign in with Google")
                    ], className='google-login-button'),
                    href=get_google_auth_url()
                ),
                html.Button('Sign Up', id='signup-button'),
                html.Div(id='login-error')
            ], className='login-card')
            return login_layout, no_update
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
        session_state.update({
            'authenticated': True,
            'username': username,
            'user_info': user_info,
            'page': 'Upload'
        })
        load_session(username)
        save_auth_state()
        sidebar = setup_sidebar()
        content = html.Div([
            render_custom_header("Upload Your Dataset"),
            html.Div(id='Upload-content')
        ], className='content')
        return html.Div([sidebar, content], className=f"{session_state['theme']}-theme"), session_state
    return no_update, no_update

# Upload Page Callbacks
@app.callback(
    Output('Upload-content', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def render_upload_page_callback(sidebar_page, stored_state):
    if sidebar_page != 'Upload':
        return no_update
    global session_state
    session_state.update(stored_state)
    try:
        return render_upload_page()
    except Exception as e:
        logger.error(f"Error rendering Upload page: {str(e)}")
        return html.Div(f"An error occurred: {str(e)}")

@app.callback(
    Output('upload-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('file-uploader', 'contents'),
    Input('file-uploader', 'filename'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename, stored_state):
    global session_state
    session_state.update(stored_state)
    session_state['progress']['Upload'] = "In Progress"
    if contents is None:
        return no_update, no_update

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.parquet'):
            df = pq.read_table(io.BytesIO(decoded)).to_pandas()
        else:
            return html.P("Unsupported file type."), no_update

        if df.empty:
            return html.P("Uploaded dataset is empty. Please upload a valid file."), no_update

        session_state.update({
            'df': df,
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
            'filtered_df': df.copy(),
            'clustering_labels': None,
            'cluster_cols': [],
            'model': None,
            'explainer': None,
            'shap_values': None,
            'X_test': None,
            'feature_cols': [],
            'progress': session_state['progress'] | {'Upload': 'Done'}
        })
        save_auth_state()
        return html.P("Dataset uploaded successfully!"), session_state
    except Exception as e:
        session_state['progress']['Upload'] = "Failed"
        return html.P(f"Error loading file: {str(e)}"), session_state

# Clean Page Callbacks
@app.callback(
    Output('Clean-content', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def render_clean_page_callback(sidebar_page, stored_state):
    if sidebar_page != 'Clean':
        return no_update
    global session_state
    session_state.update(stored_state)
    try:
        return render_clean_page()
    except Exception as e:
        logger.error(f"Error rendering Clean page: {str(e)}")
        return html.Div(f"An error occurred: {str(e)}")

@app.callback(
    Output('custom-rules-container', 'children'),
    Input('num-rules', 'value'),
    State('session-store', 'data')
)
def update_custom_rules(num_rules, stored_state):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return []
    available_columns = [col for col in df.columns if col not in session_state.get('dropped_columns', [])]
    rules = []
    for i in range(num_rules):
        rule_layout = html.Div([
            html.H4(f"Rule {i + 1}"),
            dcc.Dropdown(
                id={'type': 'rule-col', 'index': i},
                options=[{'label': col, 'value': col} for col in available_columns],
                placeholder="Select column"
            ),
            dcc.Dropdown(
                id={'type': 'rule-cond', 'index': i},
                options=[
                    {'label': 'greater than', 'value': 'greater than'},
                    {'label': 'less than', 'value': 'less than'},
                    {'label': 'equal to', 'value': 'equal to'}
                ],
                placeholder="Select condition"
            ),
            dcc.Input(id={'type': 'rule-threshold', 'index': i}, type='number', value=0.0),
            dcc.Dropdown(
                id={'type': 'rule-action', 'index': i},
                options=[
                    {'label': 'Set to NaN', 'value': 'Set to NaN'},
                    {'label': 'Set to Value', 'value': 'Set to Value'}
                ],
                placeholder="Select action"
            ),
            dcc.Input(id={'type': 'rule-action-value', 'index': i}, type='number', value=0.0, style={'display': 'none'})
        ])
        rules.append(rule_layout)
    return rules

@app.callback(
    Output({'type': 'rule-action-value', 'index': ALL}, 'style'),
    Input({'type': 'rule-action', 'index': ALL}, 'value')
)
def toggle_action_value_visibility(actions):
    return [{'display': 'block' if action == 'Set to Value' else 'none'} for action in actions]

@app.callback(
    Output('ai-suggestions-container', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def update_ai_suggestions(sidebar_page, stored_state):
    if sidebar_page != 'Clean':
        return no_update
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return html.P("No dataset available.")
    available_columns = [col for col in df.columns if col not in session_state.get('dropped_columns', [])]
    try:
        suggestions = get_cleaning_suggestions(df[available_columns])
        suggestion_elements = []
        for idx, (suggestion, explanation) in enumerate(suggestions):
            if "Based on the provided dataset analysis" in suggestion:
                suggestion_elements.append(html.P(f"{suggestion} - {explanation}"))
            else:
                element = [
                    dcc.Checklist(
                        id={'type': 'suggestion', 'index': idx},
                        options=[{'label': suggestion, 'value': suggestion}],
                        value=[],
                        labelStyle={'display': 'block'}
                    ),
                    html.P(f"Explanation: {explanation}")
                ]
                if "Handle special characters" in suggestion:
                    element.append(dcc.RadioItems(
                        id={'type': 'special-chars-opt', 'index': idx},
                        options=[
                            {'label': 'Drop them', 'value': 'Drop them'},
                            {'label': 'Replace with underscores', 'value': 'Replace with underscores'}
                        ],
                        value='Drop them'
                    ))
                elif "Fill missing values" in suggestion:
                    col = extract_column(suggestion)
                    if col and col in available_columns and df[col].dtype in ['int64', 'float64']:
                        element.append(dcc.RadioItems(
                            id={'type': 'fill-opt', 'index': idx},
                            options=[
                                {'label': 'mean', 'value': 'mean'},
                                {'label': 'median', 'value': 'median'},
                                {'label': 'mode', 'value': 'mode'}
                            ],
                            value='mode'
                        ))
                elif "Handle outliers" in suggestion:
                    col = extract_column(suggestion)
                    if col and col in available_columns:
                        element.append(dcc.RadioItems(
                            id={'type': 'outlier-opt', 'index': idx},
                            options=[
                                {'label': 'Remove', 'value': 'Remove'},
                                {'label': 'Cap at bounds', 'value': 'Cap at bounds'}
                            ],
                            value='Remove'
                        ))
                suggestion_elements.append(html.Div(element))
        session_state['suggestions'] = suggestions
        return suggestion_elements
    except Exception as e:
        logger.error(f"Error generating AI suggestions: {str(e)}")
        return html.P(f"Error generating AI suggestions: {str(e)}")

@app.callback(
    Output('cleaning-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('preview-button', 'n_clicks'),
    Input('apply-button', 'n_clicks'),
    Input('auto-clean-button', 'n_clicks'),
    Input('save-template-button', 'n_clicks'),
    Input('apply-template-button', 'n_clicks'),
    Input('export-tableau-button', 'n_clicks'),
    State('columns-to-drop', 'value'),
    State({'type': 'suggestion', 'index': ALL}, 'value'),
    State({'type': 'special-chars-opt', 'index': ALL}, 'value'),
    State({'type': 'fill-opt', 'index': ALL}, 'value'),
    State({'type': 'outlier-opt', 'index': ALL}, 'value'),
    State('replace-value', 'value'),
    State('replace-with', 'value'),
    State('replace-with-custom', 'value'),
    State('replace-scope', 'value'),
    State('encode-cols', 'value'),
    State('encode-method', 'value'),
    State('enrich-col', 'value'),
    State('enrich-api-key', 'value'),
    State('target-col', 'value'),
    State('feature-cols', 'value'),
    State('train-ml', 'value'),
    State({'type': 'rule-col', 'index': ALL}, 'value'),
    State({'type': 'rule-cond', 'index': ALL}, 'value'),
    State({'type': 'rule-threshold', 'index': ALL}, 'value'),
    State({'type': 'rule-action', 'index': ALL}, 'value'),
    State({'type': 'rule-action-value', 'index': ALL}, 'value'),
    State('template-name', 'value'),
    State('apply-template', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_cleaning_operations(
    preview_clicks, apply_clicks, auto_clean_clicks, save_template_clicks, apply_template_clicks, export_tableau_clicks,
    columns_to_drop, suggestion_values, special_chars_opts, fill_opts, outlier_opts, replace_value, replace_with, replace_with_custom,
    replace_scope, encode_cols, encode_method, enrich_col, enrich_api_key, target_col, feature_cols, train_ml,
    rule_cols, rule_conds, rule_thresholds, rule_actions, rule_action_values, template_name, apply_template, stored_state
):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return html.P("No dataset available."), no_update

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'export-tableau-button' and export_tableau_clicks:
        if session_state.get('cleaned_df') is not None:
            filename = f"cleaned_for_tableau_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv = session_state['cleaned_df'].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return html.A(f"Download {filename}", href=f"data:file/csv;base64,{b64}", download=filename), no_update
        return html.P("No cleaned dataset available for export."), no_update

    selected_suggestions = []
    options = {}
    for idx, (suggestion, explanation) in enumerate(session_state.get('suggestions', [])):
        if suggestion_values[idx] and suggestion_values[idx]:
            selected_suggestions.append((suggestion, explanation))
            if "Handle special characters" in suggestion and special_chars_opts[idx]:
                options["special_chars"] = special_chars_opts[idx]
            elif "Fill missing values" in suggestion and fill_opts[idx]:
                col = extract_column(suggestion)
                if col:
                    options[f"fill_{col}"] = fill_opts[idx]
            elif "Handle outliers" in suggestion and outlier_opts[idx]:
                col = extract_column(suggestion)
                if col:
                    options[f"outlier_{col}"] = outlier_opts[idx]

    custom_rules = []
    for i in range(len(rule_cols)):
        if rule_cols[i] and rule_conds[i] and rule_actions[i]:
            custom_rules.append({
                "column": rule_cols[i],
                "condition": rule_conds[i],
                "threshold": rule_thresholds[i],
                "action": rule_actions[i],
                "action_value": rule_action_values[i] if rule_actions[i] == "Set to Value" else None
            })

    if replace_with == 'Custom':
        replace_with = replace_with_custom

    if triggered_id == 'save-template-button' and save_template_clicks and template_name:
        template = {
            "columns_to_drop": columns_to_drop or [],
            "selected_suggestions": selected_suggestions,
            "options": options,
            "replace_value": replace_value or "",
            "replace_with": replace_with or "",
            "replace_scope": replace_scope or "All columns",
            "encode_cols": encode_cols or [],
            "encode_method": encode_method or "Label Encoding",
            "enrich_col": enrich_col or "None",
            "train_ml": bool(train_ml),
            "target_col": target_col,
            "feature_cols": feature_cols or [],
            "custom_rules": custom_rules
        }
        session_state['cleaning_templates'][template_name] = template
        save_auth_state()
        return html.P(f"Saved template '{template_name}'"), session_state

    if triggered_id == 'apply-template-button' and apply_template_clicks and apply_template != 'None':
        template = session_state['cleaning_templates'].get(apply_template, {})
        try:
            cleaned_df, logs = apply_cleaning_operations(
                df,
                selected_suggestions=template.get("selected_suggestions", []),
                columns_to_drop=template.get("columns_to_drop", []),
                options=template.get("options", {}),
                replace_value=template.get("replace_value", ""),
                replace_with=template.get("replace_with", ""),
                replace_scope=template.get("replace_scope", "All columns"),
                encode_cols=template.get("encode_cols", []),
                encode_method=template.get("encode_method", "Label Encoding"),
                auto_clean=False,
                enrich_col=template.get("enrich_col"),
                enrich_api_key=enrich_api_key,
                train_ml=template.get("train_ml", False),
                target_col=template.get("target_col"),
                feature_cols=template.get("feature_cols", [])
            )
            session_state['previous_states'].append((df.copy(), session_state['logs'].copy()))
            if len(session_state['previous_states']) > 5:
                session_state['previous_states'].pop(0)
            session_state['redo_states'] = []
            session_state['cleaned_df'] = cleaned_df
            session_state['logs'] = logs
            if template.get("columns_to_drop"):
                session_state['dropped_columns'].extend(template.get("columns_to_drop"))
            session_state['cleaning_history'].append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": logs + [f"Applied template '{apply_template}'"]
            })
            session_state['suggestions'] = get_cleaning_suggestions(
                cleaned_df[[col for col in cleaned_df.columns if col not in session_state['dropped_columns']]]
            )
            session_state['progress']['Clean'] = "Done"
            save_auth_state()
            return html.Div([
                html.H3("Applied Template"),
                html.Ul([html.Li(log) for log in logs]),
                dash_table.DataTable(
                    data=cleaned_df.head(10).to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in cleaned_df.columns],
                    style_table={'overflowX': 'auto'}
                )
            ]), session_state
        except Exception as e:
            session_state['progress']['Clean'] = "Failed"
            return html.P(f"Error applying template: {str(e)}"), session_state

    if preview_clicks or apply_clicks or auto_clean_clicks:
        operations_selected = (
            selected_suggestions or columns_to_drop or (replace_value and replace_with) or
            encode_cols or (enrich_col != "None" and enrich_api_key) or auto_clean_clicks or
            (train_ml and target_col and feature_cols) or custom_rules
        )
        if not operations_selected:
            return html.P("Please select at least one cleaning operation, custom rule, or ML deployment."), no_update

        try:
            if replace_value and replace_with == "Custom" and not replace_with_custom.strip():
                return html.P("Please provide a custom replacement value."), no_update
            if replace_value and replace_scope not in ["All columns", "Numeric columns", "Categorical columns"]:
                return html.P("Invalid replacement scope selected."), no_update

            cleaned_df, logs = apply_cleaning_operations(
                df, selected_suggestions, columns_to_drop or [], options,
                replace_value or "", replace_with or "NaN", replace_scope or "All columns",
                encode_cols or [], encode_method or "Label Encoding", auto_clean=bool(auto_clean_clicks),
                enrich_col=enrich_col if enrich_col != "None" else None, enrich_api_key=enrich_api_key,
                train_ml=bool(train_ml), target_col=target_col, feature_cols=feature_cols or []
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
                    logs.append(f"Applied custom rule on {col}: {condition} {threshold}, {action} {'NaN' if action == 'Set to NaN' else action_value}")

            if preview_clicks:
                return html.Div([
                    html.H3("Preview of Changes"),
                    html.H4("Before:"),
                    dash_table.DataTable(
                        data=df.head(10).to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in df.columns],
                        style_table={'overflowX': 'auto'}
                    ),
                    html.H4("After:"),
                    dash_table.DataTable(
                        data=cleaned_df.head(10).to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in cleaned_df.columns],
                        style_table={'overflowX': 'auto'}
                    ),
                    html.H4("Preview Logs:"),
                    html.Ul([html.Li(log) for log in logs])
                ]), no_update

            if apply_clicks or auto_clean_clicks:
                session_state['previous_states'].append((df.copy(), session_state['logs'].copy()))
                if len(session_state['previous_states']) > 5:
                    session_state['previous_states'].pop(0)
                session_state['redo_states'] = []
                session_state['cleaned_df'] = cleaned_df
                session_state['logs'] = logs
                if columns_to_drop:
                    session_state['dropped_columns'].extend(columns_to_drop)
                session_state['cleaning_history'].append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "logs": logs
                })
                session_state['suggestions'] = get_cleaning_suggestions(
                    cleaned_df[[col for col in cleaned_df.columns if col not in session_state['dropped_columns']]]
                )
                session_state['progress']['Clean'] = "Done"
                save_auth_state()
                return html.Div([
                    html.H3("Changes Applied"),
                    html.Ul([html.Li(log) for log in logs]),
                    dash_table.DataTable(
                        data=cleaned_df.head(10).to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in cleaned_df.columns],
                        style_table={'overflowX': 'auto'}
                    )
                ]), session_state
        except Exception as e:
            logger.error(f"Error processing cleaning operations: {str(e)}")
            session_state['progress']['Clean'] = "Failed"
            return html.P(f"Error processing cleaning operations: {str(e)}"), session_state

    return no_update, no_update

@app.callback(
    Output('cleaning-history-container', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def update_cleaning_history(sidebar_page, stored_state):
    if sidebar_page != 'Clean':
        return no_update
    global session_state
    session_state.update(stored_state)
    if not session_state.get('cleaning_history'):
        return html.P("No cleaning operations have been performed yet.")
    history = []
    for entry in session_state['cleaning_history']:
        history.append(html.Div([
            html.H4(entry['timestamp']),
            html.Ul([html.Li(log) for log in entry['logs']])
        ]))
    return html.Div(history)

# Insights Page Callback
@app.callback(
    Output('Insights-content', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def render_insights_page_callback(sidebar_page, stored_state):
    if sidebar_page != 'Insights':
        return no_update
    global session_state
    session_state.update(stored_state)
    try:
        return render_insights_page()
    except Exception as e:
        logger.error(f"Error rendering Insights page: {str(e)}")
        return html.Div(f"An error occurred: {str(e)}")

# Visualize Page Callbacks
@app.callback(
    Output('Visualize-content', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def render_visualization_page_callback(sidebar_page, stored_state):
    if sidebar_page != 'Visualize':
        return no_update
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return html.Div("Please upload a dataset first on the Upload page.")
    try:
        return render_visualization_page(df)
    except Exception as e:
        logger.error(f"Error rendering Visualize page: {str(e)}")
        return html.Div(f"An error occurred: {str(e)}")

@app.callback(
    Output('filter-controls', 'children'),
    Input('global-filter-col', 'value'),
    State('session-store', 'data')
)
def update_filter_controls(filter_col, stored_state):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('filtered_df') or session_state.get('cleaned_df') or session_state.get('df')
    if df is None or filter_col == 'None':
        return []
    col_type = df[filter_col].dtype
    if pd.api.types.is_numeric_dtype(col_type):
        min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
        if pd.isna(min_val) or pd.isna(max_val):
            return html.P(f"Column {filter_col} contains missing values.")
        if min_val == max_val:
            return html.P(f"Column {filter_col} has identical values ({min_val}).")
        return dcc.RangeSlider(
            id='numeric-filter',
            min=min_val,
            max=max_val,
            value=[min_val, max_val],
            step=(max_val - min_val) / 100 if max_val != min_val else 1.0
        )
    elif pd.api.types.is_datetime64_any_dtype(col_type):
        min_date, max_date = df[filter_col].min(), df[filter_col].max()
        if pd.isna(min_date) or pd.isna(max_date):
            return html.P(f"Column {filter_col} contains missing values.")
        if min_date == max_date:
            return html.P(f"Column {filter_col} has identical dates ({min_date}).")
        return dcc.DatePickerRange(
            id='date-filter',
            min_date_allowed=min_date,
            max_date_allowed=max_date,
            start_date=min_date,
            end_date=max_date
        )
    else:
        unique_vals = df[filter_col].dropna().unique().tolist()
        if len(unique_vals) == 1:
            return html.P(f"Column {filter_col} has a single value ({unique_vals[0]}).")
        return dcc.Dropdown(
            id='category-filter',
            options=[{'label': val, 'value': val} for val in unique_vals],
            value=unique_vals,
            multi=True
        )

@app.callback(
    Output('viz-params', 'children'),
    Input('viz-type', 'value'),
    State('session-store', 'data')
)
def update_viz_params(viz_type, stored_state):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('filtered_df') or session_state.get('cleaned_df') or session_state.get('df')
    if df is None or not viz_type:
        return []
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_cols = df.columns.tolist()
    time_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    params = []
    if viz_type in ["Bar", "Scatter", "Line", "3D Scatter", "Bubble Chart"]:
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="X-Axis Column"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Y-Axis Column"),
            dcc.Dropdown(id='hue-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="Group By (Optional)")
        ])
        if viz_type == "3D Scatter":
            params.append(dcc.Dropdown(id='z-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Z-Axis Column"))
        if viz_type == "Bubble Chart":
            params.append(dcc.Dropdown(id='size-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Size By"))
    elif viz_type == "Histogram":
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Column"),
            dcc.Dropdown(id='hue-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="Group By (Optional)")
        ])
    elif viz_type in ["Box", "Violin", "Strip Plot", "Swarm Plot"]:
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="X-Axis Column (Optional)"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Y-Axis Column"),
            dcc.Dropdown(id='hue-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="Group By (Optional)")
        ])
    elif viz_type == "Heatmap (Correlation)":
        params.append(dcc.Dropdown(id='corr-cols', options=[{'label': col, 'value': col} for col in numeric_cols], multi=True, value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols, placeholder="Columns for Correlation"))
    elif viz_type == "Pie":
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Categories"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Values")
        ])
    elif viz_type == "Time Series Forecast":
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in time_cols], placeholder="Time Column"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Value Column"),
            dcc.Slider(id='periods', min=1, max=30, value=5, step=1, marks={1: '1', 30: '30'}),
            dcc.Dropdown(id='freq', options=[{'label': 'Daily', 'value': 'D'}, {'label': 'Monthly', 'value': 'M'}, {'label': 'Yearly', 'value': 'Y'}], value='D')
        ])
    elif viz_type == "Geospatial Map":
        params.extend([
            dcc.Dropdown(id='lat-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Latitude Column"),
            dcc.Dropdown(id='lon-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Longitude Column"),
            dcc.Dropdown(id='size-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in numeric_cols], value='None', placeholder="Size By (Optional)"),
            dcc.Dropdown(id='color-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="Color By (Optional)")
        ])
    elif viz_type == "Choropleth Map":
        params.extend([
            dcc.Dropdown(id='geo-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Geographic Column"),
            dcc.Dropdown(id='value-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Values")
        ])
    elif viz_type == "Heatmap (Geospatial)":
        params.extend([
            dcc.Dropdown(id='lat-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Latitude Column"),
            dcc.Dropdown(id='lon-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Longitude Column")
        ])
    elif viz_type == "Area Chart":
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in time_cols], placeholder="Time Column"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Y-Axis Column"),
            dcc.Dropdown(id='hue-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="Group By (Optional)")
        ])
    elif viz_type in ["Density Plot", "ECDF Plot"]:
        params.append(dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Column"))
    elif viz_type in ["Treemap", "Sunburst Chart"]:
        params.extend([
            dcc.Dropdown(id='path-cols', options=[{'label': col, 'value': col} for col in all_cols], multi=True, placeholder="Hierarchy"),
            dcc.Dropdown(id='values-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Values")
        ])
    elif viz_type == "Dendrogram":
        params.append(dcc.Dropdown(id='selected-cols', options=[{'label': col, 'value': col} for col in numeric_cols], multi=True, value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols, placeholder="Select numerical columns"))
    elif viz_type == "Network Graph":
        params.extend([
            dcc.Dropdown(id='source-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Source Node"),
            dcc.Dropdown(id='target-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Target Node"),
            dcc.Dropdown(id='weight-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in numeric_cols], value='None', placeholder="Weight (Optional)")
        ])
    elif viz_type == "Timeline":
        params.extend([
            dcc.Dropdown(id='time-col', options=[{'label': col, 'value': col} for col in time_cols], placeholder="Time Column"),
            dcc.Dropdown(id='event-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Event Column")
        ])
    elif viz_type == "Gantt Chart":
        params.extend([
            dcc.Dropdown(id='start-col', options=[{'label': col, 'value': col} for col in time_cols], placeholder="Start Time"),
            dcc.Dropdown(id='end-col', options=[{'label': col, 'value': col} for col in time_cols], placeholder="End Time"),
            dcc.Dropdown(id='task-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Task")
        ])
    elif viz_type == "Calendar Heatmap":
        params.extend([
            dcc.Dropdown(id='date-col', options=[{'label': col, 'value': col} for col in time_cols], placeholder="Date Column"),
            dcc.Dropdown(id='value-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Values")
        ])
    elif viz_type == "Parallel Coordinates":
        params.extend([
            dcc.Dropdown(id='selected-cols', options=[{'label': col, 'value': col} for col in numeric_cols], multi=True, value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols, placeholder="Select numerical columns"),
            dcc.Dropdown(id='color-col', options=[{'label': 'None', 'value': 'None'}] + [{'label': col, 'value': col} for col in all_cols], value='None', placeholder="Color By (Optional)")
        ])
    elif viz_type == "Radar Chart":
        params.extend([
            dcc.Dropdown(id='selected-cols', options=[{'label': col, 'value': col} for col in numeric_cols], multi=True, value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols, placeholder="Select numerical columns"),
            dcc.Dropdown(id='group-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Group")
        ])
    elif viz_type == "Surface Plot":
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="X-Axis Column"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Y-Axis Column"),
            dcc.Dropdown(id='z-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Z-Axis Column")
        ])
    elif viz_type == "Word Cloud":
        params.append(dcc.Dropdown(id='text-col', options=[{'label': col, 'value': col} for col in object_cols], placeholder="Text Column"))
    elif viz_type == "Gauge Chart":
        params.extend([
            dcc.Dropdown(id='value-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Value"),
            dcc.Input(id='max-value', type='number', value=df[numeric_cols[0]].max() * 1.2 if numeric_cols else 100, placeholder="Max Value")
        ])
    elif viz_type == "Funnel Chart":
        params.extend([
            dcc.Dropdown(id='stages-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Stages"),
            dcc.Dropdown(id='values-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Values")
        ])
    elif viz_type == "Sankey Diagram":
        params.extend([
            dcc.Dropdown(id='source-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Source"),
            dcc.Dropdown(id='target-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Target"),
            dcc.Dropdown(id='value-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Value")
        ])
    elif viz_type == "Waterfall Chart":
        params.extend([
            dcc.Dropdown(id='measure-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="Measure"),
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in all_cols], placeholder="X-Axis (categories)"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Y-Axis (values)")
        ])
    elif viz_type == "Pair Plot":
        params.append(dcc.Dropdown(id='selected-cols', options=[{'label': col, 'value': col} for col in numeric_cols], multi=True, value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols, placeholder="Select numerical columns"))
    elif viz_type == "Joint Plot":
        params.extend([
            dcc.Dropdown(id='x-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="X-Axis Column"),
            dcc.Dropdown(id='y-col', options=[{'label': col, 'value': col} for col in numeric_cols], placeholder="Y-Axis Column")
        ])



    elif viz_type == "Clustering":
        params.extend([
            dcc.Dropdown(id='cluster-cols', options=[{'label': col, 'value': col} for col in numeric_cols],
                    multi=True,
                    value=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols,
                    placeholder="Select columns for clustering"
                ),
                dcc.Slider(
                    id='n-clusters',
                    min=2,
                    max=10,
                    value=3,
                    step=1,
                    marks={2: '2', 10: '10'}
                )
            ])
    return params

@app.callback(
    Output('viz-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Output('suggested-viz', 'children'),
    Output('download-viz', 'data'),
    Input('generate-viz-button', 'n_clicks'),
    Input('export-button', 'n_clicks'),
    State('viz-type', 'value'),
    State('chart-title', 'value'),
    State('add-to-dashboard', 'value'),
    State('x-col', 'value'),
    State('y-col', 'value'),
    State('hue-col', 'value'),
    State('z-col', 'value'),
    State('lat-col', 'value'),
    State('lon-col', 'value'),
    State('size-col', 'value'),
    State('color-col', 'value'),
    State('periods', 'value'),
    State('freq', 'value'),
    State('corr-cols', 'value'),
    State('geo-col', 'value'),
    State('value-col', 'value'),
    State('path-cols', 'value'),
    State('values-col', 'value'),
    State('selected-cols', 'value'),
    State('source-col', 'value'),
    State('target-col', 'value'),
    State('weight-col', 'value'),
    State('time-col', 'value'),
    State('event-col', 'value'),
    State('start-col', 'value'),
    State('end-col', 'value'),
    State('task-col', 'value'),
    State('date-col', 'value'),
    State('group-col', 'value'),
    State('text-col', 'value'),
    State('max-value', 'value'),
    State('stages-col', 'value'),
    State('measure-col', 'value'),
    State('cluster-cols', 'value'),
    State('n-clusters', 'value'),
    State('global-filter-col', 'value'),
    State('numeric-filter', 'value'),
    State('date-filter', 'start_date'),
    State('date-filter', 'end_date'),
    State('category-filter', 'value'),
    State('export-format', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def generate_visualization(
    generate_clicks, export_clicks, viz_type, title, add_to_dashboard, x_col, y_col, hue_col, z_col, lat_col, lon_col,
    size_col, color_col, periods, freq, corr_cols, geo_col, value_col, path_cols, values_col, selected_cols, source_col,
    target_col, weight_col, time_col, event_col, start_col, end_col, task_col, date_col, group_col, text_col, max_value,
    stages_col, measure_col, cluster_cols, n_clusters, filter_col, numeric_filter, date_start, date_end, category_filter,
    export_format, stored_state
):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('filtered_df') or session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return html.P("No dataset available."), no_update, no_update, no_update

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Apply filters
    filtered_df = df.copy()
    if filter_col != 'None':
        col_type = df[filter_col].dtype
        if pd.api.types.is_numeric_dtype(col_type) and numeric_filter:
            filtered_df = filtered_df[(filtered_df[filter_col] >= numeric_filter[0]) & (filtered_df[filter_col] <= numeric_filter[1])]
            session_state['dashboard_filters'][filter_col] = numeric_filter
        elif pd.api.types.is_datetime64_any_dtype(col_type) and date_start and date_end:
            filtered_df = filtered_df[(filtered_df[filter_col] >= pd.to_datetime(date_start)) & (filtered_df[filter_col] <= pd.to_datetime(date_end))]
            session_state['dashboard_filters'][filter_col] = (date_start, date_end)
        elif category_filter:
            filtered_df = filtered_df[filtered_df[filter_col].isin(category_filter)]
            session_state['dashboard_filters'][filter_col] = category_filter

    if filtered_df.empty:
        return html.P("Filters resulted in an empty dataset."), no_update, no_update, no_update
    session_state['filtered_df'] = filtered_df

    if viz_type and 'Cluster' in filtered_df.columns and viz_type != "Clustering":
        filtered_df = filtered_df.rename(columns={'Cluster': 'Cluster_Old'})
        session_state['clustering_labels'] = None
        session_state['cluster_cols'] = []

    if triggered_id == 'generate-viz-button' and generate_clicks:
        try:
            if len(filtered_df) > 1000:
                filtered_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
            fig = None
            output = []

            if viz_type == "Bar":
                fig = px.bar(filtered_df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Histogram":
                fig = px.histogram(filtered_df, x=x_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Scatter":
                fig = px.scatter(filtered_df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Line":
                fig = px.line(filtered_df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Box":
                fig = px.box(filtered_df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Violin":
                fig = px.violin(filtered_df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Heatmap (Correlation)":
                if len(corr_cols) < 2:
                    return html.P("Select at least two numerical columns."), no_update, no_update, no_update
                corr = filtered_df[corr_cols].corr()
                fig = px.imshow(corr, text_auto=True, title=title)
            elif viz_type == "Pie":
                fig = px.pie(filtered_df, names=x_col, values=y_col, title=title)
            elif viz_type == "Time Series Forecast":
                if not time_cols:
                    return html.P("No datetime columns available."), no_update, no_update, no_update
                forecast_df = forecast_time_series(filtered_df, y_col, periods, time_col=x_col, freq=freq)
                historical = filtered_df[[x_col, y_col]].copy()
                historical['Type'] = 'Historical'
                forecast_df = forecast_df.reset_index().rename(columns={'index': x_col, y_col: y_col})
                forecast_df['Type'] = 'Forecast'
                combined_df = pd.concat([historical, forecast_df], ignore_index=True)
                fig = px.line(combined_df, x=x_col, y=y_col, color='Type', title=title)
                fig.add_vline(x=filtered_df[x_col].iloc[-1], line_dash="dash", line_color="red")
            elif viz_type == "3D Scatter":
                fig = px.scatter_3d(filtered_df, x=x_col, y=y_col, z=z_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Geospatial Map":
                fig = px.scatter_mapbox(
                    filtered_df, lat=lat_col, lon=lon_col, size=None if size_col == "None" else size_col,
                    color=None if color_col == "None" else color_col, title=title, zoom=3
                )
                fig.update_layout(mapbox_style="open-street-map")
            elif viz_type == "Choropleth Map":
                if not geo_col or not value_col:
                    return html.P("Please select both geographic and value columns."), no_update, no_update, no_update
                fig = px.choropleth(filtered_df, locations=geo_col, locationmode="country names", color=value_col, title=title)
            elif viz_type == "Heatmap (Geospatial)":
                fig = px.density_mapbox(
                    filtered_df, lat=lat_col, lon=lon_col, radius=10,
                    center=dict(lat=filtered_df[lat_col].mean(), lon=filtered_df[lon_col].mean()),
                    zoom=5, mapbox_style="open-street-map", title=title
                )
            elif viz_type == "Area Chart":
                if not time_cols:
                    return html.P("No datetime columns available."), no_update, no_update, no_update
                fig = px.area(filtered_df, x=x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Strip Plot":
                fig = px.strip(filtered_df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Swarm Plot":
                fig = px.strip(filtered_df, x=None if x_col == "None" else x_col, y=y_col, color=None if hue_col == "None" else hue_col, title=title)
                fig.update_traces(jitter=1)
            elif viz_type == "Density Plot":
                fig = px.density_contour(filtered_df, x=x_col, title=title)
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            elif viz_type == "ECDF Plot":
                x = filtered_df[x_col].dropna()
                ecdf = np.arange(1, len(x) + 1) / len(x)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.sort(x), y=ecdf, mode='lines', name='ECDF'))
                fig.update_layout(title=title, xaxis_title=x_col, yaxis_title="Cumulative Probability")
            elif viz_type == "Treemap":
                if not path_cols:
                    return html.P("Select at least one column for the hierarchy."), no_update, no_update, no_update
                fig = px.treemap(filtered_df, path=path_cols, values=values_col, title=title)
            elif viz_type == "Sunburst Chart":
                if not path_cols:
                    return html.P("Select at least one column for the hierarchy."), no_update, no_update, no_update
                fig = px.sunburst(filtered_df, path=path_cols, values=values_col, title=title)
            elif viz_type == "Dendrogram":
                if len(selected_cols) < 2:
                    return html.P("Select at least two numerical columns."), no_update, no_update, no_update
                X = filtered_df[selected_cols].dropna()
                Z = linkage(X, method='ward')
                fig = go.Figure()
                dendro = dendrogram(Z, no_plot=True)
                fig.add_trace(go.Scatter(x=dendro['icoord'][0], y=dendro['dcoord'][0], mode='lines', line=dict(color='white')))
                for i in range(1, len(dendro['icoord'])):
                    fig.add_trace(go.Scatter(x=dendro['icoord'][i], y=dendro['dcoord'][i], mode='lines', line=dict(color='white'), showlegend=False))
                fig.update_layout(title=title, xaxis_title="Sample Index", yaxis_title="Distance")
            elif viz_type == "Network Graph":
                G = nx.from_pandas_edgelist(filtered_df, source=source_col, target=target_col, edge_attr=None if weight_col == "None" else weight_col)
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
                fig.update_layout(title=title, showlegend=False)
            elif viz_type == "Timeline":
                fig = px.scatter(filtered_df, x=time_col, y=[0] * len(filtered_df), text=event_col, title=title)
                fig.update_traces(textposition="top center")
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            elif viz_type == "Gantt Chart":
                fig = px.timeline(filtered_df, x_start=start_col, x_end=end_col, y=task_col, title=title)
            elif viz_type == "Calendar Heatmap":
                data = filtered_df.groupby(date_col)[value_col].sum().reset_index()
                fig = px.density_heatmap(data, x=data[date_col].dt.day, y=data[date_col].dt.month, z=value_col, title=title)
            elif viz_type == "Parallel Coordinates":
                if len(selected_cols) < 2:
                    return html.P("Select at least two numerical columns."), no_update, no_update, no_update
                fig = px.parallel_coordinates(filtered_df, dimensions=selected_cols, color=None if color_col == "None" else color_col, title=title)
            elif viz_type == "Radar Chart":
                if len(selected_cols) < 2:
                    return html.P("Select at least two numerical columns."), no_update, no_update, no_update
                grouped = filtered_df.groupby(group_col)[selected_cols].mean().reset_index()
                fig = go.Figure()
                for _, row in grouped.iterrows():
                    fig.add_trace(go.Scatterpolar(r=[row[col] for col in selected_cols], theta=selected_cols, fill='toself', name=row[group_col]))
                fig.update_layout(title=title)
            elif viz_type == "Bubble Chart":
                fig = px.scatter(filtered_df, x=x_col, y=y_col, size=size_col, color=None if hue_col == "None" else hue_col, title=title)
            elif viz_type == "Surface Plot":
                data = filtered_df.pivot_table(index=x_col, columns=y_col, values=z_col).fillna(0)
                fig = go.Figure(data=[go.Surface(z=data.values, x=data.columns, y=data.index)])
                fig.update_layout(title=title, scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))
            elif viz_type == "Word Cloud":
                text = " ".join(filtered_df[text_col].dropna().astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                plt.figure(figsize=(10, 5), facecolor='black')
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
                output.append(html.Img(src=f'data:image/png;base64,{encoded}', style={'width': '100%'}))
            elif viz_type == "Gauge Chart":
                value = filtered_df[value_col].mean()
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': title}, gauge={'axis': {'range': [0, max_value]}, 'bar': {'color': "white"}}
                ))
            elif viz_type == "Funnel Chart":
                fig = px.funnel(filtered_df, x=values_col, y=stages_col, title=title)
            elif viz_type == "Sankey Diagram":
                label_list = list(set(filtered_df[source_col].tolist() + filtered_df[target_col].tolist()))
                label_dict = {label: idx for idx, label in enumerate(label_list)}
                source = filtered_df[source_col].map(label_dict)
                target = filtered_df[target_col].map(label_dict)
                value = filtered_df[value_col]
                fig = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label_list),
                    link=dict(source=source, target=target, value=value)
                )])
                fig.update_layout(title=title)
            elif viz_type == "Waterfall Chart":
                fig = go.Figure(go.Waterfall(x=filtered_df[x_col], measure=filtered_df[measure_col], y=filtered_df[y_col], textposition="auto"))
                fig.update_layout(title=title)
            elif viz_type == "Pair Plot":
                if len(selected_cols) < 2:
                    return html.P("Select at least two numerical columns."), no_update, no_update, no_update
                fig = px.scatter_matrix(filtered_df, dimensions=selected_cols, title=title)
            elif viz_type == "Joint Plot":
                fig = sns.jointplot(data=filtered_df, x=x_col, y=y_col, kind="scatter")
                buf = BytesIO()
                fig.savefig(buf, format='png')
                plt.close()
                encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
                output.append(html.Img(src=f'data:image/png;base64,{encoded}', style={'width': '100%'}))
            elif viz_type == "Clustering":
                if len(cluster_cols) < 2:
                    return html.P("Select at least two numerical columns."), no_update, no_update, no_update
                labels = perform_clustering(filtered_df, cluster_cols, n_clusters)
                session_state['clustering_labels'] = labels
                session_state['cluster_cols'] = cluster_cols
                filtered_df['Cluster'] = labels
                session_state['filtered_df'] = filtered_df
                if len(cluster_cols) >= 2:
                    fig_2d = px.scatter(
                        filtered_df, x=cluster_cols[0], y=cluster_cols[1], color=labels.astype(str),
                        labels={'color': 'Cluster'}, title="Clustering Results (2D Scatter Plot)", hover_data=cluster_cols
                    )
                    output.append(dcc.Graph(figure=fig_2d))
                if len(cluster_cols) >= 3:
                    fig_3d = px.scatter_3d(
                        filtered_df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color=labels.astype(str),
                        labels={'color': 'Cluster'}, title="Clustering Results (3D Scatter Plot)", hover_data=cluster_cols
                    )
                    output.append(dcc.Graph(figure=fig_3d))
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                fig_dist = px.bar(
                    x=cluster_counts.index.astype(str), y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Number of Points'}, title="Cluster Distribution",
                    color=cluster_counts.index.astype(str)
                )
                output.append(dcc.Graph(figure=fig_dist))
                output.append(html.H3("Dataset with Cluster Labels"))
                output.append(dash_table.DataTable(
                    data=filtered_df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in filtered_df.columns],
                    style_table={'overflowX': 'auto'},
                    page_size=10
                ))

            if fig and viz_type not in ["Word Cloud", "Joint Plot", "Clustering"]:
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white' if session_state['theme'] == 'dark' else 'black',
                    title_font_color='white' if session_state['theme'] == 'dark' else 'black',
                    showlegend=True
                )
                output.append(dcc.Graph(figure=fig))

            if add_to_dashboard:
                chart_config = {
                    "type": viz_type,
                    "title": title,
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
                    "corr_cols": corr_cols,
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
                    "measure_col": measure_col,
                    "cluster_cols": cluster_cols,
                    "n_clusters": n_clusters
                }
                session_state['dashboard_charts'].append(chart_config)
                save_auth_state()

            suggested_viz, reason = suggest_visualization(filtered_df)
            suggestions = [html.Li(f"{suggested_viz}: {reason}")]
            if viz_type == "Scatter" and len(numeric_cols) >= 2:
                suggestions.append(html.Li("Heatmap (Correlation): Explore correlations."))
            elif viz_type == "Bar" and len(object_cols) > 0:
                suggestions.append(html.Li("Pie Chart: Visualize categorical distribution."))
            elif viz_type == "Line" and time_cols:
                suggestions.append(html.Li("Time Series Forecast: Predict future values."))

            session_state['progress']['Visualize'] = "Done"
            save_auth_state()
            return html.Div(output), session_state, html.Ul(suggestions), no_update
        except ValueError as e:
            session_state['progress']['Visualize'] = "Failed"
            return html.P(f"Invalid input: {str(e)}."), no_update, no_update, no_update
        except Exception as e:
            session_state['progress']['Visualize'] = "Failed"
            return html.P(f"Error generating visualization: {str(e)}."), no_update, no_update, no_update

    if triggered_id == 'export-button' and export_clicks and fig:
        try:
            buffer = io.BytesIO()
            fig.write_image(buffer, format=export_format.lower())
            return no_update, no_update, no_update, dcc.send_bytes(buffer.getvalue(), f"{title}.{export_format.lower()}")
        except Exception as e:
            return html.P(f"Error exporting visualization: {str(e)}."), no_update, no_update, no_update

    return no_update, no_update, no_update, no_update

@app.callback(
    Output('dashboard-output', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def update_dashboard(sidebar_page, stored_state):
    if sidebar_page != 'Visualize':
        return no_update
    global session_state
    session_state.update(stored_state)
    if not session_state.get('dashboard_charts'):
        return html.P("No charts added to dashboard yet.")
    dashboard = []
    for i, chart in enumerate(session_state['dashboard_charts']):
        df = session_state.get('filtered_df') or session_state.get('cleaned_df') or session_state.get('df')
        if not df:
            continue
        fig = None
        if chart['type'] == "Bar":
            fig = px.bar(df, x=chart['x_col'], y=chart['y_col'], color=None if chart['hue_col'] == "None" else chart['hue_col'], title=chart['title'])
        elif chart['type'] == "Scatter":
            fig = px.scatter(df, x=chart['x_col'], y=chart['y_col'], color=None if chart['hue_col'] == "None" else chart['hue_col'], title=chart['title'])
        # Add more chart types as needed
        if fig:
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white' if session_state['theme'] == 'dark' else 'black',
                title_font_color='white' if session_state['theme'] == 'dark' else 'black'
            )
            dashboard.append(html.Div([
                html.H3(f"Chart {i + 1}: {chart['title']}"),
                dcc.Graph(figure=fig),
                html.Button(f"Remove Chart {i + 1}", id={'type': 'remove-chart', 'index': i})
            ]))
        else:
            dashboard.append(html.P(f"(Chart type {chart['type']} not yet supported for dashboard regeneration)"))
    return html.Div(dashboard)

@app.callback(
    Output('session-store', 'data', allow_duplicate=True),
    Input({'type': 'remove-chart', 'index': ALL}, 'n_clicks'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def remove_chart(remove_clicks, stored_state):
    global session_state
    session_state.update(stored_state)
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    index = int(json.loads(triggered_id)['index'])
    if remove_clicks[index]:
        session_state['dashboard_charts'].pop(index)
        save_auth_state()
    return session_state

# Predictive Page Callbacks
@app.callback(
    Output('Predictive-content', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def render_predictive_page_callback(sidebar_page, stored_state):
    if sidebar_page != 'Predictive':
        return no_update
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return html.Div("Please upload a dataset first on the Upload page.")
    try:
        return render_predictive_page(df)
    except Exception as e:
        logger.error(f"Error rendering Predictive page: {str(e)}")
        return html.Div(f"An error occurred: {str(e)}")

@app.callback(
    Output('model-training-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('train-model-button', 'n_clicks'),
    State('task-type', 'value'),
    State('target-col', 'value'),
    State('feature-cols', 'value'),
    State('model-type', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def train_model(n_clicks, task_type, target_col, feature_cols, model_type, stored_state):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if not n_clicks or not target_col or not feature_cols:
        return html.P("Please select a target column and at least one feature column."), no_update

    try:
        model, score, explainer, shap_values, X_test = train_ml_model(
            df, target_col, feature_cols, task_type=task_type, model_type=model_type
        )
        if model is None:
            return html.P("Model training failed. Please check the dataset and try again."), no_update

        session_state.update({
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values,
            'X_test': X_test,
            'feature_cols': feature_cols,
            'task_type': task_type
        })
        save_auth_state()

        output = [html.P(f"Model trained successfully! {task_type.capitalize()} score: {score:.2f}")]
        if task_type == "classification":
            y_test = df.loc[X_test.index, target_col]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            output.append(dcc.Graph(figure=fig_cm))
            output.append(html.H4("Classification Report"))
            output.append(html.Pre(classification_report(y_test, y_pred)))
            if "gender" in df.columns:
                sensitive_attr = df.loc[X_test.index, "gender"]
                pred_df = pd.DataFrame({'prediction': y_pred, 'gender': sensitive_attr})
                parity = pred_df.groupby('gender')['prediction'].mean()
                output.append(html.H4("Demographic Parity (Prediction Rates by Gender)"))
                output.append(html.Pre(parity.to_string()))
                if abs(parity.diff().iloc[-1]) > 0.1:
                    output.append(html.P("Potential fairness issue: Prediction rates differ significantly across groups.", style={'color': '#FFD700'}))

        return html.Div(output), session_state
    except Exception as e:
        return html.P(f"Error during model training: {str(e)}."), no_update

@app.callback(
    Output('shap-summary-plot', 'children'),
    Output('shap-force-plot', 'children'),
    Input('shap-sample-idx', 'value'),
    State('session-store', 'data')
)
def update_shap_plots(sample_idx, stored_state):
    global session_state
    session_state.update(stored_state)
    if 'model' not in session_state or 'explainer' not in session_state:
        return no_update, no_update

    try:
        shap_values = session_state['shap_values']
        X_test = session_state['X_test']
        feature_cols = session_state['feature_cols']

        # Summary Plot
        shap_df = pd.DataFrame(shap_values, columns=feature_cols)
        mean_shap = np.abs(shap_df).mean().sort_values(ascending=False)
        fig = px.bar(
            x=mean_shap.values, y=mean_shap.index, orientation='h',
            labels={'x': 'Mean |SHAP Value| (Impact on Model Output)', 'y': 'Feature'},
            title="Feature Importance (Mean |SHAP Value|)",
            color=mean_shap.values, color_continuous_scale='Viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='white' if session_state['theme'] == 'dark' else 'black',
            title_font_color='white' if session_state['theme'] == 'dark' else 'black',
            showlegend=False
        )
        summary_plot = dcc.Graph(figure=fig)

        # Force Plot
        force_plot = st_shap(
            shap.force_plot(session_state['explainer'].expected_value, shap_values[sample_idx], X_test.iloc[sample_idx], show=False, matplotlib=False),
            height=200
        )

        return summary_plot, force_plot
    except Exception as e:
        return html.P(f"Error generating SHAP visualizations: {str(e)}."), no_update

@app.callback(
    Output('lime-explanation-plot', 'children'),
    Input('lime-sample-idx', 'value'),
    State('session-store', 'data')
)
def update_lime_plot(sample_idx, stored_state):
    global session_state
    session_state.update(stored_state)
    if 'model' not in session_state:
        return no_update

    try:
        X_test = session_state['X_test']
        feature_cols = session_state['feature_cols']
        model = session_state['model']
        task_type = session_state['task_type']
        target_col = session_state.get('target_col', feature_cols[0])

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test.values,
            feature_names=feature_cols,
            class_names=[str(i) for i in range(len(np.unique(df[target_col]))) if task_type == "classification"],
            mode="classification" if task_type == "classification" else "regression"
        )
        instance = X_test.iloc[sample_idx].values
        if task_type == "classification":
            exp = lime_explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_cols))
        else:
            exp = lime_explainer.explain_instance(instance, lambda x: model.predict(x).reshape(-1), num_features=len(feature_cols))

        fig, ax = plt.subplots()
        exp.as_pyplot_figure()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close()
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        return html.Img(src=f'data:image/png;base64,{encoded}', style={'width': '100%'})
    except Exception as e:
        return html.P(f"Error generating LIME explanations: {str(e)}.")

@app.callback(
    Output('clustering-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('run-clustering-button', 'n_clicks'),
    State('cluster-cols', 'value'),
    State('n-clusters', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def run_clustering(n_clicks, cluster_cols, n_clusters, stored_state):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if not n_clicks or len(cluster_cols or []) < 2:
        return html.P("Please select at least two columns for clustering."), no_update

    try:
        labels = perform_clustering(df, cluster_cols, n_clusters)
        df['Cluster'] = labels
        session_state['cleaned_df'] = df
        session_state['clustering_labels'] = labels
        session_state['cluster_cols'] = cluster_cols
        save_auth_state()

        output = [html.P("Clustering completed successfully!")]
        if len(cluster_cols) >= 2:
            fig_2d = px.scatter(
                df, x=cluster_cols[0], y=cluster_cols[1], color=labels.astype(str),
                labels={'color': 'Cluster'}, title="Clustering Results (2D Scatter Plot)",
                hover_data=cluster_cols
            )
            output.append(dcc.Graph(figure=fig_2d))
        if len(cluster_cols) >= 3:
            fig_3d = px.scatter_3d(
                df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color=labels.astype(str),
                labels={'color': 'Cluster'}, title="Clustering Results (3D Scatter Plot)",
                hover_data=cluster_cols
            )
            output.append(dcc.Graph(figure=fig_3d))
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig_dist = px.bar(
            x=cluster_counts.index.astype(str), y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Points'}, title="Cluster Distribution",
            color=cluster_counts.index.astype(str)
        )
        output.append(dcc.Graph(figure=fig_dist))
        return html.Div(output), session_state
    except Exception as e:
        return html.P(f"Error performing clustering: {str(e)}."), no_update

@app.callback(
    Output('predictive-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True),
    Input('generate-synthetic-button', 'n_clicks'),
    Input('forecast-button', 'n_clicks'),
    Input('decompose-button', 'n_clicks'),
    Input('ui-perform-clustering-button', 'n_clicks'),
    State('task-type', 'value'),
    State('forecast-col', 'value'),
    State('forecast-periods', 'value'),
    State('forecast-freq', 'value'),
    State('decompose-col', 'value'),
    State('decompose-period', 'value'),
    State('cluster-cols', 'value'),
    State('n-clusters', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_predictive_actions(
    synthetic_clicks, forecast_clicks, decompose_clicks, clustering_clicks, task_type, forecast_col, forecast_periods,
    forecast_freq, decompose_col, decompose_period, cluster_cols, n_clusters, stored_state
):
    global session_state
    session_state.update(stored_state)
    df = session_state.get('cleaned_df') or session_state.get('df')
    if df is None:
        return html.P("No dataset available."), no_update

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'generate-synthetic-button' and synthetic_clicks:
        try:
            synthetic_df = generate_synthetic_data(df, task_type)
            session_state['cleaned_df'] = synthetic_df
            session_state['suggestions'] = get_cleaning_suggestions(
                synthetic_df[[col for col in synthetic_df.columns if col not in session_state.get('dropped_columns', [])]]
            )
            session_state['cleaning_history'].append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Generated synthetic data"]
            })
            save_auth_state()
            filename = f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv = synthetic_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return html.Div([
                html.H3("Synthetic Dataset Preview"),
                dash_table.DataTable(
                    data=synthetic_df.head(10).to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in synthetic_df.columns],
                    style_table={'overflowX': 'auto'},
                    page_size=10
                ),
                html.A(f"Download {filename}", href=f"data:file/csv;base64,{b64}", download=filename)
            ]), session_state
        except Exception as e:
            session_state['progress']['Predictive'] = "Failed"
            return html.P(f"Error generating synthetic data: {str(e)}."), no_update

    if triggered_id == 'forecast-button' and forecast_clicks and forecast_col:
        try:
            forecast_df = forecast_time_series(df, forecast_col, forecast_periods, time_col=forecast_col, freq=forecast_freq)
            filename = f"forecast_{forecast_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv = forecast_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return html.Div([
                html.H3("Forecasted Values"),
                dash_table.DataTable(
                    data=forecast_df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in forecast_df.columns],
                    style_table={'overflowX': 'auto'},
                    page_size=10
                ),
                html.A(f"Download {filename}", href=f"data:file/csv;base64,{b64}", download=filename)
            ]), session_state
        except Exception as e:
            session_state['progress']['Predictive'] = "Failed"
            return html.P(f"Error forecasting time series: {str(e)}."), no_update

    if triggered_id == 'decompose-button' and decompose_clicks and decompose_col:
        try:
            decomposition = analyze_time_series(df, decompose_col, decompose_period)
            if not decomposition:
                session_state['progress']['Predictive'] = "Failed"
                return html.P("Failed to decompose time series."), no_update
            output = [
                html.H3("Trend Component"),
                dcc.Graph(figure=px.line(x=decomposition['trend'].index, y=decomposition['trend'].values)),
                html.H3("Seasonal Component"),
                dcc.Graph(figure=px.line(x=decomposition['seasonal'].index, y=decomposition['seasonal'].values)),
                html.H3("Residual Component"),
                dcc.Graph(figure=px.line(x=decomposition['residual'].index, y=decomposition['residual'].values))
            ]
            return html.Div(output), session_state
        except Exception as e:
            session_state['progress']['Predictive'] = "Failed"
            return html.P(f"Error decomposing time series: {str(e)}."), no_update

    if triggered_id == 'ui-perform-clustering-button' and clustering_clicks and len(cluster_cols or []) >= 2:
        try:
            labels = perform_clustering(df, cluster_cols, n_clusters)
            df['Cluster'] = labels
            session_state['cleaned_df'] = df
            session_state['suggestions'] = get_cleaning_suggestions(
                df[[col for col in df.columns if col not in session_state.get('dropped_columns', [])]]
            )
            session_state['cleaning_history'].append({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": ["Performed clustering"]
            })
            session_state['progress']['Predictive'] = "Done"
            save_auth_state()
            filename = f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return html.Div([
                html.H3("Dataset with Cluster Labels"),
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df.columns],
                    style_table={'overflowX': 'auto'},
                    page_size=10
                ),
                html.A(f"Download {filename}", href=f"data:file/csv;base64,{b64}", download=filename)
            ]), session_state
        except Exception as e:
            session_state['progress']['Predictive'] = "Failed"
            return html.P(f"Error performing clustering: {str(e)}."), no_update

    return no_update, no_update

# Share Page Callback
@app.callback(
    Output('Share-content', 'children'),
    Input('sidebar-page', 'value'),
    State('session-store', 'data')
)
def render_share_page_callback(sidebar_page, stored_state):
    if sidebar_page != 'Share':
        return no_update
    global session_state
    session_state.update(stored_state)
    try:
        session_state['progress']['Share'] = 'Done'
        return html.Div("Sharing and collaboration features coming soon! Stay tuned.")
    except Exception as e:
        logger.error(f"Error rendering Share page: {str(e)}")
        return html.Div(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True)
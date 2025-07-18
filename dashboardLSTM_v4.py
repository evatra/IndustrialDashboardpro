import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib
import warnings
import os
import time
import pytz
from PIL import Image
import base64

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Industrial Gas Removal Monitoring System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSV_FILE_NAME = "data2parfull_cleaned.csv"
CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE_NAME)

FOTO_NAME = "foto.jpg"
FOTO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), FOTO_NAME)

DEFAULT_UPDATE_INTERVAL = 10800
INDONESIA_TIMEZONE = pytz.timezone('Asia/Jakarta')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USER_CREDENTIALS = {
    "engineer": hash_password("engineer123"),
    "supervisor": hash_password("supervisor123"),
    "admin": hash_password("admin123"),
}

USER_ROLES = {
    "engineer": {
        "name": "Plant Engineer", 
        "role": "Engineer",
        "department": "Process Engineering",
        "permissions": ["view", "analyze"]
    },
    "supervisor": {
        "name": "Operations Supervisor", 
        "role": "Supervisor",
        "department": "Operations",
        "permissions": ["view", "analyze", "export"]
    },
    "admin": {
        "name": "System Administrator", 
        "role": "Administrator",
        "department": "IT & Maintenance",
        "permissions": ["view", "analyze", "export", "configure"]
    },
}

def check_authentication():
    return st.session_state.get('authenticated', False)

def authenticate_user(username, password):
    if username in USER_CREDENTIALS:
        return USER_CREDENTIALS[username] == hash_password(password)
    return False

def load_and_encode_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alternative_paths = [
            os.path.join(script_dir, "assets", FOTO_NAME),
            os.path.join(script_dir, "images", FOTO_NAME),
            os.path.join(script_dir, "static", FOTO_NAME),
            os.path.join(script_dir, FOTO_NAME)
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                with open(alt_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
    except Exception:
        pass
    
    return None

def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            font-family: 'Inter', sans-serif;
        }
        
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 15px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .metric-card {
            background: var(--secondary-background-color);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .status-operational { color: #28a745; font-weight: 600; }
        .status-warning { color: #ffc107; font-weight: 600; }
        .status-critical { color: #dc3545; font-weight: 600; }
        
        .login-header-container {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
            padding: 2rem 2.5rem !important;
            margin: 2rem auto 1rem auto !important;
            max-width: 800px !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1) !important;
            text-align: center !important;
        }
        
        .login-header-container h1,
        .login-header-container h2,
        .login-header-container h3,
        .login-header-container p {
            color: #333333 !important;
            text-shadow: none !important;
            margin: 0.5rem 0 !important;
        }
        
        .login-header-container h1 {
            font-size: 3rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .login-header-container .main-title {
            font-size: 3rem !important;
            font-weight: 700 !important;
            color: #2c3e50 !important;
        }
        
        .login-header-container .subtitle {
            font-size: 2rem !important;
            color: #7f8c8d !important;
            font-weight: 500 !important;
        }
        
        .login-glass-container {
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border-radius: 25px !important;
            border: 2px solid rgba(255, 255, 255, 0.25) !important;
            box-shadow: 
                0 15px 35px rgba(0, 0, 0, 0.1),
                0 5px 15px rgba(0, 0, 0, 0.07),
                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
            padding: 2.5rem 2.5rem !important;
            margin: 0 auto 2rem auto !important;
            position: relative !important;
            overflow: hidden !important;
            max-width: 500px !important;
        }
        
        .login-glass-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.1) 0%,
                rgba(255, 255, 255, 0.05) 50%,
                rgba(255, 255, 255, 0.1) 100%);
            pointer-events: none;
            z-index: -1;
        }
        
        .login-glass-container h2,
        .login-glass-container h3 {
            color: white !important;
            text-align: center !important;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6) !important;
            margin-bottom: 1.5rem !important;
        }
        
        .login-glass-container .section-title {
            font-size: 1.6rem !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
        }
        
        .login-glass-container .form-title {
            font-size: 1.3rem !important;
            font-weight: 500 !important;
            margin-bottom: 1.5rem !important;
        }
        
        .login-glass-container .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.25) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px !important;
            color: white !important;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            font-weight: 500 !important;
            padding: 15px 20px !important;
            font-size: 16px !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s ease !important;
        }
        
        .login-glass-container .stTextInput > div > div > input:focus {
            border-color: rgba(255, 255, 255, 0.5) !important;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.1),
                0 0 20px rgba(255, 255, 255, 0.2) !important;
            outline: none !important;
        }
        
        .login-glass-container .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.8) !important;
            font-weight: 400 !important;
        }
        
        .login-glass-container .stTextInput > label {
            color: white !important;
            font-weight: 600 !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6) !important;
            font-size: 16px !important;
            margin-bottom: 8px !important;
        }
        
        .login-glass-container .stButton > button {
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.9), 
                rgba(118, 75, 162, 0.9)) !important;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            color: white !important;
            font-weight: 700 !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6) !important;
            border-radius: 15px !important;
            padding: 15px 25px !important;
            font-size: 18px !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
            transition: all 0.4s ease !important;
            box-shadow: 
                0 8px 20px rgba(102, 126, 234, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        }
        
        .login-glass-container .stButton > button:hover {
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 1), 
                rgba(118, 75, 162, 1)) !important;
            transform: translateY(-3px) !important;
            box-shadow: 
                0 15px 35px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
        }
        
        .login-glass-container [data-testid="stForm"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        
        .login-glass-container .stAlert {
            background: rgba(255, 255, 255, 0.15) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
        }
        
        .login-glass-container .stAlert [data-testid="alertIndicator"] {
            color: white !important;
        }
        
        .login-glass-footer {
            text-align: center !important;
            margin-top: 2rem !important;
            padding-top: 2rem !important;
            border-top: 1px solid rgba(255,255,255,0.3) !important;
        }
        
        .login-glass-footer p {
            color: white !important;
            text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.6) !important;
            margin: 0.5rem 0 !important;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def get_current_localized_time():
    return datetime.now(INDONESIA_TIMEZONE)

def init_session_state():
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = get_current_localized_time()
    
    if 'update_interval' not in st.session_state:
        st.session_state.update_interval = DEFAULT_UPDATE_INTERVAL
    
    if 'auto_update_enabled' not in st.session_state:
        st.session_state.auto_update_enabled = True
    
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    
    if 'last_file_modified' not in st.session_state:
        st.session_state.last_file_modified = None
    
    if 'selected_interval_label' not in st.session_state:
        st.session_state.selected_interval_label = '3 hours'
    
    if 'last_manual_refresh' not in st.session_state:
        st.session_state.last_manual_refresh = None

@st.cache_data(ttl=DEFAULT_UPDATE_INTERVAL)
def load_csv_automatically(file_path):
    try:
        if not os.path.exists(file_path):
            return None
        
        delimiters = [',', ';', '\t']
        df = None
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(file_path, sep=delimiter)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
        
        if df is None:
            return None
        
        return df
        
    except Exception:
        return None

def check_and_update():
    current_time = get_current_localized_time()
    time_diff = (current_time - st.session_state.last_update_time).total_seconds()
    
    if time_diff >= st.session_state.update_interval and st.session_state.auto_update_enabled:
        st.session_state.last_update_time = current_time
        load_csv_automatically.clear()
        st.rerun()

def format_time_remaining():
    current_time = get_current_localized_time()
    time_diff = (current_time - st.session_state.last_update_time).total_seconds()
    time_remaining = st.session_state.update_interval - time_diff
    
    if time_remaining <= 0:
        return "Update pending..."
    
    hours = int(time_remaining // 3600)
    minutes = int((time_remaining % 3600) // 60)
    seconds = int(time_remaining % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def login_page():
    apply_custom_css()
    
    image_data = load_and_encode_image(FOTO_PATH)
    if image_data:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{image_data});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: -1;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-header-container">
            <h1 class="main-title">Industrial Monitoring System</h1>
            <p class="subtitle">Gas Removal Predictive Maintenance</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("secure_login", clear_on_submit=False):
            username = st.text_input(
                "👤 Username", 
                placeholder="Enter your username",
                key="username_input"
            )
            
            password = st.text_input(
                "🔑 Password", 
                type="password", 
                placeholder="Enter your password",
                key="password_input"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            login_button = st.form_submit_button("🚀 Access System", type="primary")
            
            if login_button:
                if username and password:
                    if authenticate_user(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['user_info'] = USER_ROLES.get(username, {})
                        st.success("✅ Authentication successful! Loading system...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Access denied.")
                else:
                    st.warning("⚠️ Please provide both username and password.")
        
        st.markdown("""
            <div class="login-glass-footer">
                <p style="font-weight: 600; font-size: 1.1rem;">🔒 Secure Industrial System Access</p>
                <p style="opacity: 0.8; font-style: italic;">Powered by Advanced Authentication & Monitoring</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def logout():
    for key in ['authenticated', 'username', 'user_info', 'csv_data', 'last_file_modified', 'last_update_time']:
        if key in st.session_state:
            del st.session_state[key]
    load_csv_automatically.clear()
    st.rerun()

def show_user_panel():
    if 'user_info' in st.session_state:
        user_info = st.session_state['user_info']
        
        st.sidebar.markdown("### 👤 User Profile")
        
        st.sidebar.success(f"**{user_info.get('name', 'User')}**\n\n{user_info.get('role', 'User')} • {user_info.get('department', 'General')}")
        
        permissions = user_info.get('permissions', [])
        st.sidebar.markdown("**Access Level:**")
        for perm in permissions:
            st.sidebar.write(f"✅ {perm.title()}")
        
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("### 🔄 Auto-Update Settings")
        
        st.session_state.auto_update_enabled = st.sidebar.checkbox(
            "Enable Auto-Update",
            value=st.session_state.auto_update_enabled,
            help="Automatically refresh data at specified intervals"
        )
        
        update_options = {
            "1 hour": 3600,
            "3 hours": 10800,
            "6 hours": 21600,
            "12 hours": 43200,
            "24 hours": 86400
        }
        
        current_interval_value = st.session_state.get('update_interval', DEFAULT_UPDATE_INTERVAL)
        
        default_index = 0
        for idx, (label, value) in enumerate(update_options.items()):
            if value == current_interval_value:
                default_index = idx
                break
        
        selected_interval = st.sidebar.selectbox(
            "Update Interval",
            options=list(update_options.keys()),
            index=default_index,
            key="update_interval_selector",
            help="How often to refresh the data"
        )
        
        if update_options[selected_interval] != st.session_state.update_interval:
            st.session_state.update_interval = update_options[selected_interval]
            load_csv_automatically.clear()
            st.rerun()

        st.session_state.selected_interval_label = selected_interval
        
        st.sidebar.markdown("**Auto-Update Status:**")
        if st.session_state.auto_update_enabled:
            st.sidebar.info(f"🕐 Next update in: {format_time_remaining()}")
        else:
            st.sidebar.warning("⏸️ Auto-update disabled")
        
        if st.sidebar.button("🔄 Refresh Now", type="secondary"):
            load_csv_automatically.clear()
            st.session_state.last_manual_refresh = get_current_localized_time()
            st.rerun()
        
        st.sidebar.write(f"**Next Auto-Update:** {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.session_state.last_manual_refresh:
            st.sidebar.write(f"**Last Manual Refresh:** {st.session_state.last_manual_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🚪 Secure Logout", type="primary"):
            logout()

def show_system_status(system_status, current_pressure, predicted_pressure_now, threshold, predicted_breach_time=None):
    if system_status == "CRITICAL":
        st.error(f"""
        🚨 **CRITICAL ALERT**
        
        **Immediate maintenance required!**
        
        • Current pressure: {current_pressure:.4f}
        • Predicted pressure: {predicted_pressure_now:.4f}
        • Threshold: {threshold:.4f}
        """)
    elif system_status == "WARNING":
        breach_message = ""
        if predicted_breach_time:
            breach_message = f"• **Predicted to reach threshold by:** {predicted_breach_time.strftime('%Y-%m-%d %H:%M')}"
        
        st.warning(f"""
        ⚠️ **WARNING**
        
        **System approaching critical levels.**
        
        Schedule maintenance within 24 hours.
        • Current pressure: {current_pressure:.4f}
        {breach_message}
        """)
    else:
        st.success(f"""
        ✅ **SYSTEM OPERATIONAL**
        
        All systems operating within normal parameters.
        • Current pressure: {current_pressure:.4f}
        """)

def main_dashboard():
    init_session_state()
    check_and_update()
    apply_custom_css()
    
    st.markdown("# 🏭 Industrial Gas Removal Monitoring System")
    st.markdown("### Predictive Maintenance & Real-time Process Monitoring")
    st.markdown("---")
    
    show_user_panel()
    
    MODEL_PATH = "best_lstm_model.h5"
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error("Critical System Error: Cannot load predictive model. Please contact system administrator.")
        st.stop()
    
    st.sidebar.markdown("### 🎛️ System Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        default_start_date = get_current_localized_time().date() - timedelta(days=7)
        start_date = st.date_input(
            "Start Date", 
            value=default_start_date,
            help="Data collection start date"
        )
    
    with col2:
        data_frequency = st.selectbox(
            "Data Frequency",
            options=["Hourly", "Daily", "Weekly", "Monthly"],
            index=0,
            help="Data sampling frequency"
        )
    
    threshold = st.sidebar.slider(
        "Critical Threshold", 
        min_value=0.05, 
        max_value=0.30, 
        value=0.14, 
        step=0.01,
        help="Critical pressure threshold for maintenance alerts"
    )
    
    sequence_length = st.sidebar.slider(
        "Prediction Sequence Length", 
        min_value=20, 
        max_value=120, 
        value=80, 
        step=10,
        help="Number of historical points used for prediction"
    )
    
    show_detailed_table = st.sidebar.checkbox("Show Detailed Data Table", value=False)
    
    df = load_csv_automatically(CSV_FILE_PATH)
    
    if df is not None:
        with st.spinner("🔄 Processing sensor data..."):
            with st.expander("🔍 View Sample Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            timestamp_cols = [c for c in df.columns if any(keyword in c.lower() 
                             for keyword in ['time', 'date', 'timestamp', 'waktu', 'tanggal'])]
            
            pressure_cols = [c for c in df.columns if any(keyword in c.lower() 
                            for keyword in ['tekanan', 'pressure', 'kondensor', 'condenser'])]
            
            if not pressure_cols:
                st.error("❌ Pressure column not found. Please ensure your data contains pressure measurements.")
                st.stop()
            
            pressure_col = pressure_cols[0]
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                
                date_formats = [
                    '%d/%m/%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M',
                    '%d-%m-%Y %H:%M:%S',
                    '%d-%m-%Y %H:%M',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M',
                    '%m/%d/%Y %H:%M:%S',
                    '%m/%d/%Y %H:%M',
                    '%d/%m/%Y',
                    '%d-%m-%Y',
                    '%Y-%m-%d',
                    '%m/%d/%Y'
                ]
                
                parsed_successfully = False
                
                for date_format in date_formats:
                    try:
                        df['timestamp'] = pd.to_datetime(df[timestamp_col], format=date_format, errors='coerce')
                        if not df['timestamp'].isna().all():
                            parsed_successfully = True
                            break
                    except Exception:
                        continue
                
                if not parsed_successfully:
                    try:
                        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce', infer_datetime_format=True)
                        if not df['timestamp'].isna().all():
                            parsed_successfully = True
                    except Exception:
                        pass
                
                if not parsed_successfully or df['timestamp'].isna().all():
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H', tz=INDONESIA_TIMEZONE)
                else:
                    try:
                        if df['timestamp'].dt.tz is None:
                            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(INDONESIA_TIMEZONE)
                        else:
                            df['timestamp'] = df['timestamp'].dt.tz_convert(INDONESIA_TIMEZONE)
                    except Exception as e:
                        pass
            else:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H', tz=INDONESIA_TIMEZONE)
            
            data = df[[pressure_col]].copy()
            data = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
            
            valid_indices = data.dropna().index
            data = data.dropna()
            
            if (data < 0).any().any():
                data = data.clip(lower=0)
            
            ground_truth_all = data.values.flatten()
            timestamps_all = df['timestamp'].iloc[valid_indices].tolist()
            
            if len(ground_truth_all) < sequence_length + 10:
                st.error(f"❌ Insufficient data. Need at least {sequence_length + 10} valid data points, but only have {len(ground_truth_all)}.")
                st.stop()
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data_all = scaler.fit_transform(data)
            
            train_size = int(len(scaled_data_all) * 0.8)
            train_data = scaled_data_all[:train_size]
            test_data = scaled_data_all[train_size:]
            
            def create_sequences(data, seq_length):
                sequences, targets = [], []
                for i in range(len(data) - seq_length):
                    sequences.append(data[i:i+seq_length])
                    targets.append(data[i+seq_length])
                return np.array(sequences), np.array(targets)

            X_train, y_train = create_sequences(train_data, sequence_length)
            X_test, y_test = create_sequences(test_data, sequence_length)

            if len(X_test) > 0:
                predictions_on_test = model.predict(X_test)
                predictions_on_test_inv = scaler.inverse_transform(predictions_on_test).flatten()
                actual_test_inv = scaler.inverse_transform(y_test).flatten()
                
                test_timestamps = timestamps_all[train_size + sequence_length:]
                
                mse = mean_squared_error(actual_test_inv, predictions_on_test_inv)
                mae = mean_absolute_error(actual_test_inv, predictions_on_test_inv)
                r2 = r2_score(actual_test_inv, predictions_on_test_inv)
                
                accuracy_tolerance = 0.01
                accuracy = np.mean(np.abs(actual_test_inv - predictions_on_test_inv) <= accuracy_tolerance) * 100
            else:
                mse, mae, r2, accuracy = 0, 0, 0, 0
                st.warning("Insufficient data for testing.")
            
            def predict_future(model, last_sequence, scaler, sequence_length, future_steps):
                predicted_values = []
                current_sequence = last_sequence.copy()
                
                for _ in range(future_steps):
                    input_seq = current_sequence.reshape(1, sequence_length, 1)
                    next_pred_scaled = model.predict(input_seq, verbose=0)[0]
                    predicted_values.append(next_pred_scaled[0])
                    current_sequence = np.append(current_sequence[1:], next_pred_scaled[0])
                
                predicted_values_inv = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1)).flatten()
                return predicted_values_inv

            future_steps_1_month = 30 * 24
            last_sequence = scaled_data_all[-sequence_length:]
            future_predictions_inv = predict_future(model, last_sequence, scaler, sequence_length, future_steps_1_month)
            
            if len(timestamps_all) > 0:
                last_timestamp = timestamps_all[-1]
                
                if pd.isna(last_timestamp):
                    last_timestamp = get_current_localized_time()
                
                if not isinstance(last_timestamp, pd.Timestamp):
                    last_timestamp = pd.Timestamp(last_timestamp)
                
                if last_timestamp.tz is None:
                    last_timestamp = last_timestamp.tz_localize(INDONESIA_TIMEZONE)
                elif last_timestamp.tz != INDONESIA_TIMEZONE:
                    last_timestamp = last_timestamp.tz_convert(INDONESIA_TIMEZONE)
                
                try:
                    future_timestamps = pd.date_range(
                        start=last_timestamp + timedelta(hours=1), 
                        periods=future_steps_1_month, 
                        freq='H', 
                        tz=INDONESIA_TIMEZONE
                    ).tolist()
                except Exception as e:
                    st.error(f"Error creating future timestamps: {e}")
                    fallback_start = get_current_localized_time()
                    future_timestamps = pd.date_range(
                        start=fallback_start, 
                        periods=future_steps_1_month, 
                        freq='H', 
                        tz=INDONESIA_TIMEZONE
                    ).tolist()
            else:
                st.error("No valid timestamps available for future prediction.")
                return
            
            current_pressure = ground_truth_all[-1] if len(ground_truth_all) > 0 else 0
            predicted_pressure_now = predictions_on_test_inv[-1] if len(predictions_on_test_inv) > 0 else 0
            
            system_status = "OPERATIONAL"
            predicted_breach_time = None

            for i, val in enumerate(future_predictions_inv):
                if val >= threshold:
                    predicted_breach_time = future_timestamps[i]
                    break
            
            if current_pressure > threshold or predicted_pressure_now > threshold:
                system_status = "CRITICAL"
            elif current_pressure > threshold * 0.8 or predicted_pressure_now > threshold * 0.8 or predicted_breach_time:
                system_status = "WARNING"

            show_system_status(system_status, current_pressure, predicted_pressure_now, threshold, predicted_breach_time)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                status_color = "🟢" if system_status == "OPERATIONAL" else "🟡" if system_status == "WARNING" else "🔴"
                st.metric("System Status", f"{status_color} {system_status}")
            
            with col2:
                delta_val = f"{(current_pressure - threshold):.4f}" if current_pressure > threshold else None
                st.metric("Current Pressure", f"{current_pressure:.4f}", delta=delta_val)
            
            with col3:
                delta_val = f"{(r2-0.8)*100:.1f}%" if r2 > 0.8 else None
                st.metric("Model Accuracy (R²)", f"{r2*100:.1f}%", delta=delta_val)
            
            with col4:
                delta_val = f"{(mae-0.01):.4f}" if mae > 0.01 else None
                st.metric("Prediction Error (MAE)", f"{mae:.4f}", delta=delta_val)
            
            with col5:
                maintenance_hours = 24 if system_status == "WARNING" else (0 if system_status == "CRITICAL" else 168)
                delta_val = "URGENT" if maintenance_hours == 0 else None
                st.metric("Maintenance Window", f"{maintenance_hours}h", delta=delta_val)
            
            st.markdown("### 📈 Process Monitoring & Prediction (Including 1-Month Forecast)")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps_all, 
                y=ground_truth_all,
                mode='lines',
                name='Historical Data',
                line=dict(color='#2E86AB', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_timestamps,
                y=future_predictions_inv,
                mode='lines',
                name=f'Future Prediction (30 days)',
                line=dict(color='#00CC96', width=3, dash='dot')
            ))
            
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Critical Threshold ({threshold})"
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title="Industrial Gas Removal System - Process Monitoring & Prediction",
                title_x=0.5,
                template="plotly_white",
                xaxis_title="Time",
                yaxis_title="Pressure"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📊 Model Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_data = {
                    'Metric': ['Mean Squared Error (Test)', 'Mean Absolute Error (Test)', 'R² Score (Test)', f'Accuracy (±{accuracy_tolerance}) (Test)'],
                    'Value': [f"{mse:.6f}", f"{mae:.6f}", f"{r2:.4f}", f"{accuracy:.2f}%"],
                    'Status': [
                        'Good' if mse < 0.001 else 'Acceptable' if mse < 0.01 else 'Poor',
                        'Good' if mae < 0.01 else 'Acceptable' if mae < 0.05 else 'Poor',
                        'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.8 else 'Acceptable',
                        'Excellent' if accuracy > 90 else 'Good'
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                if len(actual_test_inv) > 0:
                    error = np.abs(actual_test_inv - predictions_on_test_inv)
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=error,
                        nbinsx=20,
                        name='Error Distribution',
                        marker_color='rgba(46, 134, 171, 0.7)'
                    ))
                    fig_dist.update_layout(
                        title="Prediction Error Distribution (Test Set)",
                        xaxis_title="Absolute Error",
                        yaxis_title="Frequency",
                        template="plotly_white",
                        height=300
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            if show_detailed_table:
                st.markdown("### 📋 Detailed Process Data")
                
                full_timestamps = timestamps_all + future_timestamps
                full_pressures = ground_truth_all.tolist() + future_predictions_inv.tolist()

                detailed_df = pd.DataFrame({
                    'Timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in full_timestamps],
                    'Pressure': full_pressures,
                    'Type': ['Historical'] * len(ground_truth_all) + ['Predicted'] * len(future_predictions_inv),
                    'Status': ['Normal' if p < threshold else 'Critical' for p in full_pressures]
                })

                rows_per_page = 20
                
                if "table_page" not in st.session_state:
                    st.session_state.table_page = 0
                
                total_pages = (len(detailed_df) - 1) // rows_per_page + 1
                
                col1_p, col2_p, col3_p = st.columns([1, 2, 1])
                
                with col1_p:
                    if st.button("← Previous", disabled=st.session_state.table_page == 0):
                        st.session_state.table_page -= 1
                        st.rerun()
                
                with col2_p:
                    st.write(f"Page {st.session_state.table_page + 1} of {total_pages}")
                
                with col3_p:
                    if st.button("Next →", disabled=st.session_state.table_page >= total_pages - 1):
                        st.session_state.table_page += 1
                        st.rerun()
                
                start_idx = st.session_state.table_page * rows_per_page
                end_idx = start_idx + rows_per_page
                
                st.dataframe(detailed_df.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)
            
            st.markdown("### 📤 Data Export")
            
            col_export_1, col_export_2 = st.columns(2)
            
            with col_export_1:
                export_df = pd.DataFrame({
                    'Timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps_all + future_timestamps],
                    'Pressure': ground_truth_all.tolist() + future_predictions_inv.tolist(),
                    'Type': ['Historical'] * len(ground_truth_all) + ['Future_Prediction'] * len(future_predictions_inv),
                    'Status': ['Normal' if p < threshold else 'Critical' for p in ground_truth_all.tolist() + future_predictions_inv.tolist()]
                })
                
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Analysis Report (CSV)",
                    data=csv_data,
                    file_name=f"gas_removal_analysis_{get_current_localized_time().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_export_2:
                report_text = f"""Industrial Gas Removal System - Analysis Report
Generated: {get_current_localized_time().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM STATUS: {system_status}
Current Pressure: {current_pressure:.4f}
Critical Threshold: {threshold:.4f}

MODEL PERFORMANCE (On Test Set):
- R² Score: {r2:.4f}
- Mean Absolute Error: {mae:.6f}
- Mean Squared Error: {mse:.6f}
- Accuracy (±{accuracy_tolerance}): {accuracy:.2f}%

DATA SOURCE:
- File: {CSV_FILE_NAME}
- Last Updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}
- Auto-Update: {'Enabled' if st.session_state.auto_update_enabled else 'Disabled'}
- Update Interval: {st.session_state.get('selected_interval_label', '3 hours')}

MAINTENANCE RECOMMENDATION:
{
"Immediate maintenance required - System critical!" if system_status == "CRITICAL" else
"Schedule maintenance within 24 hours" + (f" (Predicted breach by: {predicted_breach_time.strftime('%Y-%m-%d %H:%M')})" if predicted_breach_time else "") if system_status == "WARNING" else
"No immediate maintenance required"
}"""
                
                st.download_button(
                    label="📄 Download Summary Report (TXT)",
                    data=report_text,
                    file_name=f"gas_removal_summary_report_{get_current_localized_time().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

def main():
    if not check_authentication():
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()

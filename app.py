# ============================================================================
# HR ANALYTICS DASHBOARD - PROFESSIONAL VERSION
# Clean design without emojis, comprehensive EDA with filters
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')

# Get the project root directory (parent of Streamlit folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'Dataset')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DARK MODE STATE
# ============================================================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model_path = os.path.join(MODELS_DIR, 'gb_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        feature_cols_path = os.path.join(MODELS_DIR, 'feature_columns.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(feature_cols_path)
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None

@st.cache_data
def load_data():
    """Load training data"""
    try:
        data_path = os.path.join(DATASET_DIR, 'aug_train.csv')
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

model, scaler, feature_columns = load_models()
df_raw = load_data()

if model is None or df_raw is None:
    st.error("Error loading models or data. Please check files.")
    st.stop()

# ============================================================================
# DYNAMIC STYLING - COMPREHENSIVE MODERN DARK ANALYTICS DASHBOARD
# ============================================================================
def get_theme_css(dark_mode=False):
    if dark_mode:
        # Modern Dark Analytics Dashboard with Neon Accents
        return """
        <style>
        * { box-sizing: border-box; }
        
        :root {
            --neon-blue: #4da3ff;
            --neon-purple: #8a5cff;
            --neon-teal: #00c2a8;
            --neon-pink: #ff5fa2;
            --dark-bg: #0a0e27;
            --card-dark: #151933;
            --card-light: #1a1f3a;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --border-light: #2a3050;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #151a35 100%);
        }
        
        h1 { color: #e0e0e0 !important; font-weight: 900; font-size: 36px; margin: 35px 0 15px 0; letter-spacing: -0.5px; }
        h2 { color: #e0e0e0 !important; font-weight: 800; font-size: 26px; margin: 28px 0 18px 0; letter-spacing: -0.3px; }
        h3 { color: #e0e0e0 !important; font-weight: 700; font-size: 20px; margin: 25px 0 15px 0; }
        h4 { color: #c0c0c0 !important; font-weight: 600; font-size: 16px; }
        p { color: #b0b0b0 !important; line-height: 1.6; }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1429 0%, #1a1f3a 100%) !important;
            border-right: 1px solid #2a3050 !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
            color: #e0e0e0 !important;
        }
        
        [data-testid="stMainBlockContainer"] {
            background-color: transparent !important;
            padding-right: 20px !important;
            padding-left: 20px !important;
            padding-top: 20px !important;
            padding-bottom: 20px !important;
        }
        
        /* Main content area */
        .main {
            padding-left: 20px !important;
            padding-right: 20px !important;
        }
        
        /* Dividers */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #2a3050, transparent);
            margin: 30px 0;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #151933 0%, #0f1429 100%);
            border: 1px solid #2a3050;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            border-top: 3px solid #4da3ff;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            border-top-color: #8a5cff;
            box-shadow: 0 6px 20px rgba(138, 92, 255, 0.15);
        }
        
        .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; font-weight: 600; }
        .metric-value { font-size: 32px; font-weight: 800; color: #4da3ff; margin: 5px 0; }
        .metric-delta { font-size: 12px; color: #a0a0a0; margin-top: 8px; }
        
        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1f3a !important;
            color: #ffffff !important;
            border-radius: 10px 10px 0 0;
            border: 1px solid #2a3050 !important;
            padding: 14px 20px !important;
            transition: all 0.3s ease;
            font-weight: 700 !important;
            font-size: 14px !important;
            letter-spacing: 0.3px;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #222847 !important;
            color: #ffffff !important;
            border-color: #4da3ff !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4da3ff 0%, #8a5cff 100%) !important;
            color: #ffffff !important;
            border-color: #8a5cff !important;
            font-weight: 800 !important;
            box-shadow: 0 4px 12px rgba(77, 163, 255, 0.25);
        }
        
        /* Forms and Inputs */
        input, select, textarea {
            background-color: #1a1f3a !important;
            color: #e0e0e0 !important;
            border: 1px solid #2a3050 !important;
            border-radius: 8px !important;
            padding: 12px 15px !important;
            font-size: 14px !important;
            font-family: inherit !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        input:focus, select:focus, textarea:focus {
            border-color: #4da3ff !important;
            box-shadow: 0 0 12px rgba(77, 163, 255, 0.25) !important;
            background-color: #151933 !important;
            outline: none !important;
        }
        
        /* Selectbox Container - FULL WIDTH */
        .stSelectbox {
            width: 100% !important;
        }
        
        .stSelectbox > div {
            width: 100% !important;
        }
        
        .stSelectbox > div > div {
            width: 100% !important;
            border-color: #2a3050 !important;
            background-color: #1a1f3a !important;
            border-radius: 8px !important;
            box-sizing: border-box !important;
        }
        
        .stSelectbox > div > div > div {
            color: #e0e0e0 !important;
            width: 100% !important;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: #4da3ff !important;
            box-shadow: 0 0 12px rgba(77, 163, 255, 0.25) !important;
        }
        
        /* Multiselect Container - FULL WIDTH */
        .stMultiSelect {
            width: 100% !important;
        }
        
        .stMultiSelect > div {
            width: 100% !important;
            min-height: 40px !important;
        }
        
        .stMultiSelect > div > div {
            width: 100% !important;
            background-color: #1a1f3a !important;
            border: 1px solid #2a3050 !important;
            border-radius: 8px !important;
            box-sizing: border-box !important;
            padding: 8px 12px !important;
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 6px !important;
            align-items: center !important;
            min-height: 40px !important;
        }
        
        .stMultiSelect > div > div > div {
            width: auto !important;
            max-width: 100% !important;
        }
        
        /* Multiselect tags styling */
        [data-testid="multiSelectOptionHighlightedContainer"] {
            background-color: #222847 !important;
            border-color: #4da3ff !important;
        }
        
        /* Multiselect focus */
        .stMultiSelect > div > div:focus-within {
            border-color: #4da3ff !important;
            box-shadow: 0 0 12px rgba(77, 163, 255, 0.25) !important;
        }
        
        /* Slider Container */
        .stSlider {
            width: 100% !important;
        }
        
        .stSlider > div {
            width: 100% !important;
        }
        
        /* Number Input - PROPER ALIGNMENT */
        .stNumberInput {
            width: 100% !important;
        }
        
        .stNumberInput > div {
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            gap: 0 !important;
        }
        
        .stNumberInput > div > div {
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
        }
        
        .stNumberInput input {
            width: 100% !important;
            flex: 1 !important;
            padding: 12px 15px !important;
        }
        
        /* Number input +/- buttons - ALIGNED WITH INPUT */
        .stNumberInput button {
            padding: 10px 12px !important;
            min-width: 40px !important;
            height: 40px !important;
            box-sizing: border-box !important;
            margin: 0 2px !important;
        }
        
        /* Slider alignment */
        .stSlider {
            width: 100% !important;
        }
        
        .stSlider > div {
            width: 100% !important;
        }
        
        .stSlider > div > div {
            display: flex !important;
            flex-direction: column !important;
            gap: 0 !important;
        }
        
        /* Form Container - PROPER STYLING */
        .stForm {
            width: 100% !important;
            padding: 0 !important;
        }
        
        /* Form column alignment - EQUAL WIDTH & SPACING */
        .stForm > [data-testid="column"] {
            padding-right: 8px !important;
            padding-left: 8px !important;
            box-sizing: border-box !important;
        }
        
        .stForm > [data-testid="column"]:first-child {
            padding-left: 0 !important;
        }
        
        .stForm > [data-testid="column"]:last-child {
            padding-right: 0 !important;
        }
        
        /* Column container alignment */
        [data-testid="column"] {
            width: 100% !important;
        }
        
        /* Column container alignment - EQUAL WIDTH */
        [data-testid="column"] {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        /* Ensure consistent gap between columns */
        [data-testid="stHorizontalBlock"] {
            gap: 16px !important;
        }
        
        /* Remove extra padding/margins */
        .stSelectbox > label,
        .stMultiSelect > label,
        .stSlider > label,
        .stNumberInput > label,
        .stTextInput > label {
            color: #e0e0e0 !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
            display: block !important;
        }
        
        /* Form field container */
        .stSelectbox,
        .stMultiSelect,
        .stSlider,
        .stNumberInput,
        .stTextInput {
            width: 100% !important;
            margin-bottom: 0 !important;
        }
        
        /* Text Input - FULL WIDTH */
        .stTextInput {
            width: 100% !important;
        }
        
        .stTextInput > div {
            width: 100% !important;
        }
        
        .stTextInput input {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        /* Ensure all input types have consistent styling */
        .stSelectbox,
        .stMultiSelect,
        .stSlider,
        .stNumberInput,
        .stTextInput {
            min-height: 50px !important;
        }
        
        .stRadio > div > label {
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
            padding: 10px 12px !important;
            margin: 0 !important;
            width: 100% !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            color: #b0b0b0 !important;
            font-weight: 500 !important;
            font-size: 14px !important;
        }
        
        .stRadio > div > label:hover {
            background-color: rgba(77, 163, 255, 0.1) !important;
            color: #4da3ff !important;
        }
        
        /* Active radio button */
        .stRadio > div > label input[type="radio"]:checked + div {
            color: #4da3ff !important;
        }
        
        .stRadio > div > label:has(input[type="radio"]:checked) {
            background: rgba(77, 163, 255, 0.15) !important;
            border-left: 3px solid #4da3ff !important;
            padding-left: 9px !important;
            color: #4da3ff !important;
            font-weight: 600 !important;
        }
        
        /* Form labels */
        label {
            color: #e0e0e0 !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
            display: block !important;
        }
        
        /* Sliders */
        .stSlider > label {
            color: #e0e0e0 !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSlider"] {
            padding: 10px 0 !important;
        }
        
        /* Number inputs */
        .stNumberInput > div > div {
            background-color: #1a1f3a !important;
        }
        
        /* Multiselect */
        .stMultiSelect > div > div {
            background-color: #1a1f3a !important;
            border: 1px solid #2a3050 !important;
            border-radius: 8px !important;
        }
        
        /* Buttons */
        button {
            background: linear-gradient(135deg, #4da3ff 0%, #8a5cff 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            transition: all 0.2s !important;
        }
        
        button:hover {
            box-shadow: 0 6px 20px rgba(77, 163, 255, 0.3) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Card Classes for Components */
        .gradient-card-1 {
            background: linear-gradient(135deg, #1a3a52 0%, #0f1f38 100%);
            border: 1px solid #2a5a7a;
            border-left: 4px solid #4da3ff;
        }
        
        .gradient-card-2 {
            background: linear-gradient(135deg, #2a2a52 0%, #1a1a38 100%);
            border: 1px solid #3a2a7a;
            border-left: 4px solid #8a5cff;
        }
        
        .gradient-card-3 {
            background: linear-gradient(135deg, #1a4a4f 0%, #0a2a3f 100%);
            border: 1px solid #2a5a6f;
            border-left: 4px solid #00c2a8;
        }
        
        .gradient-card-4 {
            background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
            border: 1px solid #3a2a6a;
            border-left: 4px solid #ff5fa2;
        }
        
        /* Insight/Info Boxes */
        .insight-box {
            background: linear-gradient(135deg, #1a3a52 0%, #0f2038 100%);
            border: 1px solid #2a5a7a;
            border-left: 4px solid #4da3ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            color: #e0e0e0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        .insight-box.warning {
            background: linear-gradient(135deg, #4a3a1a 0%, #3a2a0a 100%);
            border-left-color: #ff5fa2;
        }
        
        .insight-box.danger {
            background: linear-gradient(135deg, #5a2a2a 0%, #4a1a1a 100%);
            border-left-color: #ff5fa2;
        }
        
        .insight-box.success {
            background: linear-gradient(135deg, #2a4a3a 0%, #1a3a2a 100%);
            border-left-color: #00c2a8;
        }
        
        /* DataFrames */
        [data-testid="stDataFrame"] {
            background-color: #151933 !important;
        }
        
        .streamlit-expanderHeader {
            background-color: #1a1f3a !important;
            border: 1px solid #2a3050 !important;
        }
        
        /* Insight Cards */
        .insight-card {
            background: linear-gradient(135deg, #151933 0%, #0f1429 100%);
            border: 1px solid #2a3050;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        
        .insight-card:hover {
            box-shadow: 0 8px 25px rgba(77, 163, 255, 0.15);
            transform: translateY(-2px);
        }
        
        .insight-card-blue {
            border-left: 5px solid #4da3ff;
        }
        
        .insight-card-blue:hover {
            box-shadow: 0 8px 25px rgba(77, 163, 255, 0.2);
        }
        
        .insight-card-purple {
            border-left: 5px solid #8a5cff;
        }
        
        .insight-card-purple:hover {
            box-shadow: 0 8px 25px rgba(138, 92, 255, 0.2);
        }
        
        .insight-card-teal {
            border-left: 5px solid #00c2a8;
        }
        
        .insight-card-teal:hover {
            box-shadow: 0 8px 25px rgba(0, 194, 168, 0.2);
        }
        
        .insight-card-pink {
            border-left: 5px solid #ff5fa2;
        }
        
        .insight-card-pink:hover {
            box-shadow: 0 8px 25px rgba(255, 95, 162, 0.2);
        }
        
        .insight-card-title {
            font-size: 18px;
            font-weight: 700;
            color: #e0e0e0;
            margin: 0 0 15px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .insight-card-stat {
            font-size: 28px;
            font-weight: 800;
            margin: 12px 0;
            line-height: 1.2;
        }
        
        .insight-card-stat-blue {
            color: #4da3ff;
        }
        
        .insight-card-stat-purple {
            color: #8a5cff;
        }
        
        .insight-card-stat-teal {
            color: #00c2a8;
        }
        
        .insight-card-stat-pink {
            color: #ff5fa2;
        }
        
        .insight-card-text {
            color: #b0b0b0;
            font-size: 13px;
            line-height: 1.6;
            margin: 12px 0;
        }
        
        .insight-card-list {
            list-style: none;
            padding: 0;
            margin: 12px 0;
        }
        
        .insight-card-list li {
            color: #a0a0a0;
            font-size: 13px;
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .insight-card-list li:before {
            content: "▸";
            position: absolute;
            left: 0;
            font-size: 16px;
        }
        
        .insight-card-blue .insight-card-list li:before {
            color: #4da3ff;
        }
        
        .insight-card-purple .insight-card-list li:before {
            color: #8a5cff;
        }
        
        .insight-card-teal .insight-card-list li:before {
            color: #00c2a8;
        }
        
        .insight-card-pink .insight-card-list li:before {
            color: #ff5fa2;
        }
        
        .business-section {
            background: linear-gradient(135deg, #151933 0%, #0f1429 100%);
            border: 1px solid #2a3050;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        
        .business-section h3 {
            color: #e0e0e0 !important;
            font-size: 20px;
            margin-bottom: 15px;
        }
        
        .recommendation-item {
            background: rgba(77, 163, 255, 0.05);
            border-left: 3px solid #4da3ff;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .recommendation-item.priority-1 {
            border-left-color: #ff5fa2;
            background: rgba(255, 95, 162, 0.05);
        }
        
        .recommendation-item.priority-2 {
            border-left-color: #8a5cff;
            background: rgba(138, 92, 255, 0.05);
        }
        
        .recommendation-item.priority-3 {
            border-left-color: #00c2a8;
            background: rgba(0, 194, 168, 0.05);
        }
        
        .recommendation-item.quick-win {
            border-left-color: #4da3ff;
            background: rgba(77, 163, 255, 0.05);
        }
        
        .recommendation-title {
            color: #e0e0e0;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .recommendation-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .recommendation-list li {
            color: #b0b0b0;
            font-size: 13px;
            padding: 4px 0;
            padding-left: 18px;
            position: relative;
        }
        
        .recommendation-list li:before {
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
        }
        
        .recommendation-item.priority-1 .recommendation-list li:before {
            color: #ff5fa2;
        }
        
        .recommendation-item.priority-2 .recommendation-list li:before {
            color: #8a5cff;
        }
        
        .recommendation-item.priority-3 .recommendation-list li:before {
            color: #00c2a8;
        }
        
        .recommendation-item.quick-win .recommendation-list li:before {
            color: #4da3ff;
        }
        
        .theme-selector {
            display: flex;
            gap: 8px;
            justify-content: center;
            padding: 15px 0;
        }
        
        .theme-button {
            flex: 1;
            padding: 10px;
            border: 1px solid #2a3050;
            border-radius: 8px;
            background: #1a1f3a;
            color: #b0b0b0;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s;
            text-align: center;
        }
        
        .theme-button:hover {
            border-color: #4da3ff;
            color: #4da3ff;
            background: rgba(77, 163, 255, 0.1);
        }
        
        .theme-button.active {
            background: linear-gradient(135deg, #4da3ff 0%, #8a5cff 100%);
            border-color: #8a5cff;
            color: white;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #1a1f3a !important;
            border: 1px solid #2a3050 !important;
            color: #e0e0e0 !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #222847 !important;
            border-color: #4da3ff !important;
        }
        
        /* Summary cards for insights */
        .summary-card {
            background: linear-gradient(135deg, #1a3a52 0%, #0f1f38 100%);
            border: 1px solid #2a5a7a;
            border-left: 4px solid #4da3ff;
            padding: 16px;
            border-radius: 10px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }
        
        .summary-card:hover {
            border-left-color: #8a5cff;
            box-shadow: 0 4px 12px rgba(77, 163, 255, 0.15);
        }
        
        .summary-card-title {
            font-size: 13px;
            font-weight: 600;
            color: #a0a0a0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        
        .summary-card-value {
            font-size: 20px;
            font-weight: 700;
            color: #4da3ff;
        }
        
        .summary-card-desc {
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }
        </style>
        """
    else:
        # Light Mode
        return """
        <style>
        :root {
            --primary: #1f77b4;
            --secondary: #ff7f0e;
            --success: #2ca02c;
            --danger: #d62728;
            --info: #17becf;
            --light-bg: #f5f7fa;
            --dark-text: #1a1a1a;
        }
        
        .stApp { background-color: #f5f7fa; }
        
        h1 { color: #1a1a1a !important; font-weight: 800; font-size: 36px; margin: 30px 0 15px 0; }
        h2 { color: #1a1a1a !important; font-weight: 700; font-size: 26px; margin: 25px 0 12px 0; }
        h3 { color: #1a1a1a !important; font-weight: 600; font-size: 20px; margin: 20px 0 10px 0; }
        p { color: #333 !important; }
        
        [data-testid="stSidebar"] { 
            background: linear-gradient(180deg, #1f77b4 0%, #1a5fa0 100%) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { 
            color: white !important; 
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-top: 4px solid #1f77b4;
            margin-bottom: 10px;
        }
        
        .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
        .metric-value { font-size: 32px; font-weight: 800; color: #1f77b4; margin: 5px 0; }
        .metric-delta { font-size: 12px; color: #666; margin-top: 5px; }
        
        .stTabs [data-baseweb="tab"] { background-color: #f0f0f0; border-radius: 10px 10px 0 0; }
        .stTabs [aria-selected="true"] { background-color: #1f77b4 !important; color: white !important; }
        
        .insight-box {
            background: #f0f4ff;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .insight-box.warning { background: #fff8f0; border-left-color: #ff7f0e; }
        .insight-box.danger { background: #fff0f0; border-left-color: #d62728; }
        .insight-box.success { background: #f0fff4; border-left-color: #2ca02c; }
        </style>
        """

st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def categorize_risk(probability):
    """Categorize risk based on probability"""
    if probability < 0.28:
        return "Low Risk", "success"
    elif probability <= 0.70:
        return "Medium Risk", "warning"
    else:
        return "High Risk", "danger"

def prepare_input_for_prediction(input_dict):
    """Prepare input data for model prediction"""
    X = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in feature_columns:
        if col in input_dict:
            X[col] = input_dict[col]
    return X

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 25px 0 20px 0;'>
            <h2 style='
                color: #ffffff; 
                margin: 0; 
                font-size: 32px; 
                font-weight: 900;
                letter-spacing: 2px;
                background: linear-gradient(90deg, #4da3ff 0%, #8a5cff 50%, #ff5fa2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            '>
                HR ANALYTICS
            </h2>
            <p style='
                color: rgba(200, 200, 220, 0.95); 
                margin: 12px 0 0 0; 
                font-size: 12px;
                font-weight: 500;
                letter-spacing: 0.5px;
            '>
                Data-Driven Talent Analytics
            </p>
            <div style='
                width: 60px; 
                height: 2px; 
                background: linear-gradient(90deg, #4da3ff, #00c2a8); 
                margin: 15px auto 0 auto;
                border-radius: 1px;
            '></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with improved styling
    st.markdown("""
        <p style='color: #888; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin: 0 0 12px 0; text-transform: uppercase;'></p>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["Dashboard", "EDA Analysis", "Model Performance", "Prediction", "Insights", "About Me"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("**Quick Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Data", f"{len(df_raw):,}")
    with col2:
        at_risk = (df_raw['target'].sum() / len(df_raw) * 100)
        st.metric("At-Risk", f"{at_risk:.1f}%")
    
    st.markdown("---")
    
    # Theme Selector at Bottom
    st.markdown("<p style='text-align: center; color: #888; font-size: 11px; margin-top: 40px; margin-bottom: 10px;'><strong>THEME</strong></p>", unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("🌙 Dark", use_container_width=True, key="theme_dark"):
            st.session_state.dark_mode = True
            st.rerun()
    with col_t2:
        if st.button("☀️ Light", use_container_width=True, key="theme_light"):
            st.session_state.dark_mode = False
            st.rerun()

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "Dashboard":
    st.markdown("""
        <h1>📊 HR Analytics Dashboard</h1>
        <p style='color: #b0b0b0; font-size: 16px; margin: 0 0 30px 0;'>
            Talent risk assessment and data-driven insights
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # KPI Cards
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Candidates</div>
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-delta">Complete dataset</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        job_changers = df_raw['target'].sum()
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #ff5fa2;">
            <div class="metric-label">Job Changers</div>
            <div class="metric-value" style="color: #ff5fa2;">{job_changers:,}</div>
            <div class="metric-delta">{job_changers/len(df_raw)*100:.1f}% At-Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        retention = ((len(df_raw) - job_changers) / len(df_raw) * 100)
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #00c2a8;">
            <div class="metric-label">Retention Rate</div>
            <div class="metric-value" style="color: #00c2a8;">{retention:.1f}%</div>
            <div class="metric-delta">Stable workforce</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_training = df_raw['training_hours'].mean()
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #4da3ff;">
            <div class="metric-label">Avg Training Hours</div>
            <div class="metric-value" style="color: #4da3ff;">{avg_training:.0f}</div>
            <div class="metric-delta">Per year per employee</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Job Change Distribution", "🔍 Feature Analysis", "📈 Training Impact", "🏢 Company Insights"])
    
    with tab1:
        st.markdown("### Target Variable Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = df_raw['target'].value_counts()
            fig_count = go.Figure(data=[
                go.Bar(
                    x=['No Change (0)', 'Seek Change (1)'],
                    y=[target_counts[0], target_counts[1]],
                    marker_color=['#00c2a8', '#ff5fa2'],
                    text=[f"{target_counts[0]:,}", f"{target_counts[1]:,}"],
                    textposition='outside'
                )
            ])
            fig_count.update_layout(
                height=350,
                xaxis_title='',
                yaxis_title='Count',
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_count, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=target_counts.values,
                labels=['No Change', 'Seek Change'],
                color_discrete_sequence=['#00c2a8', '#ff5fa2'],
                hole=0.4
            )
            fig_pie.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        ratio = target_counts[0] / target_counts[1]
        st.markdown(f"""
        <div class="insight-box">
            <strong>Class Imbalance Ratio:</strong> 1:{ratio:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Experience Level Analysis")
        
        exp_map = {
            '<1': 0.5, '1-3': 2, '4-6': 5, '7-10': 8.5,
            '11-15': 13, '16-20': 18, '>20': 25
        }
        df_exp = df_raw.copy()
        df_exp['exp_years'] = df_exp['experience'].map(exp_map)
        df_exp['exp_category'] = pd.cut(
            df_exp['exp_years'],
            bins=[0, 3, 8, 15, 30],
            labels=['Junior (0-3yr)', 'Mid (4-8yr)', 'Senior (9-15yr)', 'Expert (15+yr)']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            exp_rate = df_exp.groupby('exp_category')['target'].agg(['count', 'sum'])
            exp_rate['rate'] = (exp_rate['sum'] / exp_rate['count'] * 100)
            exp_rate = exp_rate.sort_values('rate', ascending=False)
            
            fig_exp = go.Figure(data=[go.Bar(
                x=exp_rate.index,
                y=exp_rate['rate'].values,
                marker=dict(color=['#ff5fa2', '#8a5cff', '#4da3ff', '#00c2a8']),
                text=exp_rate['rate'].round(1),
                textposition='outside'
            )])
            fig_exp.update_layout(
                height=350,
                xaxis_title='',
                yaxis_title='Job Change Rate (%)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        
        with col2:
            edu_rate = df_raw.groupby('education_level')['target'].agg(['count', 'sum'])
            edu_rate['rate'] = (edu_rate['sum'] / edu_rate['count'] * 100)
            edu_rate = edu_rate.sort_values('rate', ascending=False)
            
            fig_edu = go.Figure(data=[go.Bar(
                x=edu_rate.index,
                y=edu_rate['rate'].values,
                marker=dict(color='#8a5cff'),
                text=edu_rate['rate'].round(1),
                textposition='outside'
            )])
            fig_edu.update_layout(
                height=350,
                xaxis_title='',
                yaxis_title='Job Change Rate (%)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_edu, use_container_width=True)
    
    with tab3:
        st.markdown("### Training Hours Impact on Retention")
        
        training_stayers = df_raw[df_raw['target'] == 0]['training_hours'].dropna()
        training_changers = df_raw[df_raw['target'] == 1]['training_hours'].dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #00c2a8;">
                <div class="metric-label">Employees Staying</div>
                <div class="metric-value" style="color: #00c2a8;">{training_stayers.mean():.1f}</div>
                <div class="metric-delta">Avg training hours/year</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #ff5fa2;">
                <div class="metric-label">Job Changers</div>
                <div class="metric-value" style="color: #ff5fa2;">{training_changers.mean():.1f}</div>
                <div class="metric-delta">Avg training hours/year</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            diff = training_stayers.mean() - training_changers.mean()
            pct = (diff / training_changers.mean() * 100)
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: #4da3ff;">
                <div class="metric-label">Training Gap</div>
                <div class="metric-value" style="color: #4da3ff;">{diff:.1f}h</div>
                <div class="metric-delta">{pct:.0f}% difference</div>
            </div>
            """, unsafe_allow_html=True)
        
        fig_train = go.Figure()
        fig_train.add_trace(go.Histogram(
            x=training_stayers, name='Staying',
            marker=dict(color='#00c2a8', opacity=0.7), nbinsx=30
        ))
        fig_train.add_trace(go.Histogram(
            x=training_changers, name='Seeking Change',
            marker=dict(color='#ff5fa2', opacity=0.7), nbinsx=30
        ))
        fig_train.update_layout(
            height=350, barmode='overlay', hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Training Hours', yaxis_title='Count'
        )
        st.plotly_chart(fig_train, use_container_width=True)
    
    with tab4:
        st.markdown("### Company & Enrollment Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'company_type' in df_raw.columns:
                company_rate = df_raw.groupby('company_type')['target'].agg(['count', 'sum'])
                company_rate['rate'] = (company_rate['sum'] / company_rate['count'] * 100)
                company_rate = company_rate.sort_values('rate', ascending=False)
                
                fig_co = go.Figure(data=[go.Bar(
                    y=company_rate.index,
                    x=company_rate['rate'].values,
                    orientation='h',
                    marker=dict(color=company_rate['rate'].values, colorscale=[[0, '#4da3ff'], [0.5, '#8a5cff'], [1, '#ff5fa2']]),
                    text=company_rate['rate'].round(1),
                    textposition='outside'
                )])
                fig_co.update_layout(
                    height=350, xaxis_title='Job Change Rate (%)',
                    yaxis_title='', paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_co, use_container_width=True)
        
        with col2:
            if 'enrolled_university' in df_raw.columns:
                enroll_rate = df_raw.groupby('enrolled_university')['target'].agg(['count', 'sum'])
                enroll_rate['rate'] = (enroll_rate['sum'] / enroll_rate['count'] * 100)
                enroll_rate = enroll_rate.sort_values('rate', ascending=False)
                
                fig_enr = go.Figure(data=[go.Bar(
                    x=enroll_rate.index,
                    y=enroll_rate['rate'].values,
                    marker=dict(color=enroll_rate['rate'].values, colorscale=[[0, '#14b8a6'], [0.5, '#a855f7'], [1, '#ec4899']]),
                    text=enroll_rate['rate'].round(1),
                    textposition='outside'
                )])
                fig_enr.update_layout(
                    height=350, yaxis_title='Job Change Rate (%)',
                    xaxis_title='', paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_enr, use_container_width=True)

# ============================================================================
# PAGE 2: EDA ANALYSIS WITH FILTERS
# ============================================================================
elif page == "EDA Analysis":
    st.markdown("""
        <h1>📊 Exploratory Data Analysis</h1>
        <p style='color: #b0b0b0; font-size: 16px; margin: 0 0 30px 0;'>
            Deep-dive analysis with interactive filters
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 1: DATASET OVERVIEW METRICS
    st.markdown("### 📈 Dataset Overview")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3, gap="medium")
    
    with metric_col1:
        st.metric("Total Records", f"{len(df_raw):,}")
    with metric_col2:
        st.metric("Total Features", len(df_raw.columns))
    with metric_col3:
        missing_pct = (df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns)) * 100)
        st.metric("Missing Data %", f"{missing_pct:.2f}%")
    
    st.markdown("---")
    
    # SECTION 2: ANALYSIS FILTERS WITH CLEAN LAYOUT
    st.markdown("### 🔍 Analysis Filters")
    
    with st.expander("Expand Filters", expanded=True):
        st.markdown("<div style='padding: 5px 0;'></div>", unsafe_allow_html=True)
        
        # First row of filters - 3 column layout
        f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
        
        with f_col1:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Experience Level</label>", unsafe_allow_html=True)
            selected_exp = st.multiselect(
                "Experience Level",
                ["<1", "1-3", "4-6", "7-10", "11-15", "16-20", ">20"],
                default=["<1", "1-3", "4-6", "7-10", "11-15", "16-20", ">20"],
                label_visibility="collapsed",
                key="exp_filter"
            )
        
        with f_col2:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Education Level</label>", unsafe_allow_html=True)
            selected_edu = st.multiselect(
                "Education Level",
                df_raw['education_level'].unique().tolist(),
                default=df_raw['education_level'].unique().tolist(),
                label_visibility="collapsed",
                key="edu_filter"
            )
        
        with f_col3:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Gender</label>", unsafe_allow_html=True)
            selected_gender = st.multiselect(
                "Gender",
                df_raw['gender'].unique().tolist(),
                default=df_raw['gender'].unique().tolist(),
                label_visibility="collapsed",
                key="gender_filter"
            )
        
        st.markdown("<div style='padding: 8px 0;'></div>", unsafe_allow_html=True)
        
        # Second row of filters
        f_col4, f_col5, f_col6 = st.columns(3, gap="medium")
        
        with f_col4:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Training Hours (Min)</label>", unsafe_allow_html=True)
            min_training = st.slider("Min Training Hours", 0, int(df_raw['training_hours'].max()), 0, label_visibility="collapsed", key="min_training")
        
        with f_col5:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Training Hours (Max)</label>", unsafe_allow_html=True)
            max_training = st.slider("Max Training Hours", 0, int(df_raw['training_hours'].max()), int(df_raw['training_hours'].max()), label_visibility="collapsed", key="max_training")
        
        with f_col6:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Job Status</label>", unsafe_allow_html=True)
            job_change_filter = st.multiselect(
                "Job Change Status",
                [("Staying", 0), ("Seeking Change", 1)],
                default=[("Staying", 0), ("Seeking Change", 1)],
                format_func=lambda x: x[0],
                label_visibility="collapsed",
                key="job_status_filter"
            )
    
    # Apply filters
    job_change_values = [item[1] for item in job_change_filter]
    df_filtered = df_raw[
        (df_raw['experience'].isin(selected_exp)) &
        (df_raw['education_level'].isin(selected_edu)) &
        (df_raw['gender'].isin(selected_gender)) &
        (df_raw['training_hours'] >= min_training) &
        (df_raw['training_hours'] <= max_training) &
        (df_raw['target'].isin(job_change_values))
    ]
    
    filtered_pct = (len(df_filtered)/len(df_raw)*100) if len(df_raw) > 0 else 0
    st.markdown(f"<p style='color: #4da3ff; font-weight: 600; font-size: 13px; margin: 12px 0;'>✓ Filtered Results: {len(df_filtered):,} records ({filtered_pct:.1f}% of total)</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 3: QUICK INSIGHTS
    st.markdown("### 🎯 Quick Insights")
    
    ins_c1, ins_c2, ins_c3 = st.columns(3, gap="medium")
    
    with ins_c1:
        changer_pct = (df_filtered['target'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-card-title">Job Change Rate</div>
            <div class="summary-card-value">{changer_pct:.1f}%</div>
            <div class="summary-card-desc">{int(df_filtered['target'].sum())} seeking change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with ins_c2:
        avg_train = df_filtered['training_hours'].mean() if len(df_filtered) > 0 else 0
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-card-title">Avg Training Hours</div>
            <div class="summary-card-value">{avg_train:.0f}</div>
            <div class="summary-card-desc">Per year per employee</div>
        </div>
        """, unsafe_allow_html=True)
    
    with ins_c3:
        avg_exp = df_filtered['city_development_index'].mean() if len(df_filtered) > 0 else 0
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-card-title">City Development</div>
            <div class="summary-card-value">{avg_exp:.2f}</div>
            <div class="summary-card-desc">Development index</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 4: MISSING VALUES ANALYSIS WITH ERROR HANDLING
    st.markdown("### 📋 Missing Values Analysis")
    
    try:
        # Create missing data dataframe with proper type conversion
        missing_counts = df_filtered.isnull().sum()
        missing_pct = (missing_counts / len(df_filtered) * 100).round(2) if len(df_filtered) > 0 else 0
        
        missing_data = pd.DataFrame({
            'Column': [str(col) for col in df_filtered.columns],  # Convert to string
            'Missing': missing_counts.values.astype(int),  # Convert to int
            'Percentage': missing_pct.values.astype(float)  # Convert to float
        }).sort_values('Missing', ascending=False)
        
        missing_data = missing_data[missing_data['Missing'] > 0]
        
        if len(missing_data) > 0:
            data_col, chart_col = st.columns(2, gap="medium")
            
            with data_col:
                st.markdown("**Missing Values Table**")
                # Create a clean dataframe for display
                display_df = missing_data.reset_index(drop=True).copy()
                display_df = display_df.astype({'Column': str, 'Missing': int, 'Percentage': float})
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with chart_col:
                try:
                    fig_missing = px.bar(
                        missing_data.reset_index(drop=True), 
                        x='Percentage', 
                        y='Column', 
                        orientation='h',
                        color='Percentage', 
                        color_continuous_scale=[[0, '#4da3ff'], [0.5, '#8a5cff'], [1, '#ff5fa2']],
                        title="Missing Values %"
                    )
                    fig_missing.update_layout(
                        height=300, 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        margin=dict(l=50, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_missing, use_container_width=True, key="missing_chart")
                except Exception as chart_error:
                    st.warning(f"Could not render chart: {str(chart_error)}")
        else:
            st.info("✓ No missing values in filtered dataset")
    
    except Exception as e:
        st.error(f"Error analyzing missing values: {str(e)}")
    
    st.markdown("---")
    
    # SECTION 5: NUMERICAL FEATURES DISTRIBUTION
    st.markdown("### 📊 Numerical Features Distribution")
    
    try:
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 10px; font-size: 13px;'>Select Numerical Features</label>", unsafe_allow_html=True)
        selected_numeric = st.multiselect(
            "Select Numerical Features",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) > 0 else [],
            label_visibility="collapsed",
            key="numeric_selector"
        )
        
        if selected_numeric and len(df_filtered) > 0:
            stats_col, viz_col = st.columns(2, gap="medium")
            
            with stats_col:
                st.markdown("**Distribution Statistics**")
                stats_df = pd.DataFrame({
                    'Feature': [str(col) for col in selected_numeric],  # Convert to string
                    'Mean': [float(df_filtered[col].mean()) for col in selected_numeric],
                    'Median': [float(df_filtered[col].median()) for col in selected_numeric],
                    'Std': [float(df_filtered[col].std()) for col in selected_numeric],
                    'Min': [float(df_filtered[col].min()) for col in selected_numeric],
                    'Max': [float(df_filtered[col].max()) for col in selected_numeric]
                }).round(2)
                # Ensure all columns are proper types
                stats_df = stats_df.astype({'Feature': str, 'Mean': float, 'Median': float, 'Std': float, 'Min': float, 'Max': float})
                st.dataframe(stats_df.reset_index(drop=True), use_container_width=True, hide_index=True)
            
            with viz_col:
                st.markdown("**Distribution Visualization**")
                selected_for_viz = st.selectbox("Select feature to visualize", selected_numeric, key="feature_viz")
                
                try:
                    if len(df_filtered) > 0:
                        fig_hist = px.histogram(
                            df_filtered, 
                            x=selected_for_viz, 
                            nbins=30,
                            color_discrete_sequence=['#8a5cff'],
                            title=f"{selected_for_viz} Distribution"
                        )
                        fig_hist.update_layout(
                            height=300, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False,
                            margin=dict(l=50, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, key="hist_chart")
                    else:
                        st.warning("No data available for visualization")
                except Exception as chart_error:
                    st.warning(f"Could not render histogram: {str(chart_error)}")
        else:
            if len(numeric_cols) == 0:
                st.info("No numerical features available in dataset")
            elif len(df_filtered) == 0:
                st.warning("No records match the selected filters")
    
    except Exception as e:
        st.error(f"Error in numerical features analysis: {str(e)}")
    
    st.markdown("---")
    
    # SECTION 6: CATEGORICAL FEATURES ANALYSIS
    st.markdown("### 🏷️ Categorical Features Analysis")
    
    try:
        categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols and len(df_filtered) > 0:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 10px; font-size: 13px;'>Select Categorical Feature</label>", unsafe_allow_html=True)
            selected_cat = st.selectbox(
                "Select Categorical Feature",
                categorical_cols,
                label_visibility="collapsed",
                key="cat_selector"
            )
            
            try:
                # Value counts
                value_counts = df_filtered[selected_cat].value_counts()
                
                left_col, right_col = st.columns(2, gap="medium")
                
                with left_col:
                    st.markdown("**Value Distribution**")
                    dist_df = pd.DataFrame({
                        'Category': [str(cat) for cat in value_counts.index],  # Convert to string
                        'Count': value_counts.values.astype(int),  # Convert to int
                        'Percentage': (value_counts.values / value_counts.sum() * 100).round(2).astype(float)  # Convert to float
                    }).reset_index(drop=True)
                    # Ensure proper types
                    dist_df = dist_df.astype({'Category': str, 'Count': int, 'Percentage': float})
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                
                with right_col:
                    try:
                        fig_cat = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            color_discrete_sequence=['#4da3ff', '#8a5cff', '#00c2a8', '#ff5fa2', '#06b6d4'],
                            title=f"{selected_cat} Distribution"
                        )
                        fig_cat.update_layout(
                            height=300,
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_cat, use_container_width=True, key="cat_pie_chart")
                    except Exception as chart_error:
                        st.warning(f"Could not render pie chart: {str(chart_error)}")
            
            except Exception as cat_error:
                st.error(f"Error analyzing categorical feature: {str(cat_error)}")
        
        elif len(df_filtered) == 0:
            st.warning("No records match the selected filters")
        else:
            st.info("No categorical features available in dataset")
    
    except Exception as e:
        st.error(f"Error in categorical features analysis: {str(e)}")

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.markdown("""
        <h1>🤖 Machine Learning Model</h1>
        <p style='color: #b0b0b0; font-size: 16px; margin: 0 0 30px 0;'>
            Model performance metrics and evaluation
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== MODEL SUMMARY SECTION ==========
    st.markdown("<h3 style='margin-top: 35px; margin-bottom: 25px;'>Model Configuration</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a3a52 0%, #1a2838 100%);
            border: 1px solid #2d5a7a;
            border-left: 4px solid #06b6d4;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Algorithm</p>
            <p style='color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0; line-height: 1.4;'>Gradient Boosting<br>Classifier</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #2a3a52 0%, #1a2838 100%);
            border: 1px solid #2d5a7a;
            border-left: 4px solid #8a5cff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Feature Engineering</p>
            <p style='color: #e0e0e0; font-size: 18px; font-weight: 700; margin: 0;'>37</p>
            <p style='color: #b0b0b0; font-size: 12px; margin: 8px 0 0 0;'>Engineered Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a4a52 0%, #0a2a38 100%);
            border: 1px solid #2d5a7a;
            border-left: 4px solid #00c2a8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Model Status</p>
            <p style='color: #00c2a8; font-size: 18px; font-weight: 700; margin: 0;'>✓ Production Ready</p>
            <p style='color: #b0b0b0; font-size: 12px; margin: 8px 0 0 0;'>Trained & Validated</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
            border: 1px solid #3d2a7a;
            border-left: 4px solid #ec4899;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Dataset Split</p>
            <p style='color: #e0e0e0; font-size: 14px; font-weight: 600; margin: 0;'>15,268 Train | 3,832 Test</p>
            <p style='color: #b0b0b0; font-size: 12px; margin: 8px 0 0 0;'>80% / 20% Split</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    
    # ========== PERFORMANCE METRICS SECTION ==========
    st.markdown("<h3 style='margin-top: 35px; margin-bottom: 25px;'>Performance Metrics</h3>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")
    
    with col_m1:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a3a52 0%, #1a2838 100%);
            border: 1px solid #2d5a7a;
            border-top: 3px solid #4da3ff;
            padding: 25px 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        '>
            <p style='color: #4da3ff; font-size: 36px; font-weight: 800; margin: 0;'>79.87</p>
            <p style='color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 10px 0 0 0;'>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #2a2a52 0%, #1a1a38 100%);
            border: 1px solid #3d2a7a;
            border-top: 3px solid #8a5cff;
            padding: 25px 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #8a5cff; font-size: 36px; font-weight: 800; margin: 0;'>60.13</p>
            <p style='color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 10px 0 0 0;'>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a4a4f 0%, #0a2a2f 100%);
            border: 1px solid #2d5a5f;
            border-top: 3px solid #00c2a8;
            padding: 25px 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #00c2a8; font-size: 36px; font-weight: 800; margin: 0;'>65.80</p>
            <p style='color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 10px 0 0 0;'>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
            border: 1px solid #3d2a7a;
            border-top: 3px solid #ff5fa2;
            padding: 25px 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        '>
            <p style='color: #ff5fa2; font-size: 36px; font-weight: 800; margin: 0;'>0.814</p>
            <p style='color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 10px 0 0 0;'>AUC-ROC</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    
    # ========== MODEL INSIGHTS SECTION ==========
    st.markdown("<h3 style='margin-top: 35px; margin-bottom: 25px;'>Key Performance Insights</h3>", unsafe_allow_html=True)
    
    col_insight1, col_insight2 = st.columns([1, 1])
    
    with col_insight1:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a3a52 0%, #1a2838 100%);
            border: 1px solid #2d5a7a;
            border-left: 4px solid #4da3ff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            height: 100%;
        '>
            <p style='color: #4da3ff; font-size: 15px; font-weight: 700; margin: 0 0 10px 0;'>Excellent Discrimination Power</p>
            <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.6;'>AUC of 0.814 demonstrates strong ability to distinguish between job changers and stayers across all probability thresholds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_insight2:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1a4a4f 0%, #0a2a2f 100%);
            border: 1px solid #2d5a5f;
            border-left: 4px solid #00c2a8;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            height: 100%;
        '>
            <p style='color: #00c2a8; font-size: 15px; font-weight: 700; margin: 0 0 10px 0;'>Balanced Performance</p>
            <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.6;'>Precision (60%) and Recall (66%) are well-balanced, effectively minimizing both false positives and negatives</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col_insight3 = st.columns(1)[0]
    with col_insight3:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
            border: 1px solid #3d2a7a;
            border-left: 4px solid #ff5fa2;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        '>
            <p style='color: #ff5fa2; font-size: 15px; font-weight: 700; margin: 0 0 10px 0;'>Production-Optimized Threshold</p>
            <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.6;'>Decision boundary set at 0.28 to maximize recall, capturing 66% of actual job changers for proactive workforce intervention</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 4: PREDICTION
# ============================================================================
elif page == "Prediction":
    st.markdown("""
        <h1>⚠️ Individual Risk Prediction</h1>
        <p style='color: #b0b0b0; font-size: 16px; margin: 0 0 30px 0;'>
            Assess job change risk for individual employees
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.form("employee_form"):
        # SECTION 1: EMPLOYEE PROFILE
        st.markdown("<h3 style='color: #e0e0e0; margin: 20px 0 25px 0;'>👤 Employee Profile</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Gender</label>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], label_visibility="collapsed", key="gender_input")
        
        with col2:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Education Level</label>", unsafe_allow_html=True)
            education = st.selectbox("Education Level",
                ["High School", "Graduate", "Masters", "Phd"],
                label_visibility="collapsed", key="education_input")
        
        with col3:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Major Discipline</label>", unsafe_allow_html=True)
            major = st.selectbox("Major Discipline",
                ["STEM", "Business Degree", "Arts", "Humanities", "No Major"],
                label_visibility="collapsed", key="major_input")
        
        with col4:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Years of Experience</label>", unsafe_allow_html=True)
            experience = st.selectbox("Years of Experience", 
                ["<1", "1-3", "4-6", "7-10", "11-15", "16-20", ">20"],
                label_visibility="collapsed", key="experience_input")
        
        st.markdown("<div style='padding: 10px 0;'></div>", unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4, gap="medium")
        
        with col5:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>City Development Index</label>", unsafe_allow_html=True)
            city_dev = st.slider("City Development Index", 0.0, 1.0, 0.5, label_visibility="collapsed", key="city_dev", step=0.01)
        
        with col6:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Relevant Experience</label>", unsafe_allow_html=True)
            rel_exp = st.selectbox("Relevant Experience",
                ["Has relevent experience", "No relevent experience"],
                label_visibility="collapsed", key="rel_exp_input")
        
        with col7:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Company Size</label>", unsafe_allow_html=True)
            company_size = st.selectbox("Company Size",
                ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"],
                label_visibility="collapsed", key="company_size_input")
        
        with col8:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Company Type</label>", unsafe_allow_html=True)
            company_type = st.selectbox("Company Type",
                ["Pvt Ltd", "Public Sector", "Funded Startup", "Early Stage Startup", "Other", "NGO"],
                label_visibility="collapsed", key="company_type_input")
        
        st.markdown("---")
        
        # SECTION 2: DEVELOPMENT & TRAINING
        st.markdown("<h3 style='color: #e0e0e0; margin: 20px 0 25px 0;'>📚 Development & Training</h3>", unsafe_allow_html=True)
        
        col9, col10, col11, col12 = st.columns(4, gap="medium")
        
        with col9:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Training Hours per Year</label>", unsafe_allow_html=True)
            training_hours = st.number_input("Training Hours per Year", 0, 300, 60, 5, label_visibility="collapsed", key="training_input")
        
        with col10:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Years Since Last Job Change</label>", unsafe_allow_html=True)
            last_job = st.selectbox("Years Since Last Job Change",
                ["never", "<1", "1", "2", "3", "4", ">4"],
                label_visibility="collapsed", key="last_job_input")
        
        with col11:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'>Enrollment Status</label>", unsafe_allow_html=True)
            enrollment = st.selectbox("Enrollment Status",
                ["no_enrollment", "Part time course", "Full time course"],
                label_visibility="collapsed", key="enrollment_input")
        
        with col12:
            st.markdown("<label style='color: #ffffff; font-weight: 700; display: block; margin-bottom: 8px; font-size: 13px;'> </label>", unsafe_allow_html=True)
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='padding: 15px 0;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 PREDICT RISK", use_container_width=True)
    
    if submitted:
        exp_map = {'<1': 0.5, '1-3': 2, '4-6': 5, '7-10': 8.5, '11-15': 13, '16-20': 18, '>20': 25}
        last_job_map = {'never': 0, '<1': 0.5, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5}
        
        try:
            input_dict = {
                'city_development_index': city_dev,
                'experience_years': exp_map[experience],
                'training_hours': training_hours,
                'last_new_job_years': last_job_map[last_job],
            }
            
            for col in feature_columns:
                if 'gender' in col:
                    val = 1 if col.endswith(gender) else 0
                    input_dict[col] = val
                elif 'education' in col:
                    val = 1 if col.endswith(education) else 0
                    input_dict[col] = val
                elif 'major' in col:
                    val = 1 if col.endswith(major) else 0
                    input_dict[col] = val
                elif 'relevent' in col:
                    val = 1 if col.endswith(rel_exp.split()[0]) else 0
                    input_dict[col] = val
                elif 'company_size' in col:
                    val = 1 if col.endswith(company_size) else 0
                    input_dict[col] = val
                elif 'company_type' in col:
                    val = 1 if col.endswith(company_type) else 0
                    input_dict[col] = val
                elif 'enrolled' in col:
                    val = 1 if col.endswith(enrollment) else 0
                    input_dict[col] = val
            
            X_input = prepare_input_for_prediction(input_dict)
            X_scaled = scaler.transform(X_input)
            prob = model.predict_proba(X_scaled)[0][1]
            confidence = model.predict_proba(X_scaled)[0].max()
            risk_label, risk_type = categorize_risk(prob)
            
            st.markdown("---")
            st.markdown("### 📈 Risk Assessment")
            
            col_risk1, col_risk2, col_risk3 = st.columns([1, 1, 1], gap="medium")
            
            risk_colors = {"success": "#00c2a8", "warning": "#ff5fa2", "danger": "#8a5cff"}
            
            with col_risk1:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #1a3a52 0%, #1a2838 100%);
                    border: 1px solid #2d5a7a;
                    border-top: 3px solid {risk_colors[risk_type]};
                    padding: 25px 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                '>
                    <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Risk Level</p>
                    <p style='color: {risk_colors[risk_type]}; font-size: 28px; font-weight: 800; margin: 0;'>{risk_label}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_risk2:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #2a2a52 0%, #1a1a38 100%);
                    border: 1px solid #3d2a7a;
                    border-top: 3px solid #4da3ff;
                    padding: 25px 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                '>
                    <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Probability</p>
                    <p style='color: #4da3ff; font-size: 28px; font-weight: 800; margin: 0;'>{prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_risk3:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
                    border: 1px solid #3d2a7a;
                    border-top: 3px solid #ff5fa2;
                    padding: 25px 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                '>
                    <p style='color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>Confidence</p>
                    <p style='color: #ff5fa2; font-size: 28px; font-weight: 800; margin: 0;'>{confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            try:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    title={"text": "Job Change Probability (%)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': risk_colors[risk_type]},
                        'steps': [
                            {'range': [0, 28], 'color': 'rgba(0, 194, 168, 0.2)'},
                            {'range': [28, 70], 'color': 'rgba(255, 95, 162, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(138, 92, 255, 0.2)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=50, r=50, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")
            except Exception as gauge_error:
                st.warning(f"Could not render gauge chart: {str(gauge_error)}")
            
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            if risk_type == "success":
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, #1a4a4f 0%, #0a2a2f 100%);
                    border: 1px solid #2d5a5f;
                    border-left: 4px solid #00c2a8;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                '>
                    <p style='color: #00c2a8; font-weight: 700; font-size: 15px; margin: 0 0 10px 0;'>✓ Low Risk - Maintain Status</p>
                    <ul style='color: #b0b0b0; margin: 10px 0 0 20px; padding: 0;'>
                        <li>Continue current engagement and development activities</li>
                        <li>Quarterly check-ins to maintain satisfaction</li>
                        <li>Recognize and appreciate contributions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_type == "warning":
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
                    border: 1px solid #3d2a7a;
                    border-left: 4px solid #ff5fa2;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                '>
                    <p style='color: #ff5fa2; font-weight: 700; font-size: 15px; margin: 0 0 10px 0;'>⚠️ Medium Risk - Proactive Action Needed</p>
                    <ul style='color: #b0b0b0; margin: 10px 0 0 20px; padding: 0;'>
                        <li>Schedule career development conversation within 2 weeks</li>
                        <li>Assess training and development satisfaction</li>
                        <li>Review compensation vs market benchmarks</li>
                        <li>Propose new opportunities or stretch assignments</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div style='
                    background: linear-gradient(135deg, #2a1a4a 0%, #1a0a2a 100%);
                    border: 1px solid #3d1a5a;
                    border-left: 4px solid #8a5cff;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                '>
                    <p style='color: #8a5cff; font-weight: 700; font-size: 15px; margin: 0 0 10px 0;'>🚨 High Risk - Urgent Intervention Required</p>
                    <ul style='color: #b0b0b0; margin: 10px 0 0 20px; padding: 0;'>
                        <li>Emergency meeting with HR and direct manager within 5 days</li>
                        <li>Conduct deep career discussion to understand drivers</li>
                        <li>Prepare competitive retention package</li>
                        <li>Weekly monitoring for next 90 days</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# ============================================================================
# PAGE 5: STRATEGIC INSIGHTS
# ============================================================================
elif page == "Insights":
    st.markdown("""
        <h1>💡 Strategic Insights</h1>
        <p style='color: #b0b0b0; font-size: 16px; margin: 0 0 30px 0;'>
            Data-driven insights and actionable recommendations
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights Cards
    st.markdown("<h2 style='margin-bottom: 20px;'>Key Insights</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-card insight-card-blue">
            <div class="insight-card-title">💼 Training Impact on Retention</div>
            <div class="insight-card-stat insight-card-stat-blue">68.8 hrs/year</div>
            <div class="insight-card-text"><strong>Stayers vs Changers</strong></div>
            <ul class="insight-card-list">
                <li>Employees staying receive 68.8 hrs training annually</li>
                <li>Job changers receive only 52.3 hrs - 31% gap</li>
            </ul>
            <div class="insight-card-text" style="margin-top: 15px; font-weight: 600; color: #4da3ff;">→ Training is critical for retention</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card insight-card-pink">
            <div class="insight-card-title">📊 Experience Level Risk</div>
            <div class="insight-card-stat insight-card-stat-pink">35% Attrition</div>
            <div class="insight-card-text"><strong>Junior Staff (0-3 years)</strong></div>
            <ul class="insight-card-list">
                <li>Highest risk demographic - 35% job change rate</li>
                <li>Expert staff (15+ yrs) only 19.8% - stable</li>
            </ul>
            <div class="insight-card-text" style="margin-top: 15px; font-weight: 600; color: #ff5fa2;">→ Focus retention on junior talent</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-card insight-card-teal">
            <div class="insight-card-title">🎓 Education Level Mobility</div>
            <div class="insight-card-stat insight-card-stat-teal">27.1%</div>
            <div class="insight-card-text"><strong>Masters Degree Holders</strong></div>
            <ul class="insight-card-list">
                <li>Higher education correlates with mobility</li>
                <li>High school: 15.2% attrition vs Masters: 27.1%</li>
            </ul>
            <div class="insight-card-text" style="margin-top: 15px; font-weight: 600; color: #00c2a8;">→ Advanced holders have more options</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Business Impact Section
    st.markdown("<h2 style='margin-bottom: 20px;'>Business Impact</h2>", unsafe_allow_html=True)
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        st.markdown("""
        <div class="business-section">
            <h3>Cost of Attrition</h3>
            <div style="margin-top: 15px;">
                <div style="background: rgba(255, 95, 162, 0.1); border-left: 3px solid #ff5fa2; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="color: #ff5fa2; font-weight: 700; font-size: 18px;">80K - 150K</div>
                    <div style="color: #b0b0b0; font-size: 12px; margin-top: 5px;">Cost per junior talent loss</div>
                </div>
                <div style="background: rgba(138, 92, 255, 0.1); border-left: 3px solid #8a5cff; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="color: #8a5cff; font-weight: 700; font-size: 18px;">250K - 400K</div>
                    <div style="color: #b0b0b0; font-size: 12px; margin-top: 5px;">Cost per senior talent loss</div>
                </div>
                <div style="background: rgba(77, 163, 255, 0.1); border-left: 3px solid #4da3ff; padding: 12px 15px; border-radius: 6px;">
                    <div style="color: #4da3ff; font-weight: 700;">Immeasurable</div>
                    <div style="color: #b0b0b0; font-size: 12px; margin-top: 5px;">Knowledge loss & team disruption</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b2:
        st.markdown("""
        <div class="business-section">
            <h3>Retention ROI</h3>
            <div style="margin-top: 15px;">
                <div style="background: rgba(0, 194, 168, 0.1); border-left: 3px solid #00c2a8; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="color: #00c2a8; font-weight: 700; font-size: 18px;">5-7x Return</div>
                    <div style="color: #b0b0b0; font-size: 12px; margin-top: 5px;">1 invested in training = 5-7 in saved costs</div>
                </div>
                <div style="background: rgba(77, 163, 255, 0.1); border-left: 3px solid #4da3ff; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
                    <div style="color: #4da3ff; font-weight: 700;">Highest ROI</div>
                    <div style="color: #b0b0b0; font-size: 12px; margin-top: 5px;">Focus on junior talent retention</div>
                </div>
                <div style="background: rgba(138, 92, 255, 0.1); border-left: 3px solid #8a5cff; padding: 12px 15px; border-radius: 6px;">
                    <div style="color: #8a5cff; font-weight: 700;">20-30% Improvement</div>
                    <div style="color: #b0b0b0; font-size: 12px; margin-top: 5px;">With targeted development programs</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations Section
    st.markdown("<h2 style='margin-bottom: 20px;'>Actionable Recommendations</h2>", unsafe_allow_html=True)
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown("""
        <div class="recommendation-item priority-1">
            <div class="recommendation-title">🎯 Priority 1: Junior Talent (0-3 yrs) - HIGH RISK</div>
            <ul class="recommendation-list">
                <li>Implement structured mentorship program</li>
                <li>Double training budget for junior staff</li>
                <li>Create clear career progression pathways</li>
                <li>Quarterly career development conversations</li>
            </ul>
        </div>
        
        <div class="recommendation-item priority-2">
            <div class="recommendation-title">📈 Priority 2: Mid-Level (4-8 yrs)</div>
            <ul class="recommendation-list">
                <li>Provide leadership training opportunities</li>
                <li>Create clear promotion pathways</li>
                <li>Assign stretch projects and mentoring roles</li>
                <li>Regular compensation benchmarking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown("""
        <div class="recommendation-item priority-3">
            <div class="recommendation-title">⭐ Priority 3: Senior Talent (9+ yrs)</div>
            <ul class="recommendation-list">
                <li>Position as thought leaders and mentors</li>
                <li>Support conference attendance & publications</li>
                <li>Facilitate internal mentoring programs</li>
                <li>Recognize and reward loyalty</li>
            </ul>
        </div>
        
        <div class="recommendation-item quick-win">
            <div class="recommendation-title">⚡ Quick Wins (30 days)</div>
            <ul class="recommendation-list">
                <li>Audit current training spending by level</li>
                <li>Identify high-risk junior talent</li>
                <li>Schedule career conversations with at-risk</li>
                <li>Prepare compensation adjustments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 6: ABOUT ME
# ============================================================================
elif page == "About Me":
    st.markdown("""
        <h1>👋 Tentang Saya</h1>
        <p style='color: #b0b0b0; font-size: 16px; margin: 0 0 30px 0;'>
            Data Analyst & Data Scientist | R&D Department
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content with two columns
    col_intro, col_profile = st.columns([1.5, 1], gap="large")
    
    with col_intro:
        st.markdown("""
        ### 📖 Tentang Saya
        
        Saya adalah seorang lulusan Teknik Industri dari Universitas Diponegoro dengan passion yang kuat dalam 
        bidang analitik data dan data science. Saat ini, saya bekerja sebagai Data Analyst & Data Scientist di 
        departemen Research and Development, di mana saya menggabungkan keahlian teknis dengan business acumen 
        untuk menghasilkan insight yang berdampak.
        
        Pengalaman saya mencakup analisis data komprehensif, riset pasar yang mendalam, dan pemodelan prediktif 
        yang membantu organisasi membuat keputusan berbasis data. Saya berdedikasi untuk mengubah data kompleks 
        menjadi informasi strategis yang dapat ditindaklanjuti, mendorong inovasi produk dan efisiensi operasional.
        
        Tujuan karir saya adalah menjadi Data Scientist berpengalaman yang tidak hanya menguasai teknik advanced 
        machine learning, tetapi juga memahami konteks bisnis secara menyeluruh. Saya percaya bahwa data science 
        yang efektif harus menggabungkan technical excellence dengan business intelligence untuk menciptakan 
        value jangka panjang bagi organisasi.
        """)
        
        st.markdown("---")
        
        # Skills Section
        st.markdown("### 💼 Keahlian")
        
        skill_col1, skill_col2 = st.columns(2, gap="medium")
        
        with skill_col1:
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #1a3a52 0%, #0f1f38 100%);
                border: 1px solid #2a5a7a;
                border-left: 4px solid #4da3ff;
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 12px;
            '>
                <p style='color: #4da3ff; font-weight: 700; margin: 0 0 10px 0; font-size: 14px;'>📊 Data Analysis</p>
                <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.5;'>Python • SQL • Excel • Power BI • Tableau • Pandas • NumPy</p>
            </div>
            
            <div style='
                background: linear-gradient(135deg, #2a2a52 0%, #1a1a38 100%);
                border: 1px solid #3a2a7a;
                border-left: 4px solid #8a5cff;
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 12px;
            '>
                <p style='color: #8a5cff; font-weight: 700; margin: 0 0 10px 0; font-size: 14px;'>🤖 Data Science</p>
                <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.5;'>Machine Learning • Predictive Modeling • Statistical Analysis • Scikit-learn</p>
            </div>
            """, unsafe_allow_html=True)
        
        with skill_col2:
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #1a4a4f 0%, #0a2a3f 100%);
                border: 1px solid #2a5a6f;
                border-left: 4px solid #00c2a8;
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 12px;
            '>
                <p style='color: #00c2a8; font-weight: 700; margin: 0 0 10px 0; font-size: 14px;'>🎨 UI/UX Design</p>
                <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.5;'>Figma • Canva • Website Prototyping • User Research</p>
            </div>
            
            <div style='
                background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
                border: 1px solid #3d2a7a;
                border-left: 4px solid #ff5fa2;
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 12px;
            '>
                <p style='color: #ff5fa2; font-weight: 700; margin: 0 0 10px 0; font-size: 14px;'>📈 Market Research</p>
                <p style='color: #b0b0b0; font-size: 13px; margin: 0; line-height: 1.5;'>Consumer Insights • Trend Analysis • Competitive Intelligence</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_profile:
        # Profile Card
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #151933 0%, #0f1429 100%);
            border: 1px solid #2a3050;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        '>
            <div style='
                width: 120px;
                height: 120px;
                margin: 0 auto 20px;
                background: linear-gradient(135deg, #4da3ff 0%, #8a5cff 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 50px;
            '>👩‍💼</div>
            
            <h3 style='color: #e0e0e0; margin: 0 0 5px 0; font-size: 18px;'>Melly Marcellia Aziza</h3>
            <p style='color: #4da3ff; margin: 0 0 15px 0; font-weight: 600; font-size: 13px;'>DATA ANALYST & DATA SCIENTIST</p>
            
            <p style='color: #b0b0b0; font-size: 12px; line-height: 1.6; margin: 0 0 20px 0;'>
                Industrial Engineering Graduate | R&D Department | Jakarta, Indonesia
            </p>
            
            <div style='
                background: rgba(77, 163, 255, 0.1);
                border-left: 2px solid #4da3ff;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 15px;
            '>
                <p style='color: #4da3ff; font-weight: 600; font-size: 12px; margin: 0;'>CORE COMPETENCIES</p>
                <p style='color: #b0b0b0; font-size: 11px; margin: 8px 0 0 0;'>
                    Data Analysis • MLOps • Business Intelligence • Product Innovation
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contact Section
    st.markdown("### 📬 Hubungi Saya")
    
    contact_col1, contact_col2 = st.columns(2, gap="large")
    
    with contact_col1:
        st.markdown("""
        <div style='display: flex; flex-direction: column; gap: 12px;'>
            <div style='
                background: linear-gradient(135deg, #1a3a52 0%, #0f1f38 100%);
                border: 1px solid #2a5a7a;
                border-left: 4px solid #4da3ff;
                padding: 12px;
                border-radius: 10px;
            '>
                <p style='color: #4da3ff; font-weight: 700; margin: 0 0 6px 0; font-size: 13px;'>📧 Email</p>
                <a href='mailto:mellymarceliaaziza@gmail.com' style='color: #b0b0b0; text-decoration: none; font-size: 13px;'>
                    <span style='color: #b0b0b0;'>mellymarceliaaziza@gmail.com</span>
                </a>
            </div>
            
            <div style='
                background: linear-gradient(135deg, #2a2a52 0%, #1a1a38 100%);
                border: 1px solid #3a2a7a;
                border-left: 4px solid #8a5cff;
                padding: 12px;
                border-radius: 10px;
            '>
                <p style='color: #8a5cff; font-weight: 700; margin: 0 0 6px 0; font-size: 13px;'>💼 LinkedIn</p>
                <a href='https://www.linkedin.com/in/mellymarceliaaziza/' target='_blank' style='color: #b0b0b0; text-decoration: none; font-size: 13px;'>
                    <span style='color: #b0b0b0;'>linkedin.com/in/mellymarceliaaziza</span>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with contact_col2:
        st.markdown("""
        <div style='display: flex; flex-direction: column; gap: 12px;'>
            <div style='
                background: linear-gradient(135deg, #1a4a4f 0%, #0a2a3f 100%);
                border: 1px solid #2a5a6f;
                border-left: 4px solid #00c2a8;
                padding: 12px;
                border-radius: 10px;
            '>
                <p style='color: #00c2a8; font-weight: 700; margin: 0 0 6px 0; font-size: 13px;'>💻 GitHub</p>
                <a href='https://github.com/mellymaaziza' target='_blank' style='color: #b0b0b0; text-decoration: none; font-size: 13px;'>
                    <span style='color: #b0b0b0;'>github.com/mellymaaziza</span>
                </a>
            </div>
            
            <div style='
                background: linear-gradient(135deg, #3a2a52 0%, #1a1a38 100%);
                border: 1px solid #3d2a7a;
                border-left: 4px solid #ff5fa2;
                padding: 12px;
                border-radius: 10px;
            '>
                <p style='color: #ff5fa2; font-weight: 700; margin: 0 0 6px 0; font-size: 13px;'>🌐 Portfolio</p>
                <a href='#' target='_blank' style='color: #b0b0b0; text-decoration: none; font-size: 13px;'>
                    <span style='color: #b0b0b0;'>Portfolio / Website</span>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; padding: 20px; color: #999; font-size: 11px;'>
        <p>HR Analytics Dashboard | Data-Driven Talent Management | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
""", unsafe_allow_html=True)

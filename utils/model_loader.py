import os
import json
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

MODELS_DIR = "models"
REPORTS_DIR = "reports"
DATASET_PATH = "AI_Placement_Predictor_Dataset_V3.csv"

@st.cache_resource
def load_xgboost_model():
    """Loads the trained XGBoost model from pkl or json."""
    pkl_path = os.path.join(MODELS_DIR, "xgboost_placement_model.pkl")
    json_path = os.path.join(MODELS_DIR, "xgboost_placement_model.json")
    
    if os.path.exists(pkl_path):
        return joblib.load(pkl_path)
    elif os.path.exists(json_path):
        model = xgb.XGBClassifier()
        model.load_model(json_path)
        return model
    else:
        raise FileNotFoundError("Trained XGBoost model file not found in models/ directory.")

@st.cache_resource
def load_preprocessor():
    """Loads the scikit-learn ColumnTransformer preprocessor."""
    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if os.path.exists(prep_path):
        return joblib.load(prep_path)
    else:
        raise FileNotFoundError("Preprocessor pipeline file preprocessor.pkl not found.")

@st.cache_data
def load_feature_names():
    """Loads transformed feature column names list."""
    path = os.path.join(MODELS_DIR, "feature_names.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

@st.cache_data
def load_metrics():
    """Loads model metrics report json."""
    path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_model_config():
    """Loads model configuration json."""
    path = os.path.join(MODELS_DIR, "model_config.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feature_importances():
    """Loads feature importances dataframe."""
    path = os.path.join(MODELS_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_dataset():
    """Loads AI_Placement_Predictor_Dataset_V3.csv."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df['verified_certifications'] = df['verified_certifications'].fillna('None')
        return df
    else:
        raise FileNotFoundError(f"Dataset {DATASET_PATH} not found.")

@st.cache_data
def load_reports():
    """Loads ablation and 6-group report CSVs if present."""
    ablation_path = os.path.join(REPORTS_DIR, "ablation_results.csv")
    six_group_path = os.path.join(REPORTS_DIR, "six_group_analysis.csv")
    
    ablation_df = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else pd.DataFrame()
    six_group_df = pd.read_csv(six_group_path) if os.path.exists(six_group_path) else pd.DataFrame()
    
    return ablation_df, six_group_df

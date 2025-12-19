"""Streamlit dashboard for microplastic detection visualization."""
# #region agent log
import json, os, time
try:
    with open("/home/arvind/Downloads/Projects/.cursor/debug.log", "a") as f:
        log_data = {"sessionId":"debug-session","runId":"dashboard-import","hypothesisId":"A","location":"dashboard/app.py:3","message":"Starting dashboard imports","data":{"python_path":os.environ.get("VIRTUAL_ENV","none")},"timestamp":int(time.time()*1000)}
        f.write(json.dumps(log_data) + "\n")
except: pass
# #endregion

import streamlit as st
import pandas as pd
import numpy as np

# #region agent log
try:
    with open("/home/arvind/Downloads/Projects/.cursor/debug.log", "a") as f:
        log_data = {"sessionId":"debug-session","runId":"dashboard-import","hypothesisId":"B","location":"dashboard/app.py:11","message":"Before plotly import","data":{},"timestamp":int(time.time()*1000)}
        f.write(json.dumps(log_data) + "\n")
except: pass
# #endregion

try:
    import plotly.express as px
    import plotly.graph_objects as go
    # #region agent log
    try:
        with open("/home/arvind/Downloads/Projects/.cursor/debug.log", "a") as f:
            log_data = {"sessionId":"debug-session","runId":"dashboard-import","hypothesisId":"B","location":"dashboard/app.py:18","message":"Plotly import successful","data":{},"timestamp":int(time.time()*1000)}
            f.write(json.dumps(log_data) + "\n")
    except: pass
    # #endregion
except ImportError as e:
    # #region agent log
    try:
        with open("/home/arvind/Downloads/Projects/.cursor/debug.log", "a") as f:
            log_data = {"sessionId":"debug-session","runId":"dashboard-import","hypothesisId":"B","location":"dashboard/app.py:22","message":"Plotly import failed","data":{"error":str(e)},"timestamp":int(time.time()*1000)}
            f.write(json.dumps(log_data) + "\n")
    except: pass
    # #endregion
    st.error("Plotly is not installed. Please run: pip install plotly>=5.15.0")
    st.stop()

from pathlib import Path
import requests
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Microplastic Detection Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Base URL for the prediction API"
)

# Sidebar
st.sidebar.title("🌊 Microplastic Detection")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Predictions", "Monitoring", "Data Explorer", "Model Performance"]
)

# Helper functions
@st.cache_data
def load_sample_data():
    """Load sample data for visualization."""
    # In practice, load from processed data
    return pd.DataFrame({
        "location": ["Site A", "Site B", "Site C", "Site D", "Site E"],
        "latitude": [40.7128, 34.0522, 37.7749, 25.7617, 47.6062],
        "longitude": [-74.0060, -118.2437, -122.4194, -80.1918, -122.3321],
        "concentration": [120, 85, 150, 95, 110],
        "media_type": ["water", "water", "soil", "water", "blood"],
        "date": pd.date_range("2024-01-01", periods=5, freq="D")
    })


def make_api_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make API request."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=data,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None


# Main content
if page == "Overview":
    st.title("Microplastic Detection Dashboard")
    st.markdown("### Overview & Statistics")
    
    # Load sample data
    df = load_sample_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Avg Concentration", f"{df['concentration'].mean():.1f} particles/L")
    with col3:
        st.metric("Detection Rate", "85%")
    with col4:
        st.metric("Active Locations", df['location'].nunique())
    
    st.markdown("---")
    
    # Map visualization
    st.subheader("Sample Locations")
    fig_map = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        size="concentration",
        color="media_type",
        hover_name="location",
        hover_data=["concentration", "date"],
        mapbox_style="open-street-map",
        zoom=3,
        height=500
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Concentration by media type
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Concentration by Media Type")
        fig_bar = px.bar(
            df.groupby("media_type")["concentration"].mean().reset_index(),
            x="media_type",
            y="concentration",
            labels={"concentration": "Avg Concentration (particles/L)"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Time Series")
        fig_line = px.line(
            df,
            x="date",
            y="concentration",
            color="location",
            labels={"concentration": "Concentration (particles/L)"}
        )
        st.plotly_chart(fig_line, use_container_width=True)


elif page == "Predictions":
    st.title("Make Predictions")
    
    prediction_type = st.selectbox(
        "Prediction Type",
        ["Presence/Absence", "Concentration", "Polymer Type", "Image Detection"]
    )
    
    if prediction_type == "Presence/Absence":
        st.subheader("Predict Microplastic Presence")
        
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location", "Site A")
            latitude = st.number_input("Latitude", value=40.7128)
            longitude = st.number_input("Longitude", value=-74.0060)
            sample_depth = st.number_input("Sample Depth (m)", value=1.0)
        
        with col2:
            sample_mass = st.number_input("Sample Mass (g)", value=100.0)
            temperature = st.number_input("Temperature (°C)", value=20.0)
            ph = st.number_input("pH", value=7.0)
        
        if st.button("Predict"):
            features = {
                "latitude": latitude,
                "longitude": longitude,
                "sample_depth": sample_depth,
                "sample_mass": sample_mass,
                "temperature": temperature,
                "ph": ph
            }
            
            result = make_api_request("/api/v1/predict/presence", {"features": features})
            
            if result:
                st.success(f"Prediction: {'Present' if result['prediction'] == 1 else 'Absent'}")
                if result.get("confidence"):
                    st.info(f"Confidence: {result['confidence']:.2%}")
                if result.get("probabilities"):
                    st.json(result["probabilities"])
    
    elif prediction_type == "Concentration":
        st.subheader("Predict Microplastic Concentration")
        
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location", "Site A")
            latitude = st.number_input("Latitude", value=40.7128)
            longitude = st.number_input("Longitude", value=-74.0060)
        
        with col2:
            sample_depth = st.number_input("Sample Depth (m)", value=1.0)
            sample_mass = st.number_input("Sample Mass (g)", value=100.0)
        
        if st.button("Predict"):
            features = {
                "latitude": latitude,
                "longitude": longitude,
                "sample_depth": sample_depth,
                "sample_mass": sample_mass
            }
            
            result = make_api_request("/api/v1/predict/concentration", {"features": features})
            
            if result:
                st.success(f"Predicted Concentration: {result['prediction']:.2f} particles/L")
    
    elif prediction_type == "Polymer Type":
        st.subheader("Predict Polymer Type from Spectrum")
        
        # Upload spectrum file or enter manually
        spectrum_file = st.file_uploader("Upload Spectrum (CSV)", type=["csv"])
        
        if spectrum_file:
            df_spectrum = pd.read_csv(spectrum_file)
            st.dataframe(df_spectrum.head())
            
            if st.button("Predict"):
                # Assume first column is wavelength, second is intensity
                wavelength = df_spectrum.iloc[:, 0].tolist()
                intensity = df_spectrum.iloc[:, 1].tolist()
                
                result = make_api_request("/api/v1/predict/type", {
                    "wavelength": wavelength,
                    "intensity": intensity
                })
                
                if result:
                    st.success(f"Predicted Type: {result['prediction']}")
                    if result.get("probabilities"):
                        st.bar_chart(result["probabilities"])
    
    elif prediction_type == "Image Detection":
        st.subheader("Detect Microplastics in Image")
        
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tiff"])
        
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Detect"):
                files = {"file": image_file.getvalue()}
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/predict/image",
                        files={"file": (image_file.name, image_file.getvalue(), image_file.type)},
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    st.success(f"Detected {result['prediction']['count']} particles")
                    if result['prediction']['count'] > 0:
                        st.json(result['prediction'])
                except Exception as e:
                    st.error(f"Detection failed: {e}")


elif page == "Monitoring":
    st.title("Real-time Monitoring")
    
    # Simulated monitoring data
    st.subheader("Anomaly Detection Status")
    
    media_types = ["water", "soil", "blood"]
    selected_media = st.selectbox("Select Media Type", media_types)
    
    # Simulated detector status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline Mean", "100.5 particles/L")
    with col2:
        st.metric("Baseline Std", "12.3 particles/L")
    with col3:
        st.metric("Sample Count", "150")
    
    st.markdown("---")
    
    # Recent alerts
    st.subheader("Recent Alerts")
    alerts_data = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=5, freq="H"),
        "Type": ["anomaly", "spike", "anomaly", "spike", "anomaly"],
        "Media": ["water", "water", "soil", "blood", "water"],
        "Location": ["Site A", "Site B", "Site C", "Site D", "Site A"],
        "Value": [350, 250, 180, 95, 320]
    })
    st.dataframe(alerts_data, use_container_width=True)
    
    # Concentration time series with anomalies
    st.subheader("Concentration Time Series")
    time_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="H"),
        "concentration": np.random.normal(100, 10, 100)
    })
    # Add some anomalies
    time_data.loc[20, "concentration"] = 350
    time_data.loc[50, "concentration"] = 280
    time_data.loc[75, "concentration"] = 400
    
    fig = px.line(time_data, x="timestamp", y="concentration")
    # Highlight anomalies
    anomalies = time_data[time_data["concentration"] > 250]
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["concentration"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="x"),
        name="Anomalies"
    ))
    st.plotly_chart(fig, use_container_width=True)


elif page == "Data Explorer":
    st.title("Data Explorer")
    
    # Load and display data
    df = load_sample_data()
    
    st.subheader("Sample Data")
    st.dataframe(df, use_container_width=True)
    
    # Filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        selected_media = st.multiselect("Media Type", df["media_type"].unique(), default=df["media_type"].unique())
    with col2:
        selected_locations = st.multiselect("Locations", df["location"].unique(), default=df["location"].unique())
    
    # Filtered data
    filtered_df = df[
        (df["media_type"].isin(selected_media)) &
        (df["location"].isin(selected_locations))
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Statistics
    st.subheader("Statistics")
    st.json(filtered_df.describe().to_dict())


elif page == "Model Performance":
    st.title("Model Performance Metrics")
    
    model_type = st.selectbox("Model Type", ["Spectra CNN", "Image Classifier", "Tabular XGBoost", "YOLOv8 Detector"])
    
    if model_type == "Tabular XGBoost":
        st.subheader("Classification Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "0.92")
        with col2:
            st.metric("Precision", "0.89")
        with col3:
            st.metric("Recall", "0.94")
        with col4:
            st.metric("F1 Score", "0.91")
        
        st.markdown("---")
        
        # Confusion matrix (simulated)
        st.subheader("Confusion Matrix")
        confusion_data = np.array([[450, 30], [25, 495]])
        fig = px.imshow(
            confusion_data,
            labels=dict(x="Predicted", y="Actual"),
            x=["Absent", "Present"],
            y=["Absent", "Present"],
            text_auto=True,
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (simulated)
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            "feature": ["sample_depth", "latitude", "longitude", "temperature", "ph"],
            "importance": [0.35, 0.25, 0.20, 0.12, 0.08]
        })
        fig = px.bar(feature_importance, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    
    elif model_type == "YOLOv8 Detector":
        st.subheader("Detection Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("mAP@0.5", "0.87")
        with col2:
            st.metric("mAP@0.5:0.95", "0.72")
        with col3:
            st.metric("Counting MAE", "2.3 particles")

if __name__ == "__main__":
    pass


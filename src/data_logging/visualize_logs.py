import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os

# 1. Page Config must be the first Streamlit command
st.set_page_config(page_title="SLAM Telemetry", layout="wide")
st.title("Warehouse SLAM & AI Tracking Live Dashboard")

# 2. Define file paths
map_csv = os.path.expanduser('/home/diksha/slam_ws/src/data_logging/warehouse_mapping_log.csv')
yolo_csv = os.path.expanduser('/home/diksha/slam_ws/src/data_logging/yolo_tracking_log.csv')

# 3. Read and Display Data (No 'while True' loop needed!)
try:
    df_map = pd.read_csv(map_csv)
    df_yolo = pd.read_csv(yolo_csv)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3D Spatial Map (Top-Down)")
        fig_map = px.scatter(df_map, x='Centroid_X', y='Centroid_Y', color='Object_ID',
                             hover_data=['Velocity_ms', 'Is_Static'],
                             title="Live Object Positions")
        fig_map.update_yaxes(autorange="reversed") 
        # Keys are no longer needed because the script cleanly re-runs
        st.plotly_chart(fig_map, width='stretch')

    with col2:
        st.subheader("Velocity Telemetry")
        fig_vel = px.line(df_map, x='Timestamp', y='Velocity_ms', color='Object_ID',
                          title="Object Velocity over Time")
        fig_vel.add_hline(y=0.15, line_dash="dash", annotation_text="Threshold")
        st.plotly_chart(fig_vel, width='stretch')
        
    st.subheader("2D YOLO Tracking Data (Raw Bounding Boxes)")
    st.dataframe(df_yolo.tail(10), width='stretch')

except FileNotFoundError:
    st.warning("Waiting for ROS 2 to generate CSV files...")
except Exception as e:
    st.error(f"Error loading data: {e}")

# 4. Force Streamlit to automatically refresh the entire page every 1 second
time.sleep(1)
st.rerun()
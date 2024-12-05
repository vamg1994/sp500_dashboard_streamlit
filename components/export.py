import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def get_download_filename(base_name, extension):
    """Generate a filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def download_plotly_figure(fig, filename):
    """Convert Plotly figure to HTML and offer download"""
    html_str = fig.to_html(include_plotlyjs='cdn')
    st.download_button(
        label="📥 Download Chart",
        data=html_str,
        file_name=filename,
        mime="text/html"
    )

def download_dataframe(df, filename):
    """Convert dataframe to CSV and offer download"""
    csv = df.to_csv(index=True)
    st.download_button(
        label="📥 Download Data (CSV)",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

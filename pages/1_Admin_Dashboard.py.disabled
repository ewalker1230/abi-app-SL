import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the admin page function
from admin_page import admin_page

# Set page config
st.set_page_config(
    page_title="ABI Admin - Query Performance",
    page_icon="📊",
    layout="wide"
)

# Run the admin page
admin_page() 
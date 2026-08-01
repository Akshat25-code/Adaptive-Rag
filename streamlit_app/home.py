"""
Home page — login/register interface.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from streamlit_app.utils.api_client import create_user, login_user

hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Adaptive RAG - Login")
st.title("Adaptive RAG Assistant")

with st.form("auth_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    mode = st.radio("Choose action:", ["Login", "Create Account"])
    submit = st.form_submit_button("Submit")

if submit:
    if not username or not password:
        st.error("Username and password required.")
    else:
        if mode == "Create Account":
            success = create_user(username, password)
            if success:
                st.success("User created. Please log in.")
            else:
                st.error("User creation failed. Username may already exist.")
        else:
            response = login_user(username, password)
            if response and response.get("token"):
                st.session_state["session_id"] = response["token"]
                st.session_state["username"] = username
                st.switch_page("pages/chat.py")
            else:
                st.error("Login failed. Check your credentials.")

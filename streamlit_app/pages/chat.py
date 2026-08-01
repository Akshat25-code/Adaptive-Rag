"""
Chat page for the Streamlit application.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import streamlit as st

from streamlit_app.utils.api_client import document_upload_rag, query_backend

st.set_page_config(
    page_title="LangGraph Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None,
    },
)

if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

col1, col2 = st.columns([10, 2])
with col2:
    st.write("")
    if st.button("🔒 Logout", use_container_width=True):
        st.session_state.show_logout_confirm = True

if st.session_state.show_logout_confirm:
    st.warning("Are you sure you want to logout?")
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("✅ Yes, logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("home.py")
    with col_cancel:
        if st.button("❌ Cancel"):
            st.session_state.show_logout_confirm = False

st.title("💬 LangGraph Chat")

with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    file_description = None
    if uploaded_file:
        file_description = st.text_input(
            "📄 Describe your document (required)",
            max_chars=300,
            placeholder="E.g. LangGraph tutorial with workflows and code examples",
        )

        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}

        file_key = f"{uploaded_file.name}_{file_description}"

        if file_description:
            if file_key not in st.session_state.uploaded_files:
                with st.spinner("Uploading..."):
                    success = document_upload_rag(uploaded_file, file_description)
                if success:
                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.session_state.uploaded_files[file_key] = True
                else:
                    st.error(f"Document Upload Failed: {uploaded_file.name}")
            else:
                st.info(f"Uploaded: {uploaded_file.name}")
        else:
            st.warning("Please describe your document before uploading.")

    st.divider()
    st.header("ℹ️ Route Info")
    if "last_route" in st.session_state:
        route = st.session_state.last_route
        labels = {"index": "📚 Indexed Documents", "general": "🧠 General Knowledge", "search": "🌐 Web Search"}
        label = labels.get(route, route)
        st.info(f"Last response route: **{label}**")
    if "last_time" in st.session_state:
        st.caption(f"Response time: {st.session_state.last_time}s")

if "session_id" not in st.session_state:
    st.warning("Please login first.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role, text, sources = msg if len(msg) == 3 else (msg[0], msg[1], [])
    with st.chat_message(role):
        st.write(text)
        if role == "assistant" and sources:
            with st.expander("📎 Sources", expanded=False):
                for s in sources:
                    st.caption(f"📄 {s.get('source', 'unknown')} (page {s.get('page', 0)})")

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append(("user", user_input, []))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        response = query_backend(user_input, st.session_state["session_id"])
        content = response.get("content", "No response generated.")
        sources = response.get("sources", [])
        route = response.get("route", "unknown")
        response_time = response.get("time_seconds", 0)

        st.session_state.last_route = route
        st.session_state.last_time = response_time

        placeholder.markdown(content)

        if sources:
            with st.expander("📎 Sources", expanded=False):
                for s in sources:
                    st.caption(f"📄 {s.get('source', 'unknown')} (page {s.get('page', 0)})")

    st.session_state.chat_history.append(("assistant", content, sources))

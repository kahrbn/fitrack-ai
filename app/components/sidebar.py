import streamlit as st
from services.memory import save_json

def render_sidebar(st, MEMORY_FILE, LOG_FILE):
    st.sidebar.title("FitTrack")

    if st.session_state.memory:
        st.sidebar.write(st.session_state.memory)

    if st.sidebar.button("Clear Memory"):
        st.session_state.memory = {}
        save_json(MEMORY_FILE, {})
        st.rerun()
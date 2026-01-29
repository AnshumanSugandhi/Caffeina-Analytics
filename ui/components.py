import streamlit as st

def kpi(title, value):
    st.markdown(f"""
    <div class="card">
        <div class="sub">{title}</div>
        <div class="metric">{value}</div>
    </div>
    """, unsafe_allow_html=True)

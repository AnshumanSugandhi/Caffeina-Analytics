import streamlit as st
from ui.styles import load_css
from ui.components import kpi
from core.data_loader import load_demo_data
from core.location_scoring import get_top_locations
from core.price_model import predict_latte_price

st.set_page_config(
    page_title="Caffeina Analytics",
    layout="wide"
)

load_css()

st.markdown('<div class="title">☕ Caffeina Analytics</div>', unsafe_allow_html=True)
st.markdown('<p class="sub">Coffee Location Intelligence Dashboard</p>', unsafe_allow_html=True)

df = load_demo_data()
top = get_top_locations(df)

col1, col2, col3 = st.columns(3)
with col1: kpi("Top Locations", len(top))
with col2: kpi("Avg Latte Price", "$4.80")
with col3: kpi("Confidence Score", "0.87")

st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(top, use_container_width=True)

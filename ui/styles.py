import streamlit as st

def load_css():
    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 2.5rem 3rem;
    }

    :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --accent: #7c3aed;
    }

    html, body {
        background: var(--bg);
        font-family: Inter, sans-serif;
    }

    .card {
        background: var(--card);
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.08);
    }

    .title {
        font-size: 30px;
        font-weight: 800;
        color: var(--text);
    }

    .metric {
        font-size: 34px;
        font-weight: 900;
        color: var(--accent);
    }

    .sub {
        color: var(--muted);
    }
    </style>
    """, unsafe_allow_html=True)

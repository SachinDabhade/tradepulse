import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stock Scanner", page_icon="🔍", layout="wide")

import streamlit as st
import pandas as pd
import os
from Config.config import config
from Config.indexjson import get_index_json, get_index_url

# --- PAGE TITLE ---
st.title("🔍 Quantitative Stock Scanner")

st.markdown("---")


st.header("📊 DOW THEORY LIVE SCANNER 🎯")


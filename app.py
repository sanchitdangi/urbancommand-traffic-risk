import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from utils import load_and_preprocess, calculate_hotspot_risk, intervention_engine
from ml_engine import train_ml_models, predict_risk_ml, run_kmeans_clustering, generate_trend_forecast, ai_chat_assistant
from weather_api import get_real_weather
import os

st.set_page_config(page_title="UrbanCommand | Traffic ML", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# --- 1. PREMIUM CSS ARCHITECTURE ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #030712 !important; color: #e2e8f0; }
.stApp { background-color: #030712 !important; background-image: radial-gradient(circle at 50% -20%, #1e1b4b 0%, transparent 40%); }
header {visibility: hidden;} footer {visibility: hidden;}

/* Style Streamlit's native bordered containers to look like glass cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px 0 rgba(6, 182, 212, 0.1) !important;
    border: 1px solid rgba(6, 182, 212, 0.2) !important;
}

.glass-card { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.4); margin-bottom: 20px; transition: transform 0.2s ease, box-shadow 0.2s ease; }
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 10px 30px 0 rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.2); }
.kpi-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; color: #64748b; margin-bottom: 5px; font-weight: 700; }
.kpi-value { font-size: 2.2rem; font-weight: 800; color: #f8fafc; text-shadow: 0 0 15px rgba(255, 255, 255, 0.1); }
.kpi-accent-cyan { color: #22d3ee; text-shadow: 0 0 15px rgba(34, 211, 238, 0.4); }
.kpi-accent-red { color: #f43f5e; text-shadow: 0 0 15px rgba(244, 63, 94, 0.4); }
.alert-banner { display: flex; align-items: center; padding: 14px 20px; border-radius: 8px; margin-bottom: 25px; font-weight: 600; font-size: 1.05rem; animation: glow-pulse 2s infinite alternate; }
@keyframes glow-pulse { 0% { box-shadow: 0 0 10px rgba(0,0,0,0); } 100% { box-shadow: 0 0 20px currentColor; } }
.alert-high { background: rgba(244, 63, 94, 0.1); border: 1px solid #f43f5e; color: #f43f5e; }
.alert-medium { background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; color: #f59e0b; }
.alert-low { background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; color: #10b981; }
.pulse-dot { width: 10px; height: 10px; border-radius: 50%; background-color: currentColor; margin-right: 15px; box-shadow: 0 0 8px currentColor; animation: blink 1s infinite; }
@keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.2;} 100% {opacity: 1;} }
.stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 15px; }
.stTabs [data-baseweb="tab"] { height: 45px; background-color: rgba(255,255,255,0.03); border-radius: 6px 6px 0 0; color: #64748b; border: none; }
.stTabs [aria-selected="true"] { background-color: rgba(34, 211, 238, 0.1) !important; color: #22d3ee !important; border-bottom: 2px solid #22d3ee !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 2. AUTHENTICATION & DATA LOADING ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = "Viewer"
if "live_feed" not in st.session_state:
    st.session_state.live_feed = []
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

@st.cache_data
def get_data():
    return load_and_preprocess("data.csv")

if not st.session_state.authenticated:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container(border=True):
            st.markdown("<h2 style='text-align:center; color:#22d3ee;'>UrbanCommand Login</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#64748b;'>Enter credentials to access Traffic Intelligence</p>", unsafe_allow_html=True)
            user_role = st.selectbox("Role", ["Admin / Commander", "Traffic Officer", "Viewer"])
            password = st.text_input("Access Key", type="password")
            if st.button("AUTHENTICATE", type="primary", use_container_width=True):
                if password == "viva": # Hardcoded for demo
                    st.session_state.authenticated = True
                    st.session_state.role = user_role
                    st.rerun()
                else:
                    st.error("Invalid Access Key. Hint: viva")
    st.stop()

try:
    df = get_data()
except FileNotFoundError:
    st.error("data.csv not found. Please run generate_data.py first.")
    st.stop()

# --- 3. ML MODEL INITIALIZATION ---
if not os.path.exists("rf_model.pkl"):
    with st.spinner("Initializing Machine Learning Core... Training Random Forest..."):
        train_ml_models(df)

# --- 4. SIDEBAR (Command Center Filters) ---
with st.sidebar:
    st.markdown(f"<h2 style='text-align: center; color: #22d3ee;'>🌐 UrbanCommand</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>LOGGED IN AS: {st.session_state.role.upper()}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🎛️ Operational Boundaries")
    locations = df['Location'].unique().tolist()
    selected_city = st.multiselect("📍 Zone Sector", locations, default=locations)
    selected_sev = st.multiselect("⚠️ Incident Severity", df['Severity'].unique(), default=df['Severity'].unique())
    time_range = st.slider("⏱️ Operational Window (Hour)", 0, 23, (0, 23))
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

filtered_df = df[
    (df['Location'].isin(selected_city)) & 
    (df['Severity'].isin(selected_sev)) &
    (df['Hour'] >= time_range[0]) & 
    (df['Hour'] <= time_range[1])
]

total_acc = len(filtered_df)
fatality_rate = (len(filtered_df[filtered_df['Severity']=='Fatal']) / total_acc * 100) if total_acc > 0 else 0
danger_hour = filtered_df['Hour'].mode()[0] if total_acc > 0 else "N/A"
total_risk_val = filtered_df['Accident_Risk_Score'].sum() if total_acc > 0 else 0

# --- 5. TOP KPI & ALERT STRIP ---
st.markdown("<h1 style='font-size: 2rem; margin-bottom: 0px;'>Real-Time Intelligence Platform</h1>", unsafe_allow_html=True)

if total_risk_val > 1000:
    alert_class, alert_text = "alert-high", "CRITICAL: High Risk Density Detected. Deployment authorized."
elif total_risk_val > 400:
    alert_class, alert_text = "alert-medium", "WARNING: Elevated Risk Activity. Monitor zones."
else:
    alert_class, alert_text = "alert-low", "NORMAL: Parameters within safe operational bounds."

st.markdown(f"""
<div class="alert-banner {alert_class}">
    <div class="pulse-dot"></div>
    {alert_text} | Monitoring {total_acc} active telemetry nodes.
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="glass-card"><div class="kpi-title">🚦 Validated Incidents</div><div class="kpi-value">{total_acc:,}</div></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="glass-card"><div class="kpi-title">☠️ Fatality Rate</div><div class="kpi-value kpi-accent-red">{fatality_rate:.1f}%</div></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="glass-card"><div class="kpi-title">⏱️ Peak Danger Hour</div><div class="kpi-value">{danger_hour}:00</div></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="glass-card"><div class="kpi-title">🔥 Compound Risk Score</div><div class="kpi-value kpi-accent-cyan">{total_risk_val:,.0f}</div></div>', unsafe_allow_html=True)

# --- 6. TABS LAYOUT ---
t_dash, t_geo, t_ml, t_stream, t_chat = st.tabs(["📈 Core Analytics", "🗺️ K-Means GeoIntel", "🧠 ML Forecaster & XAI", "📡 Live Kafka Feed", "💬 AI Assistant"])

color_map = {'Fatal': '#f43f5e', 'Serious': '#f59e0b', 'Minor': '#10b981'}

# --- TAB 1: CORE ANALYTICS ---
with t_dash:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        with st.container(border=True):
            st.markdown("### Causation vs Lethality Matrix")
            cause_sev = filtered_df.groupby(['Cause', 'Severity']).size().reset_index(name='Count')
            fig_cause = px.bar(cause_sev, x='Cause', y='Count', color='Severity', barmode='stack', color_discrete_map=color_map)
            fig_cause.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#cbd5e1", xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_cause, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown("### Severity Distribution")
            sev_df = filtered_df['Severity'].value_counts().reset_index()
            fig_sev = px.pie(sev_df, names='Severity', values='count', hole=0.7, color='Severity', color_discrete_map=color_map)
            fig_sev.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#ffffff")
            st.plotly_chart(fig_sev, use_container_width=True)

# --- TAB 2: K-MEANS GEO-INTEL ---
with t_geo:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Unsupervised ML: Spatial Risk Clustering")
    st.markdown("<p style='color:#94a3b8;'>Using K-Means to automatically group high-risk corridors rather than isolated points.</p>", unsafe_allow_html=True)
    
    if len(filtered_df) > 10:
        clustered_df = run_kmeans_clustering(filtered_df.copy(), num_clusters=5)
        clustered_df['Cluster_Str'] = clustered_df['Cluster'].astype(str)
        
        fig_map = px.scatter_map(
            clustered_df, lat="Latitude", lon="Longitude", color="Cluster_Str",
            size="Severity_Weight", hover_name="Location",
            color_discrete_sequence=['#06b6d4', '#f43f5e', '#8b5cf6', '#10b981', '#f59e0b'],
            zoom=4.5, height=550, opacity=0.8
        )
        fig_map.update_layout(map_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Insufficient data for ML Clustering.")

# --- TAB 3: ML FORECASTER & XAI ---
with t_ml:
    st.markdown("<br>", unsafe_allow_html=True)
    col_f, col_s = st.columns([1, 1])
    
    with col_f:
        with st.container(border=True):
            st.markdown("### 📈 72-Hour Moving Average Forecast")
            forecast_df = generate_trend_forecast(filtered_df)
            if not forecast_df.empty:
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_Incidents'], mode='lines+markers', name='Forecast', line=dict(color='#22d3ee', width=3)))
                fig_fc.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Confidence_Interval_High'], mode='lines', line=dict(width=0), showlegend=False))
                fig_fc.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Confidence_Interval_Low'], mode='lines', fill='tonexty', fillcolor='rgba(34, 211, 238, 0.1)', line=dict(width=0), name='Confidence Interval'))
                fig_fc.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#cbd5e1", xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.warning("Insufficient historical dates for forecasting.")

    with col_s:
        with st.container(border=True):
            st.markdown("### 🔮 Random Forest Prediction & XAI")
            st.markdown("<p style='color:#94a3b8;'>Simulate conditions using the Scikit-Learn RF Classifier.</p>", unsafe_allow_html=True)
            
            sim_hour = st.selectbox("⏱️ Time of Day", range(24), index=21, key='rf_h')
            sim_weather = st.selectbox("☁️ Environmental Condition (Live Weather)", df['Weather'].unique(), key='rf_w')
            sim_road = st.selectbox("🛣️ Infrastructure Type", df['Road_Type'].unique(), key='rf_r')
            sim_loc = st.selectbox("📍 Target Sector", df['Location'].unique(), key='rf_l')
            
            if st.button("RUN ML PREDICTION", type="primary", use_container_width=True):
                prob, explanation = predict_risk_ml(sim_loc, sim_weather, sim_road, sim_hour)
                risk_pct = round(prob * 100, 1)
                
                c_color = "#f43f5e" if risk_pct > 60 else "#f59e0b" if risk_pct > 30 else "#10b981"
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid {c_color}; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <span style="color:#64748b; font-size:0.8rem;">High-Risk Probability Score</span><br>
                    <span style="color:{c_color}; font-size:2rem; font-weight:bold;">{risk_pct}%</span>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
                    <span style="color:#8b5cf6; font-size:0.8rem;">Explainable AI (XAI) Insight:</span><br>
                    <span style="color:#cbd5e1;">{explanation}</span>
                </div>
                """, unsafe_allow_html=True)

# --- TAB 4: LIVE KAFKA FEED — Real-Time Incident Generator ---

# Tier 1/2/3 city distribution used by the live stream generator
STREAM_CITIES = {
    # Tier 1 (higher incident probability due to traffic density)
    "Delhi":     {"tier": 1, "lat": (28.40, 28.80), "lon": (76.80, 77.30), "w": 0.15},
    "Mumbai":    {"tier": 1, "lat": (18.90, 19.20), "lon": (72.80, 73.00), "w": 0.12},
    "Bangalore": {"tier": 1, "lat": (12.80, 13.10), "lon": (77.50, 77.70), "w": 0.10},
    "Chennai":   {"tier": 1, "lat": (12.90, 13.20), "lon": (80.10, 80.30), "w": 0.08},
    "Hyderabad": {"tier": 1, "lat": (17.30, 17.50), "lon": (78.30, 78.50), "w": 0.08},
    "Pune":      {"tier": 1, "lat": (18.40, 18.60), "lon": (73.70, 73.90), "w": 0.07},
    "Kolkata":   {"tier": 1, "lat": (22.50, 22.70), "lon": (88.30, 88.50), "w": 0.07},
    # Tier 2
    "Jaipur":    {"tier": 2, "lat": (26.80, 27.00), "lon": (75.70, 75.90), "w": 0.05},
    "Lucknow":   {"tier": 2, "lat": (26.70, 26.90), "lon": (80.80, 81.00), "w": 0.05},
    "Nagpur":    {"tier": 2, "lat": (21.00, 21.20), "lon": (78.90, 79.20), "w": 0.05},
    "Indore":    {"tier": 2, "lat": (22.60, 22.80), "lon": (75.70, 75.90), "w": 0.04},
    # Tier 3
    "Nashik":    {"tier": 3, "lat": (19.90, 20.10), "lon": (73.70, 73.90), "w": 0.04},
    "Madurai":   {"tier": 3, "lat": (9.80,  10.00), "lon": (78.00, 78.20), "w": 0.03},
    "Varanasi":  {"tier": 3, "lat": (25.20, 25.40), "lon": (82.90, 83.10), "w": 0.04},
    "Meerut":    {"tier": 3, "lat": (28.90, 29.10), "lon": (77.60, 77.80), "w": 0.03},
}
_city_names = list(STREAM_CITIES.keys())
_city_wts   = [STREAM_CITIES[c]["w"] for c in _city_names]
_city_wts   = [w / sum(_city_wts) for w in _city_wts]   # normalise

import random as _random

def _generate_live_incident():
    """
    Generates a single logically consistent live accident event.
    
    Logic:
    - City sampled by tier-weighted probability
    - Weather chosen independently; only then cause is inferred from weather
    - Weather/Low Visibility is a valid cause ONLY when weather ≠ Clear
    - Night hours increase drunk-driving probability
    - Highway + Over-speeding → Fatal or Serious
    - Potholes on City/Rural roads worsen in Rain
    """
    import random as r
    from datetime import datetime

    city  = np.random.choice(_city_names, p=_city_wts)
    meta  = STREAM_CITIES[city]
    tier  = meta["tier"]
    lat   = round(r.uniform(*meta["lat"]), 5)
    lon   = round(r.uniform(*meta["lon"]), 5)

    hour  = int(np.random.choice(
        [r.randint(0,5), r.randint(6,11), r.randint(12,17), r.randint(18,23)],
        p=[0.10, 0.20, 0.30, 0.40]
    ))
    month = datetime.now().month

    # ── Weather: use real-time API first, fall back to rule-based ───────────
    real_weather = get_real_weather(city)   # returns None on failure / highway

    if real_weather:
        weather = real_weather
    else:
        # Rule-based fallback (offline or unsupported city)
        FOG_VALID_MONTHS = [10, 11, 12, 1, 2]
        FOG_VALID_HOURS  = list(range(0, 9)) + [22, 23]
        FOG_TIER1_CITIES = {"Delhi", "Lucknow", "Kanpur", "Jaipur", "Meerut", "Varanasi"}
        weather_opts    = ["Clear", "Rain", "Dust Storm"]
        weather_weights = [0.60, 0.30, 0.10]
        if (month in FOG_VALID_MONTHS and hour in FOG_VALID_HOURS
                and city in FOG_TIER1_CITIES):
            weather_opts    = ["Clear", "Rain", "Fog", "Dust Storm"]
            weather_weights = [0.35, 0.25, 0.35, 0.05]
        weather = r.choices(weather_opts, weights=weather_weights)[0]

    road_type = r.choices(
        ["Highway", "City Road", "Rural Road"],
        weights=[0.30, 0.50, 0.20]
    )[0]
    
    # ── Cause: inferred from weather + context, never randomly stitched ──────
    if weather == "Fog":
        cause = r.choices(
            ["Blind Spot Collision", "Rear-End Collision", "Wrong-Side Driving", "Chain Pile-Up"],
            weights=[0.40, 0.30, 0.20, 0.10]
        )[0]
    elif weather in ["Rain"]:
        cause = r.choices(
            ["Hydroplaning / Skidding", "Brake Failure", "Rear-End Collision", "Waterlogging"],
            weights=[0.35, 0.25, 0.25, 0.15]
        )[0]
    elif weather == "Dust Storm":
        cause = r.choices(
            ["Blind Spot Collision", "Distracted Driving", "Rear-End Collision"],
            weights=[0.50, 0.30, 0.20]
        )[0]
    else:
        # Clear weather — no weather-based causes allowed
        cause = r.choices(
            ["Over-speeding", "Drunk Driving", "Distracted Driving",
             "Signal Jumping", "Lane Indiscipline", "Potholes"],
            weights=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05]
        )[0]
    
    # Late-night drunk driving uplift (22:00–03:00)
    if hour >= 22 or hour <= 3:
        if r.random() < 0.35:
            cause = "Drunk Driving"
    
    # Potholes make no sense on Highways → swap
    if cause == "Potholes" and road_type == "Highway":
        cause = "Over-speeding"
    
    # ------ Severity logic ------
    severity = r.choices(["Fatal", "Serious", "Minor"], weights=[0.15, 0.35, 0.50])[0]
    
    if cause == "Drunk Driving" and road_type == "Highway":
        severity = "Fatal"
    elif cause == "Over-speeding" and road_type == "Highway":
        severity = r.choices(["Fatal", "Serious"], weights=[0.55, 0.45])[0]
    elif cause in ["Blind Spot Collision", "Chain Pile-Up"]:
        if road_type == "Rural Road":
            severity = r.choices(["Fatal","Serious","Minor"], weights=[0.3,0.45,0.25])[0]
        else:
            severity = r.choices(["Serious","Minor"], weights=[0.55, 0.45])[0]
    elif cause == "Potholes" and weather == "Rain":
        severity = r.choices(["Serious","Minor"], weights=[0.70, 0.30])[0]
    
    tier_labels = {1: "Tier 1 Metro", 2: "Tier 2 City", 3: "Tier 3 District"}
    
    return {
        "city": city, "tier": tier_labels[tier],
        "lat": lat, "lon": lon,
        "hour": hour, "weather": weather,
        "road_type": road_type, "cause": cause, "severity": severity,
        "live_weather": real_weather is not None   # True = came from real API
    }


SEVERITY_ICONS = {"Fatal": "💀", "Serious": "🚑", "Minor": "⚠️"}

@st.fragment(run_every="2s")
def render_live_feed():
    if st.session_state.is_streaming:
        ev = _generate_live_incident()
        s_color  = "#f43f5e" if ev["severity"] == "Fatal" else "#f59e0b" if ev["severity"] == "Serious" else "#10b981"
        sev_icon = SEVERITY_ICONS.get(ev["severity"], "⚠️")
        
        new_event = f"""
<div class="glass-card" style="border-left:4px solid {s_color}; padding:14px; margin-bottom:8px;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span style="color:#64748b; font-size:0.75rem;">
      [KAFKA SOCKET | {time.strftime('%H:%M:%S')}] &nbsp;|&nbsp; {ev['tier']}
    </span>
    <span style="background:rgba(255,255,255,0.06); padding:2px 10px; border-radius:20px; font-size:0.75rem; color:{s_color};">
      {sev_icon} {ev['severity']}
    </span>
  </div>
  <div style="margin-top:8px;">
    <strong style="font-size:1rem;">🚨 {ev['city']}</strong>
    &nbsp;·&nbsp;
    <span style="color:#94a3b8;">{ev['road_type']}</span>
  </div>
  <div style="margin-top:4px; color:#cbd5e1; font-size:0.88rem;">
    <strong>Cause:</strong> {ev['cause']}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>Time:</strong> {ev['hour']:02d}:00
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <strong>Coords:</strong> ({ev['lat']}, {ev['lon']})
  </div>
</div>
"""

        st.session_state.live_feed.insert(0, new_event)
        if len(st.session_state.live_feed) > 8:
            st.session_state.live_feed.pop()
    
    if len(st.session_state.live_feed) == 0:
        st.info("System is awaiting connection to Kafka Telemetry socket…")
    else:
        for event in st.session_state.live_feed:
            st.markdown(event, unsafe_allow_html=True)


with t_stream:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📡 Continuous Kafka Telemetry Socket")
    st.markdown("""
<p style='color:#94a3b8;'>
An autonomous <strong>@st.fragment</strong> re-renders this panel every 2 seconds independently of the main app,
simulating a Kafka consumer socket receiving real-time police dispatch telemetry.
Incidents are <em>generated fresh</em> each tick with full causal reasoning — no naive CSV sampling.
</p>""", unsafe_allow_html=True)

    btn_label = "🔴 Connect Stream" if not st.session_state.is_streaming else "⏹️ Disconnect Socket"
    if st.button(btn_label, type="primary"):
        st.session_state.is_streaming = not st.session_state.is_streaming
        st.session_state.live_feed = []   # clear queue on toggle
        st.rerun()

    with st.container(border=True, height=530):
        render_live_feed()

# --- TAB 5: AI CHAT ASSISTANT ---
with t_chat:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 Operational AI Assistant")
    st.markdown("<p style='color:#94a3b8;'>Ask the NLP engine for natural language insights about the current operational dataset.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        user_q = st.text_input("Ask a question (e.g., 'What is the most dangerous location?', 'What is the top cause?', 'How many fatalities at night?')")
        if st.button("Send Query"):
            if user_q:
                answer = ai_chat_assistant(user_q, filtered_df)
                st.markdown(f"""
                <div style="background: rgba(34, 211, 238, 0.1); border-left: 4px solid #22d3ee; padding: 15px; border-radius: 4px; margin-top: 15px;">
                    <strong style="color: #22d3ee;">AI Response:</strong><br>
                    <span style="color: #f8fafc;">{answer}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a query.")

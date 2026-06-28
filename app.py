"""
╔══════════════════════════════════════════════════════════╗
║   🔥 CalorieIQ — Random Forest Fitness Analytics Hub    ║
║   Predicts calories burned using your trained RF model  ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CalorieIQ · Fitness Analytics",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #050d1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,90,31,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(234,88,12,0.08) 0%, transparent 55%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #060e1c 100%);
    border-right: 1px solid rgba(251,146,60,0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #ea580c, #fb923c) !important;
}
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1628; }
::-webkit-scrollbar-thumb { background: rgba(251,146,60,0.4); border-radius: 3px; }

/* ── Metric override ── */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
    color: #fb923c !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Plotly chart bg ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly div { background: transparent !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(251,146,60,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ea580c, #fb923c) !important;
    color: white !important;
}

/* ── Selectbox / number input ── */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(251,146,60,0.2) !important;
    color: #e2e8f0 !important;
}

/* ── Dataframe ── */
.dataframe { background: transparent !important; color: #e2e8f0 !important; }

/* ── Alert box ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
FEATURES = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
FEATURE_LABELS = {
    'Gender':     '⚥  Gender',
    'Age':        '🎂 Age (yrs)',
    'Height':     '📏 Height (cm)',
    'Weight':     '⚖️  Weight (kg)',
    'Duration':   '⏱️  Duration (min)',
    'Heart_Rate': '❤️  Heart Rate (bpm)',
    'Body_Temp':  '🌡️  Body Temp (°C)',
}
FEATURE_ICONS = {
    'Gender':'⚥','Age':'🎂','Height':'📏',
    'Weight':'⚖️','Duration':'⏱️','Heart_Rate':'❤️','Body_Temp':'🌡️'
}

PLOTLY_BASE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font = dict(color="#94a3b8", family="Inter", size=12),

)
MARGIN_DEFAULT = dict(l=10, r=10, t=30, b=10)

def card(content_html):
    return f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(251,146,60,0.18);
                border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:0.5rem;">
        {content_html}
    </div>"""

def kpi(label, value, delta="", color="#fb923c"):
    return f"""
    <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(251,146,60,0.15);
                border-radius:12px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
                    background:linear-gradient(90deg,{color},{color}88);border-radius:12px 12px 0 0;"></div>
        <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;
                    color:#475569;margin-bottom:0.4rem;">{label}</div>
        <div style="font-size:1.9rem;font-weight:700;color:#f1f5f9;
                    font-family:'JetBrains Mono',monospace;line-height:1;">{value}</div>
        {f'<div style="font-size:0.75rem;color:#34d399;margin-top:0.35rem;">{delta}</div>' if delta else ''}
    </div>"""

# ══════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("random_forest_model.pkl")
    except Exception:
        import pickle
        with open("random_forest_model.pkl", "rb") as f:
            return pickle.load(f)

try:
    model = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERR = str(e)

# ══════════════════════════════════════════════════════════
#  SYNTHETIC DATA (for visualisations)
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def make_sample_data(n=800):
    np.random.seed(42)
    df = pd.DataFrame({
        'Gender':     np.random.choice([0, 1], n),
        'Age':        np.random.randint(18, 70, n).astype(float),
        'Height':     np.random.uniform(150, 200, n),
        'Weight':     np.random.uniform(45, 120, n),
        'Duration':   np.random.uniform(5, 90, n),
        'Heart_Rate': np.random.uniform(60, 180, n),
        'Body_Temp':  np.random.uniform(36.0, 42.0, n),
    })
    if MODEL_OK:
        df['Calories'] = model.predict(df[FEATURES])
    else:
        df['Calories'] = (
            df['Duration'] * 5 +
            df['Heart_Rate'] * 0.5 +
            df['Weight'] * 0.3 +
            np.random.normal(0, 10, n)
        ).clip(10, 350)
    df['Gender_Label'] = df['Gender'].map({0: 'Female', 1: 'Male'})
    df['Intensity'] = pd.cut(
        df['Calories'],
        bins=[0, 80, 160, 240, 999],
        labels=['Light 🟢', 'Moderate 🟡', 'High 🟠', 'Extreme 🔴']
    )
    return df

df = make_sample_data()

# ══════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🔥</div>
        <div style="font-size:1.3rem;font-weight:800;color:#fb923c;letter-spacing:-0.5px;">CalorieIQ</div>
        <div style="font-size:0.72rem;color:#475569;letter-spacing:0.5px;">FITNESS ANALYTICS HUB</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#fb923c;margin-bottom:0.8rem;'>🎯 PREDICTION INPUTS</div>", unsafe_allow_html=True)

    gender_label = st.selectbox("⚥ Gender", ["Female", "Male"])
    gender_val = 1 if gender_label == "Male" else 0

    age    = st.slider("🎂 Age (years)",       18,   70,  28)
    height = st.slider("📏 Height (cm)",       150,  200, 172)
    weight = st.slider("⚖️ Weight (kg)",        45,  120,  68)
    duration  = st.slider("⏱️ Duration (min)",   5,   90,  35)
    heart_rate = st.slider("❤️ Heart Rate (bpm)", 60, 180, 130)
    body_temp  = st.slider("🌡️ Body Temp (°C)", 36.0, 42.0, 39.5, step=0.1)

    st.markdown("---")

    if MODEL_OK:
        st.markdown("<div style='font-size:0.78rem;color:#34d399;'>✅ Model loaded</div>", unsafe_allow_html=True)
        st.caption(f"🌲 {model.n_estimators} trees · {model.n_features_in_} features")
    else:
        st.error("⚠️ Model not found")

    st.markdown("---")
    uploaded = st.file_uploader("🔄 Replace model (.pkl)", type=["pkl"])
    if uploaded:
        with open("random_forest_model.pkl", "wb") as f:
            f.write(uploaded.read())
        st.success("✅ Swapped! Refresh page.")


# ══════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════
user_row = pd.DataFrame([{
    'Gender': gender_val, 'Age': age, 'Height': height,
    'Weight': weight, 'Duration': duration,
    'Heart_Rate': heart_rate, 'Body_Temp': body_temp,
}])

if MODEL_OK:
    predicted_cal = float(model.predict(user_row[FEATURES])[0])
    # Individual tree predictions for uncertainty
    tree_preds = np.array([t.predict(user_row[FEATURES])[0] for t in model.estimators_])
    cal_std  = tree_preds.std()
    cal_lo   = max(0, predicted_cal - cal_std)
    cal_hi   = predicted_cal + cal_std
    fi_vals  = model.feature_importances_
else:
    predicted_cal = duration * 5 + heart_rate * 0.5 + weight * 0.3
    cal_std = 10.0
    cal_lo  = predicted_cal - cal_std
    cal_hi  = predicted_cal + cal_std
    fi_vals = np.array([0.0, 0.026, 0.001, 0.008, 0.916, 0.048, 0.001])
    tree_preds = np.random.normal(predicted_cal, cal_std, 200)

# Intensity label
if predicted_cal < 80:
    intensity, int_color, int_emoji = "Light",    "#34d399", "🟢"
elif predicted_cal < 160:
    intensity, int_color, int_emoji = "Moderate", "#fbbf24", "🟡"
elif predicted_cal < 240:
    intensity, int_color, int_emoji = "High",     "#fb923c", "🟠"
else:
    intensity, int_color, int_emoji = "Extreme",  "#f87171", "🔴"

bmi = weight / ((height / 100) ** 2)


# ══════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a0a00 0%,#0f1a0a 40%,#0a0f1a 100%);
            border:1px solid rgba(251,146,60,0.25);border-radius:18px;
            padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-60px;right:-60px;width:300px;height:300px;
                background:radial-gradient(circle,rgba(251,146,60,0.1) 0%,transparent 70%);
                pointer-events:none;"></div>
    <div style="position:absolute;bottom:-40px;left:30%;width:200px;height:200px;
                background:radial-gradient(circle,rgba(234,88,12,0.06) 0%,transparent 70%);
                pointer-events:none;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                color:#ea580c;letter-spacing:1.5px;margin-bottom:0.6rem;font-weight:600;">
        RANDOM FOREST REGRESSOR · 200 TREES · 7 FEATURES
    </div>
    <div style="font-size:2.4rem;font-weight:800;color:#f1f5f9;
                letter-spacing:-1px;margin-bottom:0.4rem;line-height:1.1;">
        🔥 CalorieIQ <span style="color:#fb923c;">Fitness</span> Analytics
    </div>
    <div style="font-size:1rem;color:#64748b;max-width:600px;">
        Real-time calories burned prediction · Feature importance · Model explainability · Interactive analytics
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  LIVE PREDICTION ROW
# ══════════════════════════════════════════════════════════
st.markdown("""<div style="font-size:1rem;font-weight:600;color:#cbd5e1;
    margin-bottom:0.8rem;display:flex;align-items:center;gap:8px;">
    ⚡ Live Prediction
    <span style="flex:1;height:1px;background:linear-gradient(90deg,rgba(251,146,60,0.4),transparent);display:inline-block;margin-left:8px;"></span>
</div>""", unsafe_allow_html=True)

pc1, pc2, pc3, pc4, pc5 = st.columns(5)
with pc1:
    st.markdown(kpi("🔥 Calories Burned", f"{predicted_cal:.1f}", f"± {cal_std:.1f} kcal", "#fb923c"), unsafe_allow_html=True)
with pc2:
    st.markdown(kpi("Intensity Level", f"{int_emoji} {intensity}", "", int_color), unsafe_allow_html=True)
with pc3:
    st.markdown(kpi("Confidence Range", f"{cal_lo:.0f}–{cal_hi:.0f}", "kcal (1σ)", "#38bdf8"), unsafe_allow_html=True)
with pc4:
    st.markdown(kpi("BMI", f"{bmi:.1f}", "kg/m²", "#a78bfa"), unsafe_allow_html=True)
with pc5:
    cal_rate = predicted_cal / duration if duration > 0 else 0
    st.markdown(kpi("Burn Rate", f"{cal_rate:.1f}", "kcal / min", "#34d399"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Prediction Detail",
    "🌲  Model Insights",
    "📊  Data Analytics",
    "🔬  Feature Explorer",
])


# ─────────────────────────────────────────────────────────
#  TAB 1 · PREDICTION DETAIL
# ─────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1.1, 1])

    # Gauge
    with c1:
        st.markdown("**Calorie Burn Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_cal,
            delta={"reference": 160, "increasing": {"color": "#fb923c"}, "decreasing": {"color": "#94a3b8"}},
            number={"suffix": " kcal", "font": {"color": "#f1f5f9", "size": 40, "family": "JetBrains Mono"}},
            title={"text": "Predicted Calories Burned", "font": {"color": "#94a3b8", "size": 14}},
            gauge={
                "axis":  {"range": [0, 350], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
                "bar":   {"color": int_color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(251,146,60,0.2)",
                "steps": [
                    {"range": [0,   80], "color": "rgba(52,211,153,0.08)"},
                    {"range": [80, 160], "color": "rgba(251,191,36,0.08)"},
                    {"range": [160,240], "color": "rgba(251,146,60,0.10)"},
                    {"range": [240,350], "color": "rgba(248,113,113,0.12)"},
                ],
                "threshold": {"line": {"color": "#38bdf8", "width": 2}, "value": predicted_cal},
            },
        ))
        fig_gauge.update_layout(**PLOTLY_BASE, margin=MARGIN_DEFAULT, height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Tree distribution
    with c2:
        st.markdown("**Prediction Distribution (200 Trees)**")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=tree_preds, nbinsx=30,
            marker=dict(
                color=tree_preds,
                colorscale=[[0,"#1a3a5c"],[0.5,"#ea580c"],[1,"#fb923c"]],
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            opacity=0.85,
            hovertemplate="Range: %{x:.1f} kcal<br>Trees: %{y}<extra></extra>"
        ))
        fig_hist.add_vline(x=predicted_cal, line_color="#38bdf8", line_width=2,
                           annotation_text=f" {predicted_cal:.1f}", annotation_font_color="#38bdf8")
        fig_hist.update_layout(
            **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=280,
            xaxis=dict(title="Predicted Calories (kcal)", gridcolor="rgba(251,146,60,0.08)"),
            yaxis=dict(title="Number of Trees",           gridcolor="rgba(251,146,60,0.08)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # User input summary
        st.markdown("**Your Input Summary**")
        inputs = {
            "Gender": gender_label, "Age": f"{age} yrs", "Height": f"{height} cm",
            "Weight": f"{weight} kg", "Duration": f"{duration} min",
            "Heart Rate": f"{heart_rate} bpm", "Body Temp": f"{body_temp} °C",
        }
        cols = st.columns(2)
        for i, (k, v) in enumerate(inputs.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:rgba(251,146,60,0.06);border:1px solid rgba(251,146,60,0.15);
                            border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;">
                    <span style="color:#64748b;font-size:0.75rem;">{k}</span><br>
                    <span style="color:#f1f5f9;font-weight:600;font-family:'JetBrains Mono',monospace;">{v}</span>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  TAB 2 · MODEL INSIGHTS
# ─────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Feature Importances (Gini)**")
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": fi_vals}).sort_values("Importance", ascending=True)
        colors_fi = [
            f"rgba(251,146,60,{0.3 + 0.7 * (v / fi_vals.max())})" for v in fi_df["Importance"]
        ]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(color=colors_fi, line=dict(color="rgba(251,146,60,0.3)", width=1)),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition="outside", textfont=dict(color="#94a3b8", size=11)
        ))
        fig_fi.update_layout(
            **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=340,
            xaxis=dict(title="Gini Importance", gridcolor="rgba(251,146,60,0.1)", range=[0, fi_vals.max()*1.18]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        st.markdown("**Model Configuration**")
        config_items = [
            ("Algorithm",    "Random Forest Regressor"),
            ("Estimators",   str(model.n_estimators if MODEL_OK else 200)),
            ("Max Depth",    str(model.max_depth if MODEL_OK else "None (Auto)")),
            ("Max Features", str(model.max_features if MODEL_OK else 1.0)),
            ("Input Features", str(model.n_features_in_ if MODEL_OK else 7)),
            ("Output",       "Calories Burned (kcal)"),
            ("Sklearn Ver.", "1.6.1"),
        ]
        for label, val in config_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.55rem 1rem;margin-bottom:0.35rem;
                        background:rgba(255,255,255,0.025);border:1px solid rgba(251,146,60,0.12);
                        border-radius:8px;">
                <span style="color:#64748b;font-size:0.85rem;">{label}</span>
                <span style="color:#f1f5f9;font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Top Feature by Importance**")
        top_f = FEATURES[np.argmax(fi_vals)]
        top_v = fi_vals.max()
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(234,88,12,0.15),rgba(251,146,60,0.08));
                    border:1px solid rgba(251,146,60,0.35);border-radius:10px;padding:1rem;text-align:center;">
            <div style="font-size:2rem;">{FEATURE_ICONS.get(top_f,'📌')}</div>
            <div style="font-size:1.1rem;font-weight:700;color:#fb923c;margin-top:0.3rem;">{top_f}</div>
            <div style="font-size:0.85rem;color:#94a3b8;">accounts for {top_v*100:.1f}% of splits</div>
        </div>""", unsafe_allow_html=True)

    # Importance donut
    st.markdown("**Feature Importance Breakdown**")
    fig_pie = go.Figure(go.Pie(
        labels=FEATURES,
        values=fi_vals,
        hole=0.55,
        marker=dict(
            colors=["#f87171","#fb923c","#fbbf24","#34d399","#38bdf8","#a78bfa","#f472b6"],
            line=dict(color="#050d1a", width=3)
        ),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>Importance: %{value:.4f} (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=340,
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8", size=11), orientation="v"),
        annotations=[dict(text=f"🌲<br>{model.n_estimators if MODEL_OK else 200}<br>trees",
                          x=0.5, y=0.5, font=dict(size=13, color="#f1f5f9"), showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  TAB 3 · DATA ANALYTICS
# ─────────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Calories by Duration**")
        fig_sc1 = px.scatter(
            df, x="Duration", y="Calories", color="Heart_Rate",
            size="Weight", hover_data=["Age", "Gender_Label"],
            color_continuous_scale=[[0,"#1e3a5f"],[0.5,"#ea580c"],[1,"#fbbf24"]],
            template="plotly_dark",
            labels={"Duration":"Duration (min)", "Calories":"Calories (kcal)", "Heart_Rate":"Heart Rate"},
        )
        # Mark current user
        fig_sc1.add_trace(go.Scatter(
            x=[duration], y=[predicted_cal], mode="markers",
            marker=dict(color="#38bdf8", size=16, symbol="star",
                        line=dict(color="white", width=2)),
            name="You", showlegend=True
        ))
        fig_sc1.update_layout(**PLOTLY_BASE, margin=MARGIN_DEFAULT, height=320,
            xaxis=dict(gridcolor="rgba(251,146,60,0.08)"),
            yaxis=dict(gridcolor="rgba(251,146,60,0.08)"),
            legend=dict(font=dict(color="#94a3b8")),
            coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_sc1, use_container_width=True)

    with c2:
        st.markdown("**Calorie Distribution by Gender**")
        fig_box = go.Figure()
        for gen, color, fill in [("Female","#f472b6","rgba(244,114,182,0.13)"), ("Male","#38bdf8","rgba(56,189,248,0.13)")]:
            vals = df[df["Gender_Label"] == gen]["Calories"]
            fig_box.add_trace(go.Violin(
                y=vals, name=gen,
                box_visible=True, meanline_visible=True,
                fillcolor=fill, line_color=color,
                hovertemplate=f"<b>{gen}</b><br>Calories: %{{y:.1f}}<extra></extra>"
            ))
        fig_box.update_layout(
            **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=320,
            yaxis=dict(title="Calories (kcal)", gridcolor="rgba(251,146,60,0.08)"),
            legend=dict(font=dict(color="#94a3b8")),
            violingap=0.3
        )
        st.plotly_chart(fig_box, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Workout Intensity Distribution**")
        intensity_counts = df["Intensity"].value_counts()
        int_colors = {"Light 🟢":"#34d399","Moderate 🟡":"#fbbf24","High 🟠":"#fb923c","Extreme 🔴":"#f87171"}
        fig_int = go.Figure(go.Bar(
            x=intensity_counts.index.astype(str),
            y=intensity_counts.values,
            marker=dict(
                color=[int_colors.get(i,"#94a3b8") for i in intensity_counts.index.astype(str)],
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        ))
        fig_int.update_layout(
            **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=290,
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(251,146,60,0.08)", title="Count"),
        )
        st.plotly_chart(fig_int, use_container_width=True)

    with c4:
        st.markdown("**Calories vs Heart Rate (by Age Group)**")
        df["Age_Group"] = pd.cut(df["Age"], bins=[0,30,45,60,100],
                                  labels=["18–30","31–45","46–60","60+"])
        age_colors = {"18–30":"#38bdf8","31–45":"#34d399","46–60":"#fb923c","60+":"#f87171"}
        fig_hr = go.Figure()
        for ag in ["18–30","31–45","46–60","60+"]:
            sub = df[df["Age_Group"]==ag]
            fig_hr.add_trace(go.Scatter(
                x=sub["Heart_Rate"], y=sub["Calories"], mode="markers",
                name=ag, marker=dict(color=age_colors[ag], size=5, opacity=0.6),
                hovertemplate=f"<b>{ag}</b><br>HR: %{{x}}<br>Cal: %{{y:.1f}}<extra></extra>"
            ))
        fig_hr.update_layout(
            **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=290,
            xaxis=dict(title="Heart Rate (bpm)", gridcolor="rgba(251,146,60,0.08)"),
            yaxis=dict(title="Calories (kcal)",  gridcolor="rgba(251,146,60,0.08)"),
            legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    # Correlation heatmap
    st.markdown("**Correlation Matrix**")
    corr_cols = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
    corr = df[corr_cols].corr().round(3)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,"#f87171"],[0.5,"#050d1a"],[1,"#fb923c"]],
        zmid=0,
        text=corr.values.round(2), texttemplate="%{text}",
        textfont={"size":10, "color":"white"},
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#94a3b8"), len=0.8),
    ))
    fig_corr.update_layout(**PLOTLY_BASE, margin=dict(l=10,r=10,t=30,b=10), height=380)
    st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  TAB 4 · FEATURE EXPLORER
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("**How does each feature affect predicted calories?**")
    feat_sel = st.selectbox(
        "Select feature to explore",
        [f for f in FEATURES if f != "Gender"],
        format_func=lambda x: FEATURE_LABELS.get(x, x)
    )

    # Sweep that feature, hold others constant
    sweep_vals = np.linspace(df[feat_sel].min(), df[feat_sel].max(), 80)
    sweep_df = pd.concat([user_row.drop(columns=[feat_sel])] * 80, ignore_index=True)
    sweep_df[feat_sel] = sweep_vals
    sweep_df = sweep_df[FEATURES]  # ensure column order

    if MODEL_OK:
        sweep_preds = model.predict(sweep_df)
    else:
        sweep_preds = sweep_vals * (predicted_cal / (user_row[feat_sel].values[0] + 1e-9))

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown(f"**Predicted Calories vs {feat_sel}** *(other features fixed at your input)*")
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=sweep_vals, y=sweep_preds, mode="lines",
            line=dict(color="#fb923c", width=2.5),
            fill="tozeroy", fillcolor="rgba(251,146,60,0.07)",
            hovertemplate=f"<b>{feat_sel}:</b> %{{x:.2f}}<br><b>Calories:</b> %{{y:.1f}} kcal<extra></extra>",
        ))
        # Mark user's value
        fig_sweep.add_vline(
            x=float(user_row[feat_sel].values[0]),
            line_color="#38bdf8", line_width=2, line_dash="dash",
            annotation_text=f" You: {float(user_row[feat_sel].values[0]):.1f}",
            annotation_font_color="#38bdf8",
        )
        fig_sweep.add_hline(
            y=predicted_cal, line_color="#34d399", line_width=1, line_dash="dot",
        )
        fig_sweep.update_layout(
            **PLOTLY_BASE, margin=MARGIN_DEFAULT, height=360,
            xaxis=dict(title=FEATURE_LABELS.get(feat_sel, feat_sel), gridcolor="rgba(251,146,60,0.08)"),
            yaxis=dict(title="Predicted Calories (kcal)", gridcolor="rgba(251,146,60,0.08)"),
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

    with c2:
        st.markdown("**Feature Statistics**")
        col_data = df[feat_sel]
        stats = {
            "Min":    f"{col_data.min():.2f}",
            "Max":    f"{col_data.max():.2f}",
            "Mean":   f"{col_data.mean():.2f}",
            "Median": f"{col_data.median():.2f}",
            "Std":    f"{col_data.std():.2f}",
            "Your Value": f"{float(user_row[feat_sel].values[0]):.2f}",
        }
        for stat, val in stats.items():
            color = "#38bdf8" if stat == "Your Value" else "#94a3b8"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0.9rem;
                        margin-bottom:0.3rem;background:rgba(255,255,255,0.025);
                        border:1px solid rgba(251,146,60,0.12);border-radius:8px;">
                <span style="color:#64748b;font-size:0.82rem;">{stat}</span>
                <span style="color:{color};font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;">{val}</span>
            </div>""", unsafe_allow_html=True)

        importance_pct = fi_vals[FEATURES.index(feat_sel)] * 100
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(234,88,12,0.12),rgba(251,146,60,0.06));
                    border:1px solid rgba(251,146,60,0.3);border-radius:10px;padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;color:#64748b;letter-spacing:0.5px;text-transform:uppercase;">Model Importance</div>
            <div style="font-size:2.2rem;font-weight:700;color:#fb923c;font-family:'JetBrains Mono',monospace;margin:0.3rem 0;">{importance_pct:.2f}%</div>
            <div style="height:8px;border-radius:4px;background:#0a1628;overflow:hidden;margin-top:0.5rem;">
                <div style="height:100%;width:{importance_pct:.1f}%;background:linear-gradient(90deg,#ea580c,#fb923c);border-radius:4px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Distribution of this feature
        st.markdown("<br>**Distribution**", unsafe_allow_html=True)
        fig_d = go.Figure(go.Histogram(
            x=col_data, nbinsx=25,
            marker=dict(color="rgba(251,146,60,0.55)", line=dict(color="rgba(251,146,60,0.2)", width=0.5))
        ))
        fig_d.add_vline(x=float(user_row[feat_sel].values[0]),
                        line_color="#38bdf8", line_width=2)
        fig_d.update_layout(**PLOTLY_BASE, margin=dict(l=0,r=0,t=10,b=0), height=180,
            xaxis=dict(gridcolor="rgba(251,146,60,0.08)"),
            yaxis=dict(gridcolor="rgba(251,146,60,0.08)"),
        )
        st.plotly_chart(fig_d, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.2rem;border-top:1px solid rgba(251,146,60,0.1);
            color:#334155;font-size:0.78rem;">
    🔥 <strong style="color:#fb923c;">CalorieIQ</strong> · Built with Streamlit · scikit-learn · Plotly &nbsp;|&nbsp;
    Random Forest Regressor · 200 Trees · 7 Features · Calories Burned Prediction
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from forecasting import (load_data, train_test_split_ts,
                          run_arima, run_sarima,
                          run_prophet, run_lstm)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="Time Series Forecasting",
                   layout="wide", page_icon="📈")

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='background:#1B3A5C; padding:30px;
            border-radius:10px; margin-bottom:25px'>
    <h1 style='color:white; margin:0'>
        📈 Time Series Forecasting Dashboard</h1>
    <p style='color:#C9DCEF; margin:8px 0 0 0; font-size:16px'>
    Comparing ARIMA, SARIMA, Prophet and LSTM for
    airline passenger forecasting
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "📊 Data Exploration",
    "🤖 Model Forecasts",
    "📏 Model Comparison",
    "📋 Insights & Recommendations"
])
st.sidebar.markdown("---")
test_months = st.sidebar.slider(
    "Test Period (months)", 12, 36, 24)
show_confidence = st.sidebar.checkbox(
    "Show Confidence Intervals", value=True)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_cached_data():
    return load_data("Data/air_passengers.csv")

@st.cache_data
def get_cached_results(test_months):
    df = load_cached_data()
    train, test = train_test_split_ts(df, test_months)
    arima = run_arima(train, test, test_months)
    sarima = run_sarima(train, test, test_months)
    prophet = run_prophet(train, test, test_months)
    lstm = run_lstm(train, test, test_months)
    return df, train, test, arima, sarima, prophet, lstm

df = load_cached_data()
train, test = train_test_split_ts(df, test_months)

# ── Model colours ────────────────────────────────────────────
COLORS = {
    "Actual":  "#1B3A5C",
    "ARIMA":   "#EF553B",
    "SARIMA":  "#00CC96",
    "Prophet": "#FFA500",
    "LSTM":    "#AB63FA"
}

# ============================================================
# PAGE 1 — DATA EXPLORATION
# ============================================================
if page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("Understanding the airline passenger time series "
                "before modelling.")
    st.markdown("---")

    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df)}")
    c2.metric("Date Range",
              f"{df.index.min().year}–{df.index.max().year}")
    c3.metric("Min Passengers",
              f"{int(df['Passengers'].min()):,}")
    c4.metric("Max Passengers",
              f"{int(df['Passengers'].max()):,}")
    st.markdown("---")

    # Main time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Passengers"],
        mode="lines", name="Monthly Passengers",
        line=dict(color=COLORS["Actual"], width=2),
        fill="tozeroy", fillcolor="rgba(27,58,92,0.1)"
    ))
    fig.update_layout(
        title="Monthly Airline Passengers (1949–1960)",
        xaxis_title="Date", yaxis_title="Passengers (thousands)",
        height=400, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Decomposition
    st.subheader("📉 Seasonal Decomposition")
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(df["Passengers"],
                                 model="multiplicative",
                                 period=12)

    fig2 = make_subplots(rows=4, cols=1,
                          subplot_titles=["Observed", "Trend",
                                          "Seasonality", "Residuals"],
                          vertical_spacing=0.08)

    components = [
        (decomp.observed, COLORS["Actual"]),
        (decomp.trend, "#00CC96"),
        (decomp.seasonal, "#FFA500"),
        (decomp.resid, "#EF553B")
    ]

    for i, (component, color) in enumerate(components, 1):
        fig2.add_trace(
            go.Scatter(x=component.index, y=component.values,
                       mode="lines",
                       line=dict(color=color, width=1.5),
                       showlegend=False),
            row=i, col=1
        )

    fig2.update_layout(height=700,
                        title="Seasonal Decomposition "
                              "(Multiplicative Model)")
    st.plotly_chart(fig2, use_container_width=True)

    # Seasonal pattern
    st.subheader("📅 Seasonal Pattern by Year")
    df_season = df.copy()
    df_season["Year"] = df_season.index.year
    df_season["Month"] = df_season.index.month

    fig3 = go.Figure()
    for year in df_season["Year"].unique():
        year_data = df_season[df_season["Year"] == year]
        fig3.add_trace(go.Scatter(
            x=year_data["Month"],
            y=year_data["Passengers"],
            mode="lines+markers",
            name=str(year),
            marker=dict(size=4)
        ))

    fig3.update_layout(
        title="Monthly Pattern by Year",
        xaxis=dict(tickvals=list(range(1, 13)),
                   ticktext=["Jan", "Feb", "Mar", "Apr",
                              "May", "Jun", "Jul", "Aug",
                              "Sep", "Oct", "Nov", "Dec"]),
        yaxis_title="Passengers (thousands)",
        height=400, hovermode="x unified"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE 2 — MODEL FORECASTS
# ============================================================
elif page == "🤖 Model Forecasts":
    st.title("🤖 Model Forecasts")
    st.markdown("Run and compare all 4 forecasting models.")
    st.markdown("---")

    st.info("⏳ Click the button below to train all models. "
            "This may take 1–2 minutes due to LSTM training.")

    if st.button("🚀 Train All Models & Generate Forecasts",
                 use_container_width=True):

        with st.spinner("Training ARIMA..."):
            arima = run_arima(train, test, test_months)
        st.success("✅ ARIMA complete")

        with st.spinner("Training SARIMA..."):
            sarima = run_sarima(train, test, test_months)
        st.success("✅ SARIMA complete")

        with st.spinner("Training Prophet..."):
            prophet = run_prophet(train, test, test_months)
        st.success("✅ Prophet complete")

        with st.spinner("Training LSTM (this takes longest)..."):
            lstm = run_lstm(train, test, test_months)
        st.success("✅ LSTM complete")

        st.session_state["arima"] = arima
        st.session_state["sarima"] = sarima
        st.session_state["prophet"] = prophet
        st.session_state["lstm"] = lstm
        st.session_state["models_trained"] = True

    if st.session_state.get("models_trained"):
        arima = st.session_state["arima"]
        sarima = st.session_state["sarima"]
        prophet = st.session_state["prophet"]
        lstm = st.session_state["lstm"]

        model_map = {
            "ARIMA": arima,
            "SARIMA": sarima,
            "Prophet": prophet,
            "LSTM": lstm
        }

        # Individual forecast plots
        for model_name, result in model_map.items():
            st.subheader(f"📈 {result['model_name']}")
            fig = go.Figure()

            # Training data
            fig.add_trace(go.Scatter(
                x=train.index, y=train["Passengers"],
                mode="lines", name="Training Data",
                line=dict(color=COLORS["Actual"], width=2)
            ))

            # Actual test data
            fig.add_trace(go.Scatter(
                x=test.index, y=test["Passengers"],
                mode="lines", name="Actual",
                line=dict(color=COLORS["Actual"],
                          width=2, dash="dot")
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=result["forecast"].index,
                y=result["forecast"].values,
                mode="lines", name=f"{model_name} Forecast",
                line=dict(color=COLORS[model_name], width=2)
            ))

            # Confidence interval
            if show_confidence:
                ci = result["conf_int"]
                fig.add_trace(go.Scatter(
                    x=ci.index.tolist() + ci.index.tolist()[::-1],
                    y=ci.iloc[:, 1].tolist() +
                      ci.iloc[:, 0].tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba(128,128,128,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Confidence Interval"
                ))

            fig.update_layout(
                height=400, hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Passengers (thousands)",
                legend=dict(orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1)
            )
            m = result["metrics"]
            st.plotly_chart(fig, use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", m["MAE"])
            col2.metric("RMSE", m["RMSE"])
            col3.metric("MAPE", f"{m['MAPE (%)']}%")
            st.markdown("---")

    else:
        st.warning("Click the button above to train the models first.")

# ============================================================
# PAGE 3 — MODEL COMPARISON
# ============================================================
elif page == "📏 Model Comparison":
    st.title("📏 Model Performance Comparison")
    st.markdown("Side-by-side comparison of all 4 models.")
    st.markdown("---")

    if st.session_state.get("models_trained"):
        arima = st.session_state["arima"]
        sarima = st.session_state["sarima"]
        prophet = st.session_state["prophet"]
        lstm = st.session_state["lstm"]

        models = [arima, sarima, prophet, lstm]
        model_names = ["ARIMA", "SARIMA", "Prophet", "LSTM"]

        # Metrics table
        st.subheader("📊 Metrics Summary")
        metrics_df = pd.DataFrame([m["metrics"] for m in models])
        best_mape = metrics_df["MAPE (%)"].min()
        st.dataframe(metrics_df.style.highlight_min(
            subset=["MAE", "RMSE", "MAPE (%)"],
            color="#D4EDDA"),
            use_container_width=True)

        st.markdown("---")

        # MAPE Bar Chart
        st.subheader("🏆 MAPE Comparison (Lower is Better)")
        fig_bar = px.bar(
            metrics_df, x="Model", y="MAPE (%)",
            color="MAPE (%)",
            color_continuous_scale="RdYlGn_r",
            title="Mean Absolute Percentage Error by Model"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        # All forecasts on one chart
        st.subheader("📈 All Forecasts vs Actual")
        fig_all = go.Figure()

        fig_all.add_trace(go.Scatter(
            x=train.index, y=train["Passengers"],
            mode="lines", name="Training Data",
            line=dict(color=COLORS["Actual"], width=2)
        ))
        fig_all.add_trace(go.Scatter(
            x=test.index, y=test["Passengers"],
            mode="lines+markers", name="Actual",
            line=dict(color=COLORS["Actual"],
                      width=2, dash="dot"),
            marker=dict(size=4)
        ))

        for model_name, result in zip(
                model_names, models):
            fig_all.add_trace(go.Scatter(
                x=result["forecast"].index,
                y=result["forecast"].values,
                mode="lines", name=model_name,
                line=dict(color=COLORS[model_name], width=2)
            ))

        fig_all.update_layout(
            height=500, hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Passengers (thousands)"
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # Best model highlight
        best_idx = metrics_df["MAPE (%)"].idxmin()
        best_model = metrics_df.iloc[best_idx]["Model"]
        best_mape_val = metrics_df.iloc[best_idx]["MAPE (%)"]
        st.success(f"🏆 **Best Model: {best_model}** with "
                   f"MAPE of {best_mape_val}% — "
                   f"lowest forecasting error on test data.")

    else:
        st.warning("Go to **Model Forecasts** page first "
                   "and train the models.")

# ============================================================
# PAGE 4 — INSIGHTS & RECOMMENDATIONS
# ============================================================
elif page == "📋 Insights & Recommendations":
    st.title("📋 Insights & Recommendations")
    st.markdown("What we learned from the models and when to "
                "use each approach.")
    st.markdown("---")

    st.subheader("🔑 Key Findings")
    st.markdown("""
    - **Strong upward trend** with multiplicative seasonality —
      seasonal swings grow proportionally with the trend
    - **Peaks every July–August** and dips every January —
      consistent 12-month seasonal cycle
    - **SARIMA outperforms ARIMA** because it explicitly models
      the seasonal component
    - **Prophet handles trend changes well** and requires the
      least technical setup
    - **LSTM captures non-linear patterns** but needs more data
      to outperform classical methods
    """)

    st.markdown("---")
    st.subheader("🤖 When to Use Each Model")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**ARIMA**\n\n"
                "Best for: Short, stationary series with no "
                "strong seasonality. Simple and interpretable. "
                "Use when you need a quick baseline.")
        st.info("**SARIMA**\n\n"
                "Best for: Series with clear seasonal patterns "
                "like monthly sales, energy usage, or airline "
                "data. Most reliable classical method here.")
    with col2:
        st.info("**Facebook Prophet**\n\n"
                "Best for: Business time series with holidays, "
                "trend changes, or missing data. Very easy to "
                "use and explain to non-technical stakeholders.")
        st.info("**LSTM**\n\n"
                "Best for: Long series (500+ points) with "
                "complex non-linear patterns. Overkill for "
                "small datasets but powerful at scale.")

    st.markdown("---")
    st.subheader("💼 Business Recommendations")
    col3, col4 = st.columns(2)
    with col3:
        st.success("**1. Use SARIMA for monthly planning**\n\n"
                   "For demand forecasting, staffing, and "
                   "inventory — SARIMA provides reliable "
                   "forecasts with clear confidence intervals.")
        st.success("**2. Use Prophet for executive reporting**\n\n"
                   "Prophet's visualisations are intuitive and "
                   "easy to present. Use it when communicating "
                   "forecasts to business stakeholders.")
    with col4:
        st.success("**3. Always visualise confidence intervals**\n\n"
                   "Never present a single forecast line. "
                   "Confidence intervals show the range of "
                   "likely outcomes and aid better decisions.")
        st.success("**4. Retrain models regularly**\n\n"
                   "Time series models drift over time. "
                   "Retrain every quarter with fresh data "
                   "to maintain forecast accuracy.")
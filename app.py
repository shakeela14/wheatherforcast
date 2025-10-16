# ============================================================
# 🌦 WEATHER FORECAST DATA ANALYSIS – STREAMLIT DASHBOARD (Enhanced)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 🌈 Page Configuration & Dark Theme
# ------------------------------------------------------------
st.set_page_config(
    page_title="WeatherSense",
    page_icon="⛅",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #00FF7F, #00C9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #CCCCCC;
        margin-bottom: 20px;
    }
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #1b1b1b;
        color: #00FF7F;
    }

    /* ===== White Selectbox Label & Text ===== */
    div[data-baseweb="select"] > div {
        background-color: #1E1E1E !important;
        color: white !important;
        border: 1px solid #00FF7F !important;
        border-radius: 8px;
    }
    div[data-baseweb="select"] svg {
        fill: white !important;
    }
    div[data-baseweb="select"] input {
        color: white !important;
    }
    div[data-baseweb="select"] span {
        color: white !important;
    }
    label[data-baseweb="select"] {
        color: white !important;
        font-weight: 600;
    }
    .stSelectbox > label, .stSlider > label, .stDateInput > label, .stMultiselect > label {
        color: white !important;
        font-weight: 600;
    }
    /* ===== White label text for file uploader ===== */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: white !important;
    font-weight: 600 !important;
}

/* ===== White file uploader text ===== */
div[data-testid="stFileUploader"] label {
    color: white !important;
    font-weight: 600 !important;
}

/* Optional: make the upload area border neon green */
div[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #00FF7F !important;
    border-radius: 8px !important;
    background-color: #1E1E1E !important;
}
.sub-title {
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        background: linear-gradient(90deg, #00FF7F, #00C9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px;
        text-shadow: 0px 0px 10px rgba(0, 255, 127, 0.3);
        animation: glow 3s ease-in-out infinite alternate;
        margin-bottom: 30px;
    }

    @keyframes glow {
        from { text-shadow: 0 0 5px #00FF7F, 0 0 10px #00FF7F; }
        to { text-shadow: 0 0 20px #00FF7F, 0 0 30px #00C9FF; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌦 WeatherSense</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'> ⚡ Interactive Analysis, EDA & Temperature Prediction</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧾 Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("📋 Controls & Filters")
uploaded_file = st.sidebar.file_uploader("📁 Upload your weather CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Required columns
    date_col = 'Formatted Date'
    temp_col = 'Temperature (C)'
    precip_col = 'Precip Type'

    if date_col not in df.columns or temp_col not in df.columns:
        st.sidebar.error("❌ Required columns not found in the CSV.")
        st.stop()

    # -------------------------------
    # Data Cleaning (Naive Datetime)
    # -------------------------------
    df[date_col] = df[date_col].astype(str).str.strip()
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df['Date'] = df['Date'].apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) else x)
    df = df[df['Date'].notna()].reset_index(drop=True)
    df = df.sort_values('Date').reset_index(drop=True)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(method='ffill')

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.subheader("📅 Date Range (up to 2016)")
    min_date = df['Date'].min().date()
    max_date = min(df['Date'].max().date(), pd.to_datetime("2016-12-31").date())  # restrict to 2016
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    st.sidebar.subheader("🌡 Temperature Range (°C)")
    min_temp = float(df[temp_col].min())
    max_temp = float(df[temp_col].max())
    temp_range = st.sidebar.slider("Select Range", min_temp, max_temp, (min_temp, max_temp))

    st.sidebar.subheader("☔ Precipitation Type")
    precip_options = df[precip_col].dropna().unique().tolist() if precip_col in df.columns else []
    selected_precip = st.sidebar.multiselect("Select Type", precip_options, default=precip_options)

    st.sidebar.subheader("🤖 Forecast Settings")
    forecast_target = st.sidebar.selectbox(
        "Forecast Target",
        [temp_col, 'Apparent Temperature (C)'] if 'Apparent Temperature (C)' in df.columns else [temp_col]
    )
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 7, 3)

    # -------------------------------
    # Apply Filters
    # -------------------------------
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    filtered_df = df[
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date) &
        (df[temp_col] >= temp_range[0]) &
        (df[temp_col] <= temp_range[1])
    ]
    if precip_col in df.columns and selected_precip:
        filtered_df = filtered_df[filtered_df[precip_col].isin(selected_precip)]

    st.markdown("### 📋 Filtered Dataset")
    st.write(f"**Rows after filter:** {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head())

    if filtered_df.empty:
        st.warning("⚠️ No data available after applying filters. Adjust the filters to see the analysis.")
    else:
        # ===============================
        # EDA Section
        # ===============================
        st.markdown("### 📊 Exploratory Data Analysis")
        numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            plot_type = st.selectbox("Select Plot Type", ["Line", "Histogram", "Boxplot", "Scatter"], key='eda_plot')
            x_var = st.selectbox("Select X-axis Variable", numeric_cols, key='eda_x')
            y_var = st.selectbox("Select Y-axis Variable (Scatter Only)", numeric_cols, key='eda_y')

            if plot_type == "Line":
                fig = px.line(filtered_df, x='Date', y=x_var, template='plotly_dark', color_discrete_sequence=['#00FF7F'])
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Histogram":
                fig, ax = plt.subplots(figsize=(8,4))
                sns.histplot(filtered_df[x_var], bins=30, kde=True, color='#00FF7F', ax=ax)
                ax.set_facecolor('#121212')
                st.pyplot(fig)
            elif plot_type == "Boxplot":
                fig, ax = plt.subplots(figsize=(8,4))
                sns.boxplot(x=filtered_df[x_var], color='#00FF7F', ax=ax)
                ax.set_facecolor('#121212')
                st.pyplot(fig)
            elif plot_type == "Scatter":
                fig = px.scatter(filtered_df, x=x_var, y=y_var, color=y_var, template='plotly_dark', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # Forecasting Section
        # ===============================
        st.markdown("### 🤖 Temperature Forecasting")

        filtered_df['Prev_Value'] = filtered_df[forecast_target].shift(1)
        model_df = filtered_df.dropna().reset_index(drop=True)

        if model_df.shape[0] < 2:
            st.warning("⚠️ Not enough data to train the forecasting model. Please adjust filters or upload more data.")
        else:
            X = model_df[['Prev_Value']]
            y = model_df[forecast_target]

            model = LinearRegression()
            model.fit(X, y)

            last_value = model_df.iloc[-1]['Prev_Value']
            future_preds = []
            current_value = last_value
            for i in range(forecast_days):
                next_val = model.predict([[current_value]])[0]
                future_preds.append(next_val)
                current_value = next_val

            future_dates = pd.date_range(start=model_df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_preds})

            st.write(f"**Forecast for next {forecast_days} day(s):**")
            st.dataframe(forecast_df)

            combined_df = pd.concat([
                model_df[['Date', forecast_target]].rename(columns={forecast_target:'Value'}),
                forecast_df.rename(columns={'Predicted':'Value'})
            ])
            fig = px.line(combined_df, x='Date', y='Value', template='plotly_dark',
                          color_discrete_sequence=['#00FF7F'], title=f'{forecast_target} Historical + Forecast')
            st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # Insights Section
        # ===============================
        st.markdown("### 💡 Key Insights")
        st.write(f"- 🌡 **Average {forecast_target}:** {round(filtered_df[forecast_target].mean(),2)}")
        st.write(f"- 🔥 **Highest {forecast_target}:** {filtered_df[forecast_target].max()}")
        st.write(f"- ❄️ **Lowest {forecast_target}:** {filtered_df[forecast_target].min()}")

        corr_numeric = filtered_df.select_dtypes(include=np.number).columns.tolist()
        selected_corr = st.multiselect("Select Variables for Correlation Matrix", corr_numeric, default=corr_numeric[:3])
        if len(selected_corr) >= 2:
            corr_df = filtered_df[selected_corr].corr()
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(corr_df, annot=True, cmap='Greens', fmt='.2f', ax=ax)
            ax.set_facecolor('#121212')
            st.pyplot(fig)

        st.success("✅ Weather Forecast Analysis Completed Successfully!")

else:
    st.sidebar.info("👆 Please upload a weather dataset CSV to begin the analysis.")
    st.info("Upload your CSV to see the dashboard.")

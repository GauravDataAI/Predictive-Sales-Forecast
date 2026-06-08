import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import warnings
import os

# Suppress Prophet's minor logging warnings for a cleaner UI
warnings.filterwarnings("ignore", category=FutureWarning)

# Page Config
st.set_page_config(
    page_title="Predictive Sales Forecast",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1 {
    color: #1f77b4;
    text-align: center;
}
.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1>🔮 Predictive Sales Forecast Dashboard</h1>",
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    import os

    st.write("Current Directory:", os.getcwd())
    st.write("Files Available:", os.listdir("."))

    try:
        df = pd.read_csv("Predictive-Sales-Forecast/superstore.csv")

        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month_name()

        return df

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

df = load_data()

# ---------------- Sidebar ---------------- #
st.sidebar.header("🔍 Filters")

selected_region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

selected_category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(selected_region)) &
    (df["Category"].isin(selected_category))
]

# Prevent errors if filters result in no data
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
    st.stop()

# ---------------- KPI Section ---------------- #
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = len(filtered_df)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Revenue", f"₹{total_sales:,.0f}")

with col2:
    st.metric("📈 Profit", f"₹{total_profit:,.0f}")

with col3:
    st.metric("🛒 Orders", f"{total_orders:,}")

st.markdown("---")

# ---------------- Predictive Forecast Section (NEW) ---------------- #
st.subheader("🔮 AI Sales Forecast (Next 6 Months)")

# Prepare data for Prophet (requires 'ds' for date and 'y' for value)
forecast_data = (
    filtered_df.groupby("Year-Month")["Sales"]
    .sum()
    .reset_index()
    .rename(columns={"Year-Month": "ds", "Sales": "y"})
)
forecast_data["ds"] = pd.to_datetime(forecast_data["ds"])
forecast_data = forecast_data.sort_values("ds")

if len(forecast_data) < 5:
    st.warning("⚠️ Not enough historical data points to generate a reliable forecast. Please broaden your filters (e.g., select more regions/categories).")
else:
    with st.spinner("🧠 Training predictive model..."):
        # Initialize and train Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(forecast_data)

        # Create future dataframe for the next 6 months ('MS' = Month Start)
        future = model.make_future_dataframe(periods=6, freq='MS')
        forecast = model.predict(future)

    # Plotting Historical vs Forecasted Data
    fig_forecast = go.Figure()

    # 1. Historical Sales
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['y'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=3)
    ))

    # 2. Forecasted Sales (Dashed line for future)
    # We only want to show the forecast part as dashed, so we filter future dates
    future_forecast = forecast[forecast['ds'] > forecast_data['ds'].max()]
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        mode='lines',
        name='Forecasted Sales',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))

    # 3. Confidence Interval Ribbon
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
        y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))

    fig_forecast.update_layout(
        template="plotly_white",
        height=400,
        title="📈 Historical vs. Predicted Sales Trend",
        xaxis_title="Date",
        yaxis_title="Sales (₹)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Display next month's specific prediction as a quick metric
    next_month_pred = future_forecast.iloc[0]
    st.info(f"💡 **Prediction for next month:** Expected sales are approximately **₹{next_month_pred['yhat']:,.0f}** (Range: ₹{next_month_pred['yhat_lower']:,.0f} - ₹{next_month_pred['yhat_upper']:,.0f})")

st.markdown("---")

# ---------------- Historical Monthly Sales Trend ---------------- #
st.subheader("📊 Historical Monthly Sales Trend")

monthly_sales = (
    filtered_df.groupby("Year-Month")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Year-Month")
)

fig1 = px.line(
    monthly_sales,
    x="Year-Month",
    y="Sales",
    markers=True,
    title="Past Performance"
)

fig1.update_layout(
    template="plotly_white",
    height=350,
    xaxis_title="Year-Month",
    yaxis_title="Sales"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- Region & Category ---------------- #
col4, col5 = st.columns(2)

with col4:
    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig2 = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        title="🌎 Region Wise Sales",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col5:
    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig3 = px.pie(
        category_sales,
        names="Category",
        values="Sales",
        hole=0.4,
        title="📦 Category Contribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Top Products ---------------- #
st.subheader("🏆 Top 10 Products")

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_products["Sales"] = top_products["Sales"].apply(lambda x: f"₹{x:,.0f}")

st.dataframe(
    top_products,
    use_container_width=True,
    hide_index=True
)

# ---------------- Download Button ---------------- #
csv = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv,
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

# ---------------- Business Insights ---------------- #
st.subheader("💡 Business Insights")

col6, col7 = st.columns(2)
with col6:
    best_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    st.success(f"🔥 Best Performing Region: **{best_region}**")

with col7:
    best_product = filtered_df.groupby("Product Name")["Sales"].sum().idxmax()
    st.success(f"🏆 Top Selling Product: **{best_product}**")

# Footer
st.markdown("---")
st.caption("Developed by Gaurav | Predictive Data Analytics Project 🚀")

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from datetime import date, timedelta

st.set_page_config(page_title="Portfolio CVaR Project", layout="wide")

st.title("Portfolio Risk Dashboard")
st.write("Import raw stock data and compare selected stocks.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")

ticker_input = st.sidebar.text_input(
    "Enter stock tickers separated by commas",
    value="AAPL, MSFT, NVDA, AMZN"
)

tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]

# -----------------------------
# Time horizon buttons
# -----------------------------
horizon_map = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
    "20 Years": 365 * 20
}

selected_horizon = st.sidebar.pills(
    "Time horizon",
    options=list(horizon_map.keys()),
    default="1 Year"
)

end_date = date.today()
start_date = end_date - timedelta(days=horizon_map[selected_horizon])

# -----------------------------
# Load stock data
# -----------------------------
@st.cache_data
def load_stock_data(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # If multiple tickers, yfinance gives multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    return prices.dropna(how="all").ffill().dropna()

if len(tickers) == 0:
    st.warning("Please enter at least one ticker.")
    st.stop()

prices = load_stock_data(tickers, start_date, end_date)

if prices.empty:
    st.error("No data found. Check your tickers or date range.")
    st.stop()

# -----------------------------
# Show raw data
# -----------------------------
st.subheader("Raw Imported Price Data")
st.dataframe(prices.tail())

# -----------------------------
# Normalize prices
# -----------------------------
normalized_prices = prices / prices.iloc[0]

# -----------------------------
# Plot graph (fixed date axis)
# -----------------------------
st.subheader("Normalized Stock Price Comparison")

fig, ax = plt.subplots(figsize=(8, 4))

for ticker in normalized_prices.columns:
    ax.plot(normalized_prices.index, normalized_prices[ticker], label=ticker)

ax.set_title("Normalized Stock Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price")
ax.legend(fontsize=8)
ax.grid(True)

# --- Fix date formatting ---
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xticks(rotation=30)

st.pyplot(fig, use_container_width=False)

# -----------------------------
# Basic stats
# -----------------------------
st.subheader("Basic Return Summary")

returns = prices.pct_change().dropna()

summary = pd.DataFrame({
    "Total Return": (prices.iloc[-1] / prices.iloc[0] - 1),
    "Average Daily Return": returns.mean(),
    "Daily Volatility": returns.std()
})

st.dataframe(summary)

# -----------------------------
# Mean-CVaR Portfolio Optimization
# -----------------------------
st.subheader("Scenario-Based Mean-CVaR Portfolio Optimization")

# --- Controls (with tooltip help) ---
confidence_level = st.slider(
    "CVaR confidence level",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="Defines tail risk. 95% means CVaR measures the average loss in the worst 5% of scenarios."
)

risk_aversion = st.slider(
    "Risk aversion",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="λ controls trade-off between return and risk. Higher = safer portfolio (lower CVaR), lower return."
)

num_portfolios = st.slider(
    "Number of random portfolios",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
    help="Number of portfolios simulated. Higher = more accurate optimization but slower."
)

@st.cache_data
def optimize_mean_cvar(returns, confidence_level, risk_aversion, num_portfolios):
    import numpy as np

    num_assets = returns.shape[1]
    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / weights.sum()

        portfolio_returns = returns.dot(weights)
        losses = -portfolio_returns

        var = np.quantile(losses, confidence_level)
        cvar = losses[losses >= var].mean()

        expected_return = portfolio_returns.mean()

        score = expected_return - risk_aversion * cvar

        results.append({
            "Expected Daily Return": expected_return,
            "VaR": var,
            "CVaR": cvar,
            "Score": score,
            **{returns.columns[i]: weights[i] for i in range(num_assets)}
        })

    results_df = pd.DataFrame(results)
    best_portfolio = results_df.loc[results_df["Score"].idxmax()]

    return best_portfolio, results_df


best_portfolio, optimization_results = optimize_mean_cvar(
    returns,
    confidence_level,
    risk_aversion,
    num_portfolios
)

st.write("### Best Portfolio Weights")

weight_data = best_portfolio[returns.columns].to_frame(name="Weight")
weight_data["Weight"] = weight_data["Weight"].astype(float)

st.dataframe(weight_data.style.format({"Weight": "{:.2%}"}))

st.write("### Best Portfolio Risk Metrics")

metrics = pd.DataFrame({
    "Metric": [
        "Expected Daily Return",
        "Expected Annual Return",
        "VaR",
        "CVaR",
        "Score"
    ],
    "Value": [
        best_portfolio["Expected Daily Return"],
        best_portfolio["Expected Daily Return"] * 252,
        best_portfolio["VaR"],
        best_portfolio["CVaR"],
        best_portfolio["Score"]
    ]
})

st.dataframe(metrics)

# -----------------------------
# Compute optimized portfolio series
# -----------------------------
optimized_weights = weight_data["Weight"].values
optimized_returns = returns.dot(optimized_weights)
optimized_growth = (1 + optimized_returns).cumprod()

# -----------------------------
# Optimized Portfolio Section (1x2 Grid)
# -----------------------------
st.write("### Optimized Portfolio Analysis")

col1, col2 = st.columns(2)

# --- Left: Portfolio Growth ---
with col1:
    st.write("#### Portfolio Growth")

    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(optimized_growth.index, optimized_growth)
    ax1.set_title("Growth of $1")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.grid(True)

    st.pyplot(fig1, use_container_width=True)

# --- Right: Loss Distribution ---
with col2:
    st.write("#### Loss Distribution")

    optimized_losses = -optimized_returns

    mean_loss = optimized_losses.mean()
    std_loss = optimized_losses.std()

    var_value = best_portfolio["VaR"]
    cvar_value = best_portfolio["CVaR"]

    import numpy as np  # safe if not already imported

    x = np.linspace(
        optimized_losses.min(),
        optimized_losses.max(),
        500
    )

    normal_pdf = (
        1 / (std_loss * np.sqrt(2 * np.pi))
    ) * np.exp(-0.5 * ((x - mean_loss) / std_loss) ** 2)

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))

    ax2.hist(
        optimized_losses,
        bins=40,
        density=True,
        alpha=0.6,
        label="Historical losses"
    )

    ax2.plot(x, normal_pdf, label="Normal fit")

    ax2.axvline(
        var_value,
        linestyle="--",
        linewidth=2,
        label=f"VaR {confidence_level:.0%}"
    )

    ax2.axvline(
        cvar_value,
        linestyle=":",
        linewidth=2,
        label=f"CVaR {confidence_level:.0%}"
    )

    ax2.set_title("Loss Distribution")
    ax2.set_xlabel("Daily Loss")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    st.pyplot(fig2, use_container_width=True)

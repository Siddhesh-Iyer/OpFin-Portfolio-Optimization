import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# ==========================================
# 1. HELPER FUNCTIONS & DATA FETCHING
# ==========================================

# Asset Universe Definitions
ASSET_CLASSES = {
    "USA (USD)": {
        "Indices": ["SPY", "QQQ", "IWM", "VTI"],
        "Sector ETFs": ["XLK", "XLV", "XLF", "XLE"],
        "Bonds": ["TLT", "IEF", "SHY", "AGG", "LQD"],
        "Commodities": ["GLD", "SLV", "USO", "DBC"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "Stocks (Top 10)": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "JNJ", "V"]
    },
    "India (INR)": {
        "Indices": ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "MID150BEES.NS"],
        "Bonds": ["LIQUIDBEES.NS", "GILT5YBEES.NS", "SETF10GILT.NS"],
        "Commodities": ["GOLDBEES.NS", "SILVERBEES.NS"],
        "Stocks (Blue Chip)": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HUL.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS"]
    }
}

def get_market_config(market):
    """Returns configuration based on selected market."""
    if market == "USA (USD)":
        return {"suffix": "", "currency": "$", "rf_ticker": "^TNX", "benchmark": "SPY"}
    elif market == "India (INR)":
        return {"suffix": ".NS", "currency": "₹", "rf_ticker": "^IGDB10Y", "benchmark": "^NSEI"}
    return {"suffix": "", "currency": "$", "rf_ticker": "^TNX", "benchmark": "SPY"}

def format_ticker(ticker, suffix):
    """Appends suffix if missing (e.g., RELIANCE -> RELIANCE.NS)."""
    ticker = ticker.strip().upper()
    if suffix and not ticker.endswith(suffix):
        return f"{ticker}{suffix}"
    return ticker

def calculate_cagr(price_series):
    """Calculates Compound Annual Growth Rate."""
    if price_series.empty or len(price_series) < 2:
        return 0.0
    
    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]
    
    if start_price <= 0: return 0.0 # Avoid division by zero or negative base for power
    if end_price <= 0: return 0.0

    # Calculate years
    days = (price_series.index[-1] - price_series.index[0]).days
    years = days / 365.25
    
    if years == 0: return 0.0
    
    cagr = (end_price / start_price) ** (1 / years) - 1
    return cagr

@st.cache_data(ttl=3600)
def fetch_financial_data(tickers, market_config, period="5y"):
    """
    Fetches price history AND fundamental info for all tickers.
    """
    data_close = pd.DataFrame()
    fundamentals = []
    sector_map = {}
    valid_tickers = []

    progress_bar = st.progress(0)
    
    # 1. Fetch History FIRST
    try:
        # We download everything at once. This is the most reliable source for PRICE.
        bulk_history = yf.download(tickers, period=period)['Close']
    except Exception:
        bulk_history = pd.DataFrame()

    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # --- ROBUST PRICE FETCHING START ---
            current_price = info.get('currentPrice')
            
            # If info doesn't have price (common for Crypto/Indices), get it from history
            if (current_price is None or np.isnan(current_price)) and not bulk_history.empty:
                if ticker in bulk_history.columns:
                    # Get the last valid price (iloc[-1])
                    price_series = bulk_history[ticker].dropna()
                    if not price_series.empty:
                        current_price = price_series.iloc[-1]
            # --- ROBUST PRICE FETCHING END ---

            # Check if we have price data for this ticker (to calculate CAGR & Validate)
            cagr = 0.0
            has_history = False
            
            if not bulk_history.empty and ticker in bulk_history.columns:
                price_series = bulk_history[ticker].dropna()
                if not price_series.empty:
                    cagr = calculate_cagr(price_series)
                    valid_tickers.append(ticker)
                    has_history = True
            
            # Fallback for single tickers if bulk failed or column missing
            if not has_history:
                hist = t.history(period=period)
                if not hist.empty:
                    cagr = calculate_cagr(hist['Close'])
                    valid_tickers.append(ticker)
                    # If we still don't have a current price, take it from here
                    if current_price is None:
                        current_price = hist['Close'].iloc[-1]

            sector = info.get('sector', 'Unknown')
            # Fix for Crypto/Indices often missing 'sector'
            if 'quoteType' in info:
                if info['quoteType'] == 'CRYPTOCURRENCY': sector = 'Crypto'
                elif info['quoteType'] == 'ETF': sector = 'ETF'
            
            sector_map[ticker] = sector
            
            fundamentals.append({
                "Ticker": ticker,
                "Name": info.get('shortName', ticker),
                "Sector": sector,
                "Price": current_price, # Now using the robust price
                "CAGR 5Y": cagr * 100,
                "P/E": info.get('trailingPE', np.nan),
                "P/B": info.get('priceToBook', np.nan),
                "Debt/Eq": info.get('debtToEquity', np.nan),
                "Beta": info.get('beta', np.nan),
                "Div Yield %": (info.get('dividendYield', 0) or 0) * 100
            })
        except Exception:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    
    if not valid_tickers:
        return None, None, None, None

    # Re-fetch only clean history for optimization
    if not bulk_history.empty:
        cols = [c for c in bulk_history.columns if c in valid_tickers]
        data_download = bulk_history[cols]
    else:
        data_download = yf.download(valid_tickers, period=period)['Close']
    
    if isinstance(data_download, pd.Series):
        data_download = data_download.to_frame(name=valid_tickers[0])
    
    data_download.dropna(axis=1, how='all', inplace=True)
    
    fundamentals_df = pd.DataFrame(fundamentals)
    # Ensure we only keep fundamentals for valid tickers
    fundamentals_df = fundamentals_df[fundamentals_df['Ticker'].isin(valid_tickers)]

    return data_download, fundamentals_df, sector_map, valid_tickers

def process_market_data(price_data):
    """Calculates Expected Returns and Covariance Matrix."""
    log_returns = np.log(price_data / price_data.shift(1))
    mu = log_returns.mean() * 252
    sigma = log_returns.cov() * 252
    return mu, sigma, log_returns

# ==========================================
# 2. OPTIMIZATION ENGINES (MODELS)
# ==========================================

def solve_mean_variance_gurobi(mu, sigma, target_return, k_max, sector_map, max_sector_pct, current_weights=None, max_turnover=None):
    """
    Model 1: Mean-Variance (Minimize Risk) using Gurobi MIQP.
    Includes Cardinality, Sector, and Turnover constraints.
    """
    n = len(mu)
    tickers = mu.index.tolist()
    
    try:
        m = gp.Model("mv_opt")
        m.setParam('LogToConsole', 0)
        
        # Variables
        w = m.addMVar(shape=n, lb=0.0, ub=1.0, name="w")
        y = m.addMVar(shape=n, vtype=GRB.BINARY, name="y")
        
        # Objective: Minimize Variance
        m.setObjective(w @ sigma.values @ w, GRB.MINIMIZE)
        
        # Constraints
        m.addConstr(w.sum() == 1.0, "budget")
        m.addConstr(mu.values @ w >= target_return, "target_return")
        
        # Cardinality & Linking
        m.addConstr(y.sum() <= k_max, "cardinality")
        m.addConstr(w <= y, "linking")
        
        # Sector Constraints
        if max_sector_pct < 1.0:
            unique_sectors = set(sector_map.values())
            for sec in unique_sectors:
                if sec == 'Unknown': continue
                idx = [i for i, t in enumerate(tickers) if sector_map[t] == sec]
                if idx:
                    m.addConstr(sum(w[i] for i in idx) <= max_sector_pct, name=f"sec_{sec}")

        # Turnover Constraint
        if max_turnover is not None and current_weights is not None:
            cw = np.array([current_weights.get(t, 0.0) for t in tickers])
            delta = m.addMVar(shape=n, lb=0.0, name="delta")
            for i in range(n):
                m.addConstr(delta[i] >= w[i] - cw[i])
                m.addConstr(delta[i] >= cw[i] - w[i])
            # Turnover limit (sum of absolute changes <= 2 * max_turnover)
            m.addConstr(delta.sum() <= 2 * max_turnover, "turnover")

        m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            return w.X, m.ObjVal
        else:
            return None, None
    except Exception as e:
        return None, None

def solve_max_sharpe_scipy(mu, sigma, rf_rate):
    """Model 2: Maximize Sharpe Ratio using Scipy."""
    n = len(mu)
    args = (mu, sigma, rf_rate)
    
    def neg_sharpe(weights, mu, sigma, rf):
        ret = np.sum(mu * weights)
        risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -(ret - rf) / risk

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = n * [1. / n]
    
    result = minimize(neg_sharpe, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        var = np.dot(result.x.T, np.dot(sigma, result.x))
        return result.x, var
    return None, None

def solve_hrp(price_data):
    """Model 3: Hierarchical Risk Parity (HRP)."""
    corr = price_data.corr()
    dist = np.sqrt((1 - corr) / 2)
    link = linkage(squareform(dist), 'single')
    
    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = get_quasi_diag(link)
    
    cov = price_data.cov()
    inv_diag = 1 / np.diag(cov)
    weights = inv_diag / np.sum(inv_diag)
    
    weight_dict = dict(zip(cov.index, weights))
    final_weights = np.array([weight_dict[t] for t in price_data.columns])
    
    var = np.dot(final_weights.T, np.dot(cov * 252, final_weights))
    
    return final_weights, var

# ==========================================
# 3. BACKTESTING ENGINE
# ==========================================

def run_backtest(price_data, weights, initial_capital=10000):
    norm_prices = price_data / price_data.iloc[0]
    portfolio_value = (norm_prices * weights).sum(axis=1) * initial_capital
    n = len(weights)
    bench_weights = np.array([1/n]*n)
    bench_value = (norm_prices * bench_weights).sum(axis=1) * initial_capital
    return portfolio_value, bench_value

# ==========================================
# 4. STREAMLIT APP LAYOUT
# ==========================================

st.set_page_config(page_title="OpFin Ultimate", layout="wide", page_icon="📈")

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.title("🌐 Market & Settings")
market_choice = st.sidebar.selectbox("Select Market", ["USA (USD)", "India (INR)"])
config = get_market_config(market_choice)

# --- SIDEBAR: ASSET SELECTION ---
st.sidebar.header("1. Define Asset Universe")
asset_dict = ASSET_CLASSES.get(market_choice, {})
category_choice = st.sidebar.selectbox("Select Category", ["Custom"] + list(asset_dict.keys()))

if category_choice != "Custom":
    default_text = ", ".join(asset_dict[category_choice])
else:
    if "India" in market_choice:
        default_text = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, NIFTYBEES.NS, GOLDBEES.NS"
    else:
        default_text = "AAPL, MSFT, SPY, TLT, GLD, BTC-USD"

ticker_input = st.sidebar.text_area("Enter Tickers (comma separated)", value=default_text, height=100)

# --- SIDEBAR: OPTIMIZATION PARAMS ---
st.sidebar.header("2. Optimization Model")
model_choice = st.sidebar.selectbox("Choose Objective", 
                                    ["Minimize Risk (Mean-Variance)", 
                                     "Maximize Sharpe Ratio", 
                                     "Hierarchical Risk Parity (HRP)"])

st.sidebar.subheader("Constraints")
target_return_input = st.sidebar.number_input("Min. Annual Return Target (%)", 5.0, 100.0, 15.0, 1.0) / 100.0
k_max = st.sidebar.slider("Max Assets (Cardinality)", 2, 20, 5)
max_sector_exp = st.sidebar.slider("Max Sector Allocation (%)", 10, 100, 100, 5) / 100.0

st.sidebar.subheader("Rebalancing")
use_turnover = st.sidebar.checkbox("Enable Turnover Limit")
max_turnover_pct = 1.0
if use_turnover:
    max_turnover_pct = st.sidebar.slider("Max Turnover (%)", 5, 100, 20, 5) / 100.0
    st.sidebar.info("Enter current weights in 'Optimization' tab before running.")

# --- MAIN APP LOGIC ---
st.title(f"OpFin: Ultimate Portfolio Tool ({market_choice})")

if 'data_download' not in st.session_state:
    st.session_state['data_download'] = None

# FETCH DATA BUTTON
if st.sidebar.button("🚀 Load Data & Analyze"):
    raw_tickers = [t.strip() for t in ticker_input.split(',')]
    fmt_tickers = [format_ticker(t, config['suffix']) for t in raw_tickers]
    
    with st.spinner(f"Fetching data for {len(fmt_tickers)} assets..."):
        prices, fundamentals, sec_map, valid_tkrs = fetch_financial_data(fmt_tickers, config)
        
        if prices is None:
            st.error("No valid data found. Check tickers.")
        else:
            st.session_state['prices'] = prices
            st.session_state['fundamentals'] = fundamentals
            st.session_state['sec_map'] = sec_map
            st.session_state['valid_tkrs'] = valid_tkrs
            st.success(f"Loaded {len(valid_tkrs)} assets successfully!")

# --- DISPLAY TABS ---
if st.session_state.get('prices') is not None:
    prices = st.session_state['prices']
    fundamentals = st.session_state['fundamentals']
    sec_map = st.session_state['sec_map']
    valid_tkrs = st.session_state['valid_tkrs']
    
    mu, sigma, log_returns = process_market_data(prices)
    
    # Reordered tabs to make Asset Index first
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📚 Asset Index", "📊 Fundamental Analysis", "🔍 Single Asset Analysis", "🧮 Optimization", "📉 Backtesting", "📈 Efficient Frontier"])
    
    # --- TAB 1: ASSET INDEX (New) ---
    with tab1:
        st.subheader("Complete Asset Index")
        st.write("Browse and filter all available assets.")
        
        # Sorting Options for Asset Index
        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            sort_col_index = st.selectbox("Sort Index By:", ["Ticker", "Name", "Sector", "Price", "CAGR 5Y"], key="index_sort")
        with col_sort2:
            ascending_index = st.checkbox("Ascending Order", value=True, key="index_asc")
        
        # Prepare Data for Display
        index_df = fundamentals[['Ticker', 'Name', 'Sector', 'Price', 'CAGR 5Y']].copy()
        sorted_index_df = index_df.sort_values(by=sort_col_index, ascending=ascending_index)
        
        # Formatting
        styled_index_df = sorted_index_df.style.format({
            "Price": f"{config['currency']}{{:.2f}}", 
            "CAGR 5Y": "{:.2f}%"
        })
        
        st.dataframe(styled_index_df, use_container_width=True)

    # --- TAB 2: FUNDAMENTALS ---
    with tab2:
        st.subheader("Asset Universe Scorecard")
        
        # Sorting Options
        sort_col = st.selectbox("Sort By:", ["Ticker", "Sector", "CAGR 5Y", "P/E", "Div Yield %"])
        ascending = st.checkbox("Ascending", value=True)
        
        # Color Coding
        def color_pe(val):
            if pd.isna(val): return ''
            color = '#90ee90' if val < 20 else '#ffcccb' if val > 50 else ''
            return f'background-color: {color}'
        def color_cagr(val):
            if pd.isna(val): return ''
            color = '#90ee90' if val > 20 else '#ffcccb' if val < 5 else ''
            return f'background-color: {color}'

        sorted_df = fundamentals.sort_values(by=sort_col, ascending=ascending)
        
        styled_df = sorted_df.style.applymap(color_pe, subset=['P/E'])\
                                   .applymap(color_cagr, subset=['CAGR 5Y'])\
                                   .format({"Price": f"{config['currency']}{{:.2f}}", "P/E": "{:.1f}", "Div Yield %": "{:.2f}%", "CAGR 5Y": "{:.2f}%"})
        
        st.dataframe(styled_df, use_container_width=True)
        st.caption("Green: High Growth/Value | Red: Low Growth/Expensive")

    # --- TAB 3: SINGLE ASSET ANALYSIS ---
    with tab3:
        st.subheader("Deep Dive Analysis")
        selected_ticker = st.selectbox("Select Asset to Analyze", valid_tkrs)
        
        if selected_ticker:
            try:
                t = yf.Ticker(selected_ticker)
                info = t.info
                
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.write(f"### {info.get('shortName', selected_ticker)}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Business Summary:**")
                    st.write(info.get('longBusinessSummary', 'No summary available.')[:400] + "...")
                    
                    st.write("#### Key Ratios")
                    metrics = {
                        "Previous Close": info.get('previousClose'),
                        "Open": info.get('open'),
                        "Market Cap": info.get('marketCap'),
                        "Trailing P/E": info.get('trailingPE'),
                        "Price/Book": info.get('priceToBook'),
                        "Beta": info.get('beta')
                    }
                    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).set_index("Metric"))

                with col_b:
                    st.write("#### Price Chart (1 Year)")
                    hist_1y = t.history(period="1y")
                    fig_candle = go.Figure(data=[go.Candlestick(x=hist_1y.index,
                                    open=hist_1y['Open'],
                                    high=hist_1y['High'],
                                    low=hist_1y['Low'],
                                    close=hist_1y['Close'])])
                    fig_candle.update_layout(xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_candle, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Could not load detailed data for {selected_ticker}: {e}")

    # --- TAB 4: OPTIMIZATION ---
    with tab4:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Strategy: {model_choice}")
            curr_weights_dict = None
            if use_turnover:
                st.write(" **Current Portfolio Weights** (Must sum to 1.0)")
                init_df = pd.DataFrame({'Asset': valid_tkrs, 'Current Weight': [0.0]*len(valid_tkrs)})
                edited_w = st.data_editor(init_df, hide_index=True)
                if abs(edited_w['Current Weight'].sum() - 1.0) > 0.01:
                    st.warning("Current weights do not sum to 1.0.")
                else:
                    curr_weights_dict = dict(zip(edited_w['Asset'], edited_w['Current Weight']))

            if st.button("Run Solver"):
                weights = None
                opt_var = 0
                with st.spinner("Optimizing..."):
                    if model_choice == "Minimize Risk (Mean-Variance)":
                        weights, opt_var = solve_mean_variance_gurobi(
                            mu, sigma, target_return_input, k_max, sec_map, max_sector_exp, 
                            current_weights=curr_weights_dict, max_turnover=max_turnover_pct if use_turnover else None
                        )
                        if weights is None: st.error("Optimization Infeasible.")
                    elif model_choice == "Maximize Sharpe Ratio":
                        rf_rate = 0.04 
                        weights, opt_var = solve_max_sharpe_scipy(mu, sigma, rf_rate)
                    elif model_choice == "Hierarchical Risk Parity (HRP)":
                        weights, opt_var = solve_hrp(prices)

                if weights is not None:
                    clean_weights = {valid_tkrs[i]: w for i, w in enumerate(weights) if w > 0.001}
                    res_df = pd.DataFrame(list(clean_weights.items()), columns=['Asset', 'Weight'])
                    res_df['Sector'] = res_df['Asset'].map(sec_map)
                    
                    ret = np.sum(mu * weights)
                    risk = np.sqrt(opt_var)
                    sharpe = ret / risk if risk > 0 else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Exp. Annual Return", f"{ret*100:.2f}%")
                    m2.metric("Annual Risk (Vol)", f"{risk*100:.2f}%")
                    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_asset = px.pie(res_df, values='Weight', names='Asset', title="Allocation by Asset", hole=0.4)
                        st.plotly_chart(fig_asset, use_container_width=True)
                    with c2:
                        fig_sec = px.pie(res_df, values='Weight', names='Sector', title="Allocation by Sector", hole=0.4)
                        st.plotly_chart(fig_sec, use_container_width=True)
                        
                    st.session_state['opt_weights'] = weights 

        with col2:
            st.write("### Chosen Portfolio")
            if 'opt_weights' in st.session_state:
                 w_display = st.session_state['opt_weights']
                 clean_w = {valid_tkrs[i]: w for i, w in enumerate(w_display) if w > 0.001}
                 disp_df = pd.DataFrame(list(clean_w.items()), columns=['Asset', 'Weight'])
                 st.dataframe(disp_df.sort_values('Weight', ascending=False).style.format({"Weight": "{:.2%}"}), use_container_width=True)

    # --- TAB 5: BACKTESTING ---
    with tab5:
        st.subheader("Historical Simulation (In-Sample)")
        if 'opt_weights' in st.session_state:
            port_val, bench_val = run_backtest(prices, st.session_state['opt_weights'])
            backtest_df = pd.DataFrame({
                "Optimized Strategy": port_val,
                "Benchmark (Equal Wt)": bench_val
            })
            fig_bt = px.line(backtest_df, title="Portfolio Performance vs Benchmark (Growth of 10k)")
            st.plotly_chart(fig_bt, use_container_width=True)
            
            total_ret = (port_val.iloc[-1] / port_val.iloc[0]) - 1
            bench_ret = (bench_val.iloc[-1] / bench_val.iloc[0]) - 1
            st.metric("Strategy Total Return", f"{total_ret*100:.2f}%", delta=f"{(total_ret-bench_ret)*100:.2f}% vs Bench")
        else:
            st.info("Please run Optimization in Tab 3 first.")

    # --- TAB 6: EFFICIENT FRONTIER ---
    with tab6:
        st.subheader("Efficient Frontier Analysis")
        n_sim = 200
        results = np.zeros((3, n_sim))
        for i in range(n_sim):
            rw = np.random.random(len(mu))
            rw /= np.sum(rw)
            p_ret = np.sum(mu * rw)
            p_risk = np.sqrt(np.dot(rw.T, np.dot(sigma, rw)))
            results[0,i] = p_risk
            results[1,i] = p_ret
            results[2,i] = p_ret / p_risk
            
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers', 
                                    marker=dict(color=results[2,:], colorscale='Viridis', showscale=True),
                                    name='Random Portfolios'))
        
        if 'opt_weights' in st.session_state:
             w_opt = st.session_state['opt_weights']
             ret_opt = np.sum(mu * w_opt)
             risk_opt = np.sqrt(np.dot(w_opt.T, np.dot(sigma, w_opt)))
             fig_ef.add_trace(go.Scatter(x=[risk_opt], y=[ret_opt], mode='markers',
                                         marker=dict(color='red', size=15, symbol='star'),
                                         name='Your Portfolio'))

        fig_ef.update_layout(xaxis_title="Risk (Volatility)", yaxis_title="Return", title="Risk-Return Landscape")
        st.plotly_chart(fig_ef, use_container_width=True)
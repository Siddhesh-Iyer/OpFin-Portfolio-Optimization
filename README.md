# OpFin: Advanced Portfolio Optimization Dashboard

**OpFin** is a comprehensive quantitative finance platform built with **Streamlit** and **Gurobi**. It allows users to construct optimal portfolios across US and Indian markets using advanced mathematical models, including Mean-Variance Optimization (MVO) and Hierarchical Risk Parity (HRP).

##  Key Features

### 1. Multi-Market Support
* **USA (USD):** S&P 500, Nasdaq 100 constituents, ETFs, and Crypto.
* **India (INR):** Nifty 50, Bank Nifty, and Sectoral Indices.
* **Asset Universe:** Automated fetching of stocks, bonds, commodities, and crypto using `yfinance`.

### 2. Optimization Models
* **Gurobi Mean-Variance (MIQP):** Solves for minimum risk with **Mixed-Integer constraints**:
    * *Cardinality Constraints:* Limit the number of assets in the portfolio (e.g., "Select exactly 5 stocks").
    * *Sector Constraints:* Cap exposure to specific sectors (e.g., "Max 20% in Tech").
    * *Turnover Constraints:* Limit trading costs during rebalancing.
* **Max Sharpe Ratio:** Classic tangency portfolio optimization using `scipy`.
* **Hierarchical Risk Parity (HRP):** Machine learning-based approach using clustering (linkage) to allocate weights based on asset hierarchy rather than covariance inversion.

### 3. Analytics & Visualization
* **Efficient Frontier:** Monte Carlo simulation to visualize the risk-return landscape.
* **Backtesting Engine:** Historical performance comparison against an equal-weight benchmark.
* **Fundamental Scorecard:** Analysis of P/E, P/B, Dividend Yield, and 5-year CAGR.

##  Tech Stack

* **Frontend:** Streamlit
* **Optimization:** Gurobi (gurobipy), Scipy
* **Data:** yfinance (Real-time data), Pandas, NumPy
* **Visualization:** Plotly (Interactive charts), Matplotlib

##  Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/Siddhesh-Iyer/OpFin-Portfolio-Optimization.git](https://github.com/Siddhesh-Iyer/OpFin-Portfolio-Optimization.git)
    cd OpFin-Portfolio-Optimization
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run OpFin_testing.py
    ```

## 📝 Usage Guide

1.  **Select Market:** Choose between USA or India in the sidebar.
2.  **Define Universe:** Import top assets by CAGR or enter a custom list of tickers.
3.  **Load Data:** Click "Load Data & Analyze" to fetch real-time prices.
4.  **Optimize:** Go to the "Optimization" tab, select your model (e.g., Mean-Variance), set constraints (Max Assets, Target Return), and run the solver.
5.  **Analyze:** View the generated pie charts and backtest the strategy in the "Backtesting" tab.

##  Requirements

* **Gurobi License:** This project uses `gurobipy`. You need a valid Gurobi license (Academic or Commercial) installed on your machine.
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Linear Optimization Portfolio Manager")
st.markdown("Upload your market data and define your constraints to generate an optimal portfolio.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 'combined_data.xlsx'", type=['xlsx'])

st.sidebar.header("2. Portfolio Constraints")
aum_input = st.sidebar.number_input(
    "Total AUM ($)", 
    min_value=1000, 
    value=10_000_000, 
    step=100_000,
    format="%d"
)

term_years = st.sidebar.selectbox(
    "Investment Horizon (Years)", 
    options=[1, 3, 5, 10], 
    index=2
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Parameters")

risk_max = st.sidebar.slider(
    "Max Risk Score (1-10)", 
    min_value=1.0, 
    max_value=10.0, 
    value=6.0,
    step=0.1,
    help="1 = Low Volatility, 10 = High Volatility"
)

hedge_min = st.sidebar.slider(
    "Min Hedge Score (1-10)", 
    min_value=1.0, 
    max_value=10.0, 
    value=4.0,
    step=0.1,
    help="Higher score = Better hedging properties (e.g., Gold, Bonds)"
)

concentration_limit = st.sidebar.slider(
    "Max Concentration per Asset (%)", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.20,
    step=0.05
)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_and_process_data(file, years):
    """Loads data and calculates metrics. Cached for performance."""
    df = pd.read_excel(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Return'])

    # Filter for term
    latest_date = df['Date'].max()
    start_date = latest_date - pd.DateOffset(years=years)
    df_subset = df[df['Date'] > start_date].copy()

    # Calc Metrics
    summary = df_subset.groupby(['Ticker', 'Type'])['Return'].agg(['mean', 'std']).reset_index()
    summary.rename(columns={'mean': 'Expected_Return', 'std': 'Volatility'}, inplace=True)

    # Risk Score
    min_vol = summary['Volatility'].min()
    max_vol = summary['Volatility'].max()
    summary['Risk_Score'] = 1 + 9 * (summary['Volatility'] - min_vol) / (max_vol - min_vol)

    # Hedge Score logic
    def assign_hedge_score(row):
        if row['Type'] == 'commodity': return 9.0
        if row['Type'] == 'fixed':     return 7.0
        if row['Type'] == 'stock':     return 3.0
        return 5.0

    summary['Hedge_Score'] = summary.apply(assign_hedge_score, axis=1)
    
    return summary

def run_optimization(summary, risk_cap, hedge_floor, limit, aum):
    n_assets = len(summary)
    
    # Objective: Maximize Return (Negative for minimization function)
    c = -summary['Expected_Return'].values

    # Constraints
    # 1. Sum of weights = 1
    A_eq = np.ones((1, n_assets))
    b_eq = np.array([1.0])

    # 2. Risk <= risk_cap AND Hedge >= hedge_floor
    # Note: linprog uses <=, so for Hedge >= X, we use -Hedge <= -X
    A_ub = np.vstack([
        summary['Risk_Score'].values,
        -summary['Hedge_Score'].values
    ])
    b_ub = np.array([
        risk_cap,
        -hedge_floor
    ])

    # 3. Individual asset limits (0 to concentration_limit)
    bounds = [(0, limit) for _ in range(n_assets)]

    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        weights = res.x
        holdings = pd.DataFrame({
            'Ticker': summary['Ticker'],
            'Type': summary['Type'],
            'Weight': weights,
            'Allocation ($)': weights * aum,
            'Expected Return': summary['Expected_Return']
        })
        
        # Filter insignificant holdings
        active_holdings = holdings[holdings['Weight'] > 0.001].sort_values(by='Weight', ascending=False).copy()
        
        portfolio_return = np.dot(weights, summary['Expected_Return'])
        return True, active_holdings, portfolio_return
    else:
        return False, None, None

# --- MAIN APP LOGIC ---

if uploaded_file is not None:
    try:
        # 1. Process Data
        summary_df = load_and_process_data(uploaded_file, term_years)
        
        # 2. Run Optimization
        success, results_df, exp_return = run_optimization(
            summary_df, risk_max, hedge_min, concentration_limit, aum_input
        )

        if success:
            # --- DASHBOARD LAYOUT ---
            
            # Top Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Allocation", f"${aum_input:,.0f}")
            col2.metric("Est. Monthly Return", f"{exp_return:.2%}")
            col3.metric("Assets Selected", f"{len(results_df)}")

            st.markdown("---")

            # Charts and Tables
            c1, c2 = st.columns([1, 1])

            with c1:
                st.subheader("Asset Allocation")
                fig = px.pie(results_df, values='Allocation ($)', names='Ticker', 
                             title=f"Portfolio Composition (Max {concentration_limit:.0%})",
                             hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Holdings Details")
                
                # Formatting for display
                display_df = results_df.copy()
                display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
                display_df['Allocation ($)'] = display_df['Allocation ($)'].map('${:,.2f}'.format)
                display_df['Expected Return'] = display_df['Expected Return'].map('{:.2%}'.format)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Asset Class Breakdown
            st.subheader("Exposure by Asset Class")
            class_group = results_df.groupby('Type')['Allocation ($)'].sum().reset_index()
            fig_bar = px.bar(class_group, x='Type', y='Allocation ($)', color='Type')
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.error("Optimization Failed: No portfolio could satisfy these constraints.")
            st.warning("Try increasing the 'Max Risk Score' or lowering the 'Min Hedge Score'.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("👋 Please upload your Excel file in the sidebar to begin.")

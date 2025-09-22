import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import math
from datetime import date, datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from utils.option_utils import (
    black_scholes_iv, bs_price, bs_delta, bs_gamma, bs_theta, bs_vega,
    calculate_intrinsic_value, validate_option_price
)
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants
DB_PATH = "market_data.db"
CONTRACTS_PATH = "contracts.csv"
RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.012
PRICE_COL_CANDIDATES = ["ltp", "last_price", "close"]
BID_COL_CANDIDATES = ["bid", "bp1", "buy_price"]
ASK_COL_CANDIDATES = ["ask", "ap1", "sell_price"]

# Time intervals for forward projection (in minutes)
TIME_INTERVALS = [5, 10, 15, 30, 60, 120, 180, 240, 300, 375]
MAX_TIME = max(TIME_INTERVALS)  # For linear scaling

st.set_page_config(
    page_title="Options Forward Price Simulator", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_contracts():
    """Load and process contracts data"""
    try:
        contracts_df = pd.read_csv(CONTRACTS_PATH, dtype=str)
        contracts_df['token'] = pd.to_numeric(contracts_df['token'], errors='coerce').astype('Int64')
        contracts_df['strike'] = pd.to_numeric(contracts_df['strike'], errors='coerce')
        contracts_df['expiry'] = pd.to_datetime(contracts_df['expiry'], errors='coerce').dt.date
        contracts_df['option_type'] = contracts_df['option_type'].str.upper().str.strip()
        contracts_df = contracts_df[['token', 'expiry', 'strike', 'option_type', 'symbol']]
        logging.info(f"Contracts loaded: {len(contracts_df)} rows")
        return contracts_df
    except Exception as e:
        logging.error(f"Error loading contracts: {e}")
        return pd.DataFrame()

def get_mid_price(row, bid_cols, ask_cols, price_col):
    """Enhanced price calculation with spread validation"""
    bid, ask = None, None
    
    for c in bid_cols:
        if c in row and pd.notna(row[c]) and row[c] > 0:
            bid = float(row[c])
            break
    for c in ask_cols:
        if c in row and pd.notna(row[c]) and row[c] > 0:
            ask = float(row[c])
            break
    
    if bid is not None and ask is not None and ask > bid:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else np.inf
        if spread > 0 and spread_pct < 0.5:
            return mid, 'mid', spread_pct
        else:
            if price_col in row and pd.notna(row[price_col]) and row[price_col] > 0:
                return float(row[price_col]), 'ltp_wide_spread', spread_pct
    
    if price_col in row and pd.notna(row[price_col]) and row[price_col] > 0:
        return float(row[price_col]), 'ltp', np.nan
    
    return None, 'none', np.nan

def get_spot_from_options(df_opts, expiry_date):
    """Enhanced spot price calculation using put-call parity"""
    if df_opts.empty:
        return None, "no_data"
    
    strikes = sorted(df_opts['strike'].dropna().unique())
    if not strikes:
        return None, "no_strikes"
    
    best_spot_estimates = []
    
    for K in strikes:
        call_data = df_opts[(df_opts['strike'] == K) & (df_opts['option_type'] == "CE")]
        put_data = df_opts[(df_opts['strike'] == K) & (df_opts['option_type'] == "PE")]
        
        if not call_data.empty and not put_data.empty:
            call_price = call_data['price'].iloc[0]
            put_price = put_data['price'].iloc[0]
            
            if call_price > 0 and put_price > 0:
                T = max((expiry_date - date.today()).days, 1) / 365.0
                spot_estimate = (call_price - put_price + K * math.exp(-RISK_FREE_RATE * T)) * math.exp(DIVIDEND_YIELD * T)
                
                price_diff = abs(call_price - put_price)
                weight = 1 / (1 + price_diff)
                best_spot_estimates.append((spot_estimate, weight, K, price_diff))
    
    if best_spot_estimates:
        best_spot_estimates.sort(key=lambda x: x[1], reverse=True)
        top_estimates = best_spot_estimates[:3]
        
        weighted_sum = sum(est[0] * est[1] for est in top_estimates)
        total_weight = sum(est[1] for est in top_estimates)
        
        if total_weight > 0:
            spot = weighted_sum / total_weight
            return spot, f"put_call_parity_weighted_{len(top_estimates)}"
    
    return np.median(strikes), "median_all"

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_market_data():
    """Load market data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        md = pd.read_sql("SELECT * FROM market_data", conn, parse_dates=['timestamp', 'exc_timestamp'])
        conn.close()
        
        if md.empty:
            return pd.DataFrame(), None, "no_data_in_table"
        
        contracts_df = load_contracts()
        if contracts_df.empty:
            return pd.DataFrame(), None, "no_contracts"
        
        md['token'] = pd.to_numeric(md['token'], errors='coerce').astype('Int64')
        md_latest = md.sort_values('timestamp').groupby('token').last().reset_index()
        
        df = md_latest.merge(contracts_df, on='token', how='inner')
        if df.empty:
            return pd.DataFrame(), None, "no_contracts_matched"
        
        # Calculate prices
        price_col = next((c for c in PRICE_COL_CANDIDATES if c in df.columns), None)
        bid_cols = [c for c in BID_COL_CANDIDATES if c in df.columns]
        ask_cols = [c for c in ASK_COL_CANDIDATES if c in df.columns]
        
        df['price'] = 0.0
        df['price_quality'] = 'none'
        df['spread_pct'] = np.nan
        
        for idx, row in df.iterrows():
            price, quality, spread_pct = get_mid_price(row, bid_cols, ask_cols, price_col)
            if price is not None:
                df.at[idx, 'price'] = price
                df.at[idx, 'price_quality'] = quality
                df.at[idx, 'spread_pct'] = spread_pct
        
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
        df["oi"] = pd.to_numeric(df.get("oi", 0), errors="coerce").fillna(0).astype(int)
        
        # Filter valid options
        df_opts = df[(df['price_quality'] != 'none') | (df['price'].notna() & (df['price'] > 0))].copy()
        
        if df_opts.empty:
            return pd.DataFrame(), None, "no_valid_prices"
        
        # Calculate spot price
        nearest_expiry = df_opts['expiry'].min()
        nearest_expiry_opts = df_opts[df_opts['expiry'] == nearest_expiry]
        spot_price, spot_method = get_spot_from_options(nearest_expiry_opts, nearest_expiry)
        
        return df_opts, spot_price, spot_method
        
    except Exception as e:
        logging.error(f"Error loading market data: {e}")
        return pd.DataFrame(), None, f"error: {str(e)}"

def find_closest_atm_strike(df_opts, expiry_date, spot_price):
    """Find the closest at-the-money strike for fallback IV calculation"""
    try:
        expiry_opts = df_opts[df_opts['expiry'] == expiry_date]
        if expiry_opts.empty:
            return None
        
        strikes = expiry_opts['strike'].unique()
        closest_strike = min(strikes, key=lambda x: abs(x - spot_price))
        return closest_strike
    except:
        return None

def get_current_iv_enhanced(df_opts, strike, option_type, expiry_date, spot_price):
    """
    ENHANCED: Get current IV with better validation and fallbacks
    """
    try:
        option_data = df_opts[
            (df_opts['strike'] == strike) & 
            (df_opts['option_type'] == option_type) & 
            (df_opts['expiry'] == expiry_date)
        ]
        
        if option_data.empty:
            return None, None, None
        
        current_price = option_data['price'].iloc[0]
        timestamp = option_data['timestamp'].iloc[0]
        
        if current_price <= 0:
            return None, None, None
        
        # Calculate time to expiry accurately
        expiry_dt = datetime.combine(expiry_date, datetime.max.time())
        total_seconds_remaining = (expiry_dt - timestamp).total_seconds()
        days_remaining = max(total_seconds_remaining / (24 * 3600), 1/365)
        T = days_remaining / 365.0
        
        # Calculate and validate intrinsic value
        intrinsic = calculate_intrinsic_value(spot_price, strike, option_type, RISK_FREE_RATE, DIVIDEND_YIELD, T)
        
        # Enhanced price validation
        if not validate_option_price(current_price, intrinsic, 0.005):
            logging.debug(f"Price {current_price} failed validation against intrinsic {intrinsic} for {strike} {option_type}")
            return None, None, None
        
        # Calculate IV using enhanced Black-Scholes
        iv = black_scholes_iv(spot_price, strike, T, current_price, RISK_FREE_RATE, DIVIDEND_YIELD, option_type.lower())
        
        # Enhanced IV validation with realistic Indian options ranges
        if pd.isna(iv) or iv <= 0.03 or iv >= 5.0:  # 3% to 500% range
            logging.debug(f"IV {iv} out of realistic range for {strike} {option_type}")
            
            # IMPROVED FALLBACK: Use ATM IV with skew adjustments
            atm_strike = find_closest_atm_strike(df_opts, expiry_date, spot_price)
            if atm_strike and atm_strike != strike:
                atm_data = df_opts[
                    (df_opts['strike'] == atm_strike) & 
                    (df_opts['option_type'] == option_type) & 
                    (df_opts['expiry'] == expiry_date)
                ]
                if not atm_data.empty:
                    atm_price = atm_data['price'].iloc[0]
                    atm_intrinsic = calculate_intrinsic_value(spot_price, atm_strike, option_type, RISK_FREE_RATE, DIVIDEND_YIELD, T)
                    
                    if validate_option_price(atm_price, atm_intrinsic, 0.005):
                        atm_iv = black_scholes_iv(spot_price, atm_strike, T, atm_price, RISK_FREE_RATE, DIVIDEND_YIELD, option_type.lower())
                        if not pd.isna(atm_iv) and 0.03 <= atm_iv <= 5.0:
                            # Apply realistic volatility skew
                            forward = spot_price * math.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * T)
                            moneyness = strike / forward
                            
                            if option_type == "CE":
                                # Call skew: higher strike = lower IV (smile effect)
                                skew_adj = max(0.7, min(1.4, 1.0 - (moneyness - 1.0) * 0.3))
                            else:
                                # Put skew: higher strike = higher IV (smile effect)
                                skew_adj = max(0.7, min(1.4, 1.0 + (moneyness - 1.0) * 0.3))
                            
                            iv = atm_iv * skew_adj
                            logging.debug(f"Using ATM-based IV: {iv:.4f} (ATM: {atm_iv:.4f}, skew: {skew_adj:.3f})")
            
            # Final validation after fallback
            if pd.isna(iv) or iv <= 0.03 or iv >= 5.0:
                return None, None, None
                
        return current_price, float(iv), timestamp
        
    except Exception as e:
        logging.debug(f"IV calculation failed for {strike} {option_type}: {e}")
        return None, None, None

def calculate_forward_price_linear_decay(current_price, spot_current, spot_forward, strike, T_current, T_forward, 
                                       current_iv, vol_shock, price_decay_rate, option_type, time_elapsed_years):
    """
    CORRECTED: Calculate forward price with LINEAR price decay (not IV decay)
    
    Args:
        price_decay_rate: Annual price decay rate as percentage (e.g., 10.0 for 10%)
        time_elapsed_years: Time elapsed in years
    """
    try:
        # Step 1: Apply volatility shock to current IV
        new_iv = current_iv * (1 + vol_shock / 100)
        new_iv = max(min(new_iv, 5.0), 0.01)  # Bound between 1% and 500%
        
        # Step 2: Calculate theoretical forward price using Black-Scholes
        theoretical_forward_price = bs_price(
            spot_forward, 
            strike, 
            T_forward, 
            new_iv, 
            RISK_FREE_RATE, 
            DIVIDEND_YIELD, 
            option_type.lower()
        )
        
        if theoretical_forward_price <= 0:
            return 0.0
        
        # Step 3: Apply LINEAR price decay
        # Linear decay formula: new_price = old_price - (decay_rate * time_elapsed)
        annual_decay_rate = price_decay_rate / 100  # Convert percentage to decimal
        
        # Enhanced decay acceleration based on time to expiry
        current_days_remaining = T_current * 365
        
        if current_days_remaining <= 7:
            # Weekly options: 4x decay acceleration (very aggressive)
            acceleration = 4.0
        elif current_days_remaining <= 15:
            # Bi-weekly options: 3x decay acceleration  
            acceleration = 3.0
        elif current_days_remaining <= 30:
            # Monthly options: 2x decay acceleration
            acceleration = 2.0
        elif current_days_remaining <= 60:
            # Quarterly options: 1.5x decay acceleration
            acceleration = 1.5
        else:
            # Standard decay for longer-dated options
            acceleration = 1.0
        
        # Apply acceleration
        effective_annual_decay = annual_decay_rate * acceleration
        
        # LINEAR DECAY: Subtract absolute decay amount from current price
        price_decay_amount = current_price * effective_annual_decay * time_elapsed_years
        
        # Calculate final forward price
        # Method 1: Apply decay to current price, then adjust for spot/vol changes
        decayed_current_price = max(0.01, current_price - price_decay_amount)
        
        # Calculate ratio of theoretical to current (for spot/vol adjustment)
        if current_price > 0:
            current_theoretical = bs_price(
                spot_current, 
                strike, 
                T_current, 
                current_iv, 
                RISK_FREE_RATE, 
                DIVIDEND_YIELD, 
                option_type.lower()
            )
            
            if current_theoretical > 0:
                adjustment_ratio = theoretical_forward_price / current_theoretical
                final_forward_price = decayed_current_price * adjustment_ratio
            else:
                final_forward_price = decayed_current_price
        else:
            final_forward_price = theoretical_forward_price
        
        # Ensure minimum value and intrinsic value bounds
        intrinsic = calculate_intrinsic_value(spot_forward, strike, option_type, RISK_FREE_RATE, DIVIDEND_YIELD, T_forward)
        final_forward_price = max(final_forward_price, intrinsic + 0.01)
        
        return max(final_forward_price, 0.01)
        
    except Exception as e:
        logging.debug(f"Forward price calculation failed: {e}")
        return 0.01

def create_scenario_table_with_linear_decay(legs_data, spot_current, spot_drift, vol_shock, price_decay_rate, selected_expiry):
    """
    CORRECTED: Create scenario analysis table with LINEAR PRICE decay
    """
    if not legs_data:
        return pd.DataFrame(), None
    
    table_data = []
    data_date = None
    
    for i, leg in enumerate(legs_data):
        # Format timestamp for display
        if leg['timestamp']:
            timestamp_dt = leg['timestamp']
            data_date = timestamp_dt.strftime("%d-%b-%Y")
            time_12hr = timestamp_dt.strftime("%I:%M:%S %p")
        else:
            time_12hr = "N/A"
            data_date = "N/A"
        
        row_data = {
            'Leg': i + 1,
            'Time': time_12hr,
            'Strike': leg['strike'],
            'Type': leg['option_type'],
            'Current_Price': f"₹{leg['current_price']:.2f}" if leg['current_price'] else "N/A",
            'Current_IV': f"{leg['current_iv']:.4f}" if leg['current_iv'] else "N/A"
        }
        
        # Calculate current time to expiry
        expiry = leg['expiry']
        timestamp = leg['timestamp']
        expiry_dt = datetime.combine(expiry, datetime.max.time())
        
        total_seconds_remaining = (expiry_dt - timestamp).total_seconds()
        current_days_remaining = max(total_seconds_remaining / (24 * 3600), 1/365)
        T_current = current_days_remaining / 365.0
        
        for minutes in TIME_INTERVALS:
            try:
                # CORRECTED: Linear spot price movement
                drift_fraction = minutes / MAX_TIME  # Linear scaling: 0 to 1
                spot_forward = spot_current * (1 + (spot_drift / 100) * drift_fraction)
                
                # Time progression
                elapsed_time_years = minutes / (365 * 24 * 60)  # Convert minutes to years
                T_forward = max(T_current - elapsed_time_years, 1/365)
                
                # Calculate forward price with LINEAR decay
                if leg['current_iv'] and leg['current_price']:
                    forward_price = calculate_forward_price_linear_decay(
                        leg['current_price'],
                        spot_current,
                        spot_forward, 
                        leg['strike'],
                        T_current,
                        T_forward,
                        leg['current_iv'],
                        vol_shock,
                        price_decay_rate,  # This is the decay rate as percentage
                        leg['option_type'],
                        elapsed_time_years
                    )
                    
                    if forward_price > 0:
                        pnl = forward_price - leg['current_price']
                        pnl_pct = (pnl / leg['current_price']) * 100 if leg['current_price'] > 0 else 0
                        
                        # Enhanced display format with color coding hints
                        if pnl >= 0:
                            row_data[f"{minutes}min"] = f"₹{forward_price:.2f} (+{pnl:.2f}, +{pnl_pct:.1f}%)"
                        else:
                            row_data[f"{minutes}min"] = f"₹{forward_price:.2f} ({pnl:.2f}, {pnl_pct:.1f}%)"
                    else:
                        row_data[f"{minutes}min"] = "₹0.01 (Error)"
                        
                else:
                    row_data[f"{minutes}min"] = "N/A"
                    
            except Exception as e:
                logging.debug(f"Error calculating {minutes}min projection for leg {i+1}: {e}")
                row_data[f"{minutes}min"] = "Error"
        
        table_data.append(row_data)
    
    return pd.DataFrame(table_data), data_date

# Streamlit UI
def main():
    st.title("🔮 Forward Price & Options Scenario Simulator")
    st.markdown("**Real-time multi-leg option strategy analysis with LINEAR price decay**")
    
    # Load market data
    with st.spinner("Loading market data..."):
        df_opts, spot_price, spot_method = load_market_data()
    
    if df_opts.empty or spot_price is None:
        st.error("❌ Unable to load market data or calculate spot price")
        st.info("Please check your database connection and data availability")
        return
    
    if df_opts['strike'].unique().size == 0:
        st.error("❌ No strikes available in any expiry. Check your data files.")
        return
    
    # Filter for NIFTY and BANKNIFTY options only
    df_indices = df_opts[df_opts['symbol'].isin(["NIFTY", "BANKNIFTY"])].copy()
    if df_indices.empty:
        st.error("❌ No NIFTY or BANKNIFTY options found in the data")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Live Market Data")
    st.sidebar.metric("Current Spot Price", f"₹{spot_price:.2f}", f"Method: {spot_method}")
    
    # Scenario inputs
    st.sidebar.header("🎛️ Scenario Parameters")
    spot_drift = st.sidebar.slider(
        "Spot Drift (%)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Projected percentage change in spot price at 375min"
    )
    
    vol_shock = st.sidebar.slider(
        "Volatility Shock (%)",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        help="Percentage change in implied volatility"
    )
    
    price_decay_rate = st.sidebar.slider(
        "Price Decay Rate (% per year)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
        help="LINEAR price reduction per year, with acceleration for short-dated options"
    )
    
    # Display projected spot price
    projected_spot = spot_price * (1 + spot_drift / 100)
    st.sidebar.metric("Projected Spot Price (at 375min)", f"₹{projected_spot:.2f}", f"{spot_drift:+.1f}%")
    
    # Display decay methodology
    st.sidebar.info("📉 **Linear Decay Model**\n\n"
                   "• Weekly (<7d): 4x acceleration\n"
                   "• Bi-weekly (<15d): 3x acceleration\n" 
                   "• Monthly (<30d): 2x acceleration\n"
                   "• Quarterly (<60d): 1.5x acceleration\n"
                   "• Long-term (>60d): Standard rate")
    
    # Add Option Legs
    st.header("🎯 Add Option Legs")
    
    # Horizontal layout for inputs
    col_symbol, col_strike, col_expiry, col_type, col_submit = st.columns([2, 2, 2, 2, 1])
    with col_symbol:
        symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], key="symbol_select")
    
    # Filter strikes by selected symbol
    symbol_options = df_indices[df_indices['symbol'] == symbol]
    available_strikes = sorted(symbol_options['strike'].unique())
    
    with col_strike:
        if not available_strikes:
            st.warning("No strikes available for selected symbol")
            strike = None
        else:
            strike = st.selectbox("Strike Price", available_strikes, key="strike_select")
    
    # Filter expiries by selected symbol and strike
    if strike:
        strike_options = df_indices[(df_indices['symbol'] == symbol) & (df_indices['strike'] == strike)]
        available_expiries_for_strike = sorted(strike_options['expiry'].unique())
        available_expiries_for_strike = [exp for exp in available_expiries_for_strike if exp > date.today()]
    else:
        available_expiries_for_strike = []
    
    with col_expiry:
        if not available_expiries_for_strike:
            st.warning("No expiries available for selected strike")
            expiry = None
        else:
            expiry = st.selectbox("Expiry", available_expiries_for_strike, format_func=lambda x: x.strftime("%d-%b-%Y"), key="expiry_select")
    
    with col_type:
        option_type = st.selectbox("Option Type", ["CE", "PE"], key="type_select")
    
    with col_submit:
        submit_leg = st.button("➕ Add Leg", disabled=not (strike and expiry), key="add_leg_button")
    
    # Initialize session state for legs
    if 'legs_data' not in st.session_state:
        st.session_state.legs_data = []
    
    if submit_leg:
        # Get current price, IV, and timestamp for the selected option
        current_price, current_iv, timestamp = get_current_iv_enhanced(df_indices, strike, option_type, expiry, spot_price)
        
        if current_price is None:
            st.error(f"❌ No valid data for {symbol} {option_type} {strike} on {expiry.strftime('%d-%b-%Y')}")
        else:
            leg_data = {
                'symbol': symbol,
                'strike': strike,
                'option_type': option_type,
                'current_price': current_price,
                'current_iv': current_iv,
                'expiry': expiry,
                'timestamp': timestamp
            }
            
            st.session_state.legs_data.append(leg_data)
            st.success(f"✅ Added {symbol} {option_type} {strike} (Expiry: {expiry.strftime('%d-%b-%Y')})")
            st.rerun()
    
    # Display current legs with inline delete button
    if st.session_state.legs_data:
        st.subheader("Current Legs")
        for i, leg in enumerate(st.session_state.legs_data):
            price_str = f"₹{leg['current_price']:.2f}" if leg['current_price'] else "N/A"
            iv_str = f"{leg['current_iv']:.4f}" if leg['current_iv'] else "N/A"
            exp_str = leg['expiry'].strftime("%d-%b-%Y")
            ts_str = leg['timestamp'].strftime("%I:%M:%S %p") if leg['timestamp'] else "N/A"
            leg_text = f"**Leg {i+1}:** {leg['symbol']} {leg['option_type']} {leg['strike']} ({exp_str}) - {price_str} (IV: {iv_str}, Time: {ts_str})"
            col_leg, col_button = st.columns([5, 1])
            with col_leg:
                st.write(leg_text)
            with col_button:
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.legs_data.pop(i)
                    st.rerun()
    else:
        st.info("No legs added yet. Add your first leg above.")
    
    # Scenario Table
    st.header("📋 Forward Price Scenario Table")
    
    if st.session_state.legs_data:
        # Create and display the scenario table
        scenario_df, data_date = create_scenario_table_with_linear_decay(
            st.session_state.legs_data, 
            spot_price, 
            spot_drift, 
            vol_shock, 
            price_decay_rate,
            None
        )
        
        if not scenario_df.empty:
            # Display data date above the table
            if data_date and data_date != "N/A":
                st.subheader(f"📅 Data Date: {data_date}")
            
            # Configure AG Grid
            gb = GridOptionsBuilder.from_dataframe(scenario_df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(editable=False, sortable=True, resizable=True)
            
            # Style columns
            gb.configure_column('Time', width=120)
            gb.configure_column('Strike', width=100)
            gb.configure_column('Type', width=80)
            gb.configure_column('Current_Price', width=120)
            gb.configure_column('Current_IV', width=100)
            for minutes in TIME_INTERVALS:
                col_name = f"{minutes}min"
                if col_name in scenario_df.columns:
                    gb.configure_column(col_name, width=140)
            
            grid_options = gb.build()
            
            # Display the grid
            grid_response = AgGrid(
                scenario_df,
                gridOptions=grid_options,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.NO_UPDATE,
                fit_columns_on_grid_load=True,
                height=400,
                theme='streamlit'
            )
            
            # Portfolio Summary
            st.subheader("📊 Portfolio Summary")
            
            total_current_value = sum([leg['current_price'] for leg in st.session_state.legs_data if leg['current_price']])
            st.metric("Total Current Value", f"₹{total_current_value:.2f}")
            
            # Calculate portfolio metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Days to expiry range
                days_to_expiries = [(leg['expiry'] - leg['timestamp'].date()).days for leg in st.session_state.legs_data]
                min_days, max_days = min(days_to_expiries), max(days_to_expiries)
                st.info(f"⏰ **Days to Expiry**\nRange: {min_days}-{max_days} days")
            
            with col2:
                # Decay acceleration summary
                acceleration_counts = {"4x": 0, "3x": 0, "2x": 0, "1.5x": 0, "1x": 0}
                for days in days_to_expiries:
                    if days <= 7:
                        acceleration_counts["4x"] += 1
                    elif days <= 15:
                        acceleration_counts["3x"] += 1
                    elif days <= 30:
                        acceleration_counts["2x"] += 1
                    elif days <= 60:
                        acceleration_counts["1.5x"] += 1
                    else:
                        acceleration_counts["1x"] += 1
                
                active_accelerations = [f"{count} legs @ {acc}" for acc, count in acceleration_counts.items() if count > 0]
                st.info(f"⚡ **Decay Acceleration**\n" + "\n".join(active_accelerations))
            
            with col3:
                # Linear decay explanation
                st.info(f"📉 **Linear Decay Model**\nBase rate: {price_decay_rate}%/year\nFormula: New = Old - (Rate × Time)")
            
        else:
            st.warning("Unable to generate scenario table. Please check your leg data.")
    
    else:
        st.info("👈 Add option legs above to see forward price projections with linear decay")
        
        # Show sample table structure
        st.subheader("Sample Table Structure (Linear Decay)")
        st.markdown("**📅 Data Date: 22-Sep-2025**")
        sample_data = pd.DataFrame({
            'Leg': [1, 2],
            'Time': ['01:49:00 PM', '01:49:00 PM'],
            'Strike': [22500, 22600],
            'Type': ['CE', 'PE'],
            'Current_Price': ['₹45.50', '₹38.25'],
            'Current_IV': ['0.1245', '0.1198'],
            '5min': ['₹45.48 (-0.02, -0.1%)', '₹38.23 (-0.02, -0.1%)'],
            '10min': ['₹45.46 (-0.04, -0.1%)', '₹38.21 (-0.04, -0.1%)'],
            '30min': ['₹45.38 (-0.12, -0.3%)', '₹38.13 (-0.12, -0.3%)'],
            '60min': ['₹45.26 (-0.24, -0.5%)', '₹38.01 (-0.24, -0.6%)'],
            '375min': ['₹44.75 (-0.75, -1.6%)', '₹37.50 (-0.75, -2.0%)']
        })
        
        st.dataframe(sample_data, width='stretch')
    
    # Footer information
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info("**Linear Decay:** Direct price reduction over time (not exponential)")
    
    with col_info2:
        st.info("**PnL Format:** New_Price (Absolute_Change, Percentage_Change)")
    
    with col_info3:
        st.info("**Auto-refresh:** Market data updates every minute")

if __name__ == "__main__":
    main()
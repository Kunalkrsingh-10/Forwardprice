
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton
import logging

logger = logging.getLogger('OptionUtils')

def bs_price(S, K, T, sigma, r=0.0, q=0.0, option_type="call"):
    """
    Black-Scholes option pricing formula.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        sigma: Volatility
        r: Risk-free rate
        q: Dividend yield
        option_type: 'call' or 'put'
    """
    if T <= 0:
        # At expiry, return intrinsic value
        if option_type.lower().startswith('c'):
            return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
    
    if sigma <= 0:
        # Zero volatility case
        forward = S * math.exp((r - q) * T)
        if option_type.lower().startswith('c'):
            return max(0.0, forward - K) * math.exp(-r * T)
        else:
            return max(0.0, K - forward) * math.exp(-r * T)
    
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type.lower().startswith('c'):
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        
        return max(price, 0.0)  # Ensure non-negative
        
    except Exception as e:
        logger.debug(f"BS price calculation failed for S={S}, K={K}, T={T}, sigma={sigma}: {e}")
        return 0.0

def black_scholes_iv(S, K, T, price, r=0.0, q=0.0, option_type="call", max_iterations=100):
    """
    Calculate Black-Scholes implied volatility using Brent's method with Newton-Raphson fallback.
    Enhanced with better bounds and validation.
    
    Args:
        S: Spot price
        K: Strike price  
        T: Time to expiry (years)
        price: Market price of option
        r: Risk-free rate
        q: Dividend yield
        option_type: 'call' or 'put'
        max_iterations: Maximum iterations for root finding
    """
    if price is None or np.isnan(price) or price <= 0:
        logger.debug(f"Invalid price for K={K}, type={option_type}: price={price}")
        return np.nan
    
    if T <= 0 or S <= 0 or K <= 0:
        logger.debug(f"Invalid parameters for K={K}, type={option_type}: S={S}, T={T}, K={K}")
        return np.nan
    
    # Calculate intrinsic value for validation
    if option_type.lower().startswith('c'):
        intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    else:
        intrinsic = max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
    
    # Enhanced intrinsic value check
    if price <= intrinsic + 0.005:  # Price must exceed intrinsic by at least 0.5 paisa
        logger.debug(f"Price {price} too close to intrinsic {intrinsic} for K={K}, type={option_type}")
        return np.nan
    
    # Enhanced bounds based on moneyness and time to expiry
    forward = S * math.exp((r - q) * T)
    moneyness = K / forward
    
    # Adaptive bounds based on option characteristics
    if T < 7/365:  # Less than 7 days
        vol_low, vol_high = 1e-6, 8.0  # Higher upper bound for short-dated
    elif T < 30/365:  # Less than 30 days
        vol_low, vol_high = 1e-6, 5.0
    else:  # Longer-dated
        vol_low, vol_high = 1e-6, 3.0
    
    # Adjust bounds for deep ITM/OTM options
    if moneyness < 0.8 or moneyness > 1.2:  # Deep ITM/OTM
        vol_high = min(vol_high * 1.5, 10.0)
    
    def objective(vol):
        try:
            model_price = bs_price(S, K, T, vol, r, q, option_type)
            return model_price - price
        except Exception:
            logger.debug(f"Objective function failed for K={K}, type={option_type}, vol={vol}")
            return 1e10
    
    try:
        # Check bounds
        f_low = objective(vol_low)
        f_high = objective(vol_high)
        
        # If bounds don't bracket the root, try wider bounds
        attempts = 0
        while f_low * f_high > 0 and attempts < 3:
            vol_high = min(vol_high * 2, 15.0)
            f_high = objective(vol_high)
            attempts += 1
            
        if f_low * f_high > 0:
            logger.debug(f"Brentq bounds failed for K={K}, type={option_type}: f_low={f_low}, f_high={f_high}")
            return _fallback_iv_calculation(S, K, T, price, r, q, option_type)
        
        # Use Brent's method
        iv = brentq(objective, vol_low, vol_high, maxiter=max_iterations, xtol=1e-8)
        
        # Enhanced validation
        if 0.005 < iv < 10.0:  # 0.5% to 1000% IV range
            return float(iv)
        else:
            logger.debug(f"IV out of range for K={K}, type={option_type}: iv={iv}")
            return np.nan
            
    except Exception as e:
        logger.debug(f"Brentq IV calculation failed for K={K}, type={option_type}: {e}")
        return _fallback_iv_calculation(S, K, T, price, r, q, option_type)

def _fallback_iv_calculation(S, K, T, price, r, q, option_type, max_iter=50):
    """
    Enhanced fallback IV calculation with better initial guess.
    """
    def objective(vol):
        try:
            return bs_price(S, K, T, vol, r, q, option_type) - price
        except:
            return 1e10
    
    # Better initial guess based on Black-Scholes approximation
    forward = S * math.exp((r - q) * T)
    moneyness = K / forward
    
    # Brenner-Subrahmanyam approximation for initial guess
    if abs(moneyness - 1.0) < 0.1:  # Near ATM
        # For ATM options: IV ≈ price * sqrt(2π / T) / S
        initial_guess = max(0.1, min(1.0, price * math.sqrt(2 * math.pi / T) / S))
    else:
        # For ITM/OTM options: use empirical scaling
        if moneyness < 1.0:  # ITM calls / OTM puts
            initial_guess = 0.3
        else:  # OTM calls / ITM puts
            initial_guess = 0.4
    
    try:
        # Try Newton-Raphson with better initial guess
        iv = newton(objective, initial_guess, maxiter=max_iter, tol=1e-6)
        if 0.005 < iv < 10.0:
            logger.debug(f"Newton-Raphson succeeded for K={K}, type={option_type}: iv={iv}")
            return float(iv)
    except Exception as e:
        logger.debug(f"Newton-Raphson failed for K={K}, type={option_type}: {e}")
    
    # Enhanced bisection method
    try:
        vol_low, vol_high = 1e-6, 10.0
        
        # Ensure bounds bracket the solution
        if objective(vol_low) * objective(vol_high) > 0:
            return np.nan
            
        for i in range(max_iter):
            vol_mid = 0.5 * (vol_low + vol_high)
            obj_mid = objective(vol_mid)
            
            if abs(obj_mid) < 1e-6:
                logger.debug(f"Bisection succeeded for K={K}, type={option_type}: iv={vol_mid}")
                return float(vol_mid)
            
            if objective(vol_low) * obj_mid < 0:
                vol_high = vol_mid
            else:
                vol_low = vol_mid
                
            if vol_high - vol_low < 1e-8:
                result = 0.5 * (vol_low + vol_high)
                logger.debug(f"Bisection converged for K={K}, type={option_type}: iv={result}")
                return float(result)
        
        logger.debug(f"Bisection failed to converge for K={K}, type={option_type}")
        return np.nan
        
    except Exception as e:
        logger.debug(f"Bisection fallback failed for K={K}, type={option_type}: {e}")
        return np.nan

def bs_delta(S, K, T, sigma, r=0.0, q=0.0, option_type="call"):
    """Calculate Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return np.nan
    
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        
        if option_type.lower().startswith('c'):
            return math.exp(-q * T) * norm.cdf(d1)
        else:
            return -math.exp(-q * T) * norm.cdf(-d1)
            
    except Exception:
        logger.debug(f"Delta calculation failed for S={S}, K={K}, T={T}, sigma={sigma}")
        return np.nan

def bs_gamma(S, K, T, sigma, r=0.0, q=0.0):
    """Calculate Black-Scholes gamma."""
    if T <= 0 or sigma <= 0:
        return np.nan
    
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    except Exception:
        logger.debug(f"Gamma calculation failed for S={S}, K={K}, T={T}, sigma={sigma}")
        return np.nan

def bs_theta(S, K, T, sigma, r=0.0, q=0.0, option_type="call"):
    """Calculate Black-Scholes theta (per day)."""
    if T <= 0 or sigma <= 0:
        return np.nan
    
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        term1 = -math.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        
        if option_type.lower().startswith('c'):
            term2 = q * S * math.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * math.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        
        theta_annual = term1 + term2 + term3
        return theta_annual / 365.0  # Convert to per day
        
    except Exception:
        logger.debug(f"Theta calculation failed for S={S}, K={K}, T={T}, sigma={sigma}")
        return np.nan 
    
def bs_vega(S, K, T, sigma, r=0.0, q=0.0):
    """Calculate Black-Scholes vega (per 1% vol change)."""
    if T <= 0 or sigma <= 0:
        return np.nan
    
    try:
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        return vega / 100.0  # Per 1% change in volatility
    except Exception:
        logger.debug(f"Vega calculation failed for S={S}, K={K}, T={T}, sigma={sigma}")
        return np.nan

def calculate_intrinsic_value(S, K, option_type, r=0.0, q=0.0, T=0.0):
    """
    Calculate intrinsic value of an option
    """
    if option_type.lower().startswith('c'):
        return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    else:
        return max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))

def validate_option_price(price, intrinsic, min_time_value=0.005):
    """
    Validate if option price is reasonable compared to intrinsic value
    """
    if price is None or np.isnan(price) or price <= 0:
        return False
    
    if price <= intrinsic + min_time_value:
        return False
    
    return True
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def calculate_portfolio_var(positions, returns_data, confidence_level=0.05, method='historical'):
    """
    Calculate portfolio Value at Risk
    
    Parameters:
    positions: dict, position sizes by asset
    returns_data: pandas DataFrame, historical returns
    confidence_level: float, confidence level (e.g., 0.05 for 95% VaR)
    method: str, calculation method ('historical', 'parametric', 'monte_carlo')
    
    Returns:
    float, VaR amount
    """
    if not positions or returns_data.empty:
        return np.nan
    
    # Align positions with returns data
    common_assets = set(positions.keys()).intersection(set(returns_data.columns))
    if not common_assets:
        return np.nan
    
    position_values = np.array([positions[asset] for asset in common_assets])
    asset_returns = returns_data[list(common_assets)].dropna()
    
    if len(asset_returns) == 0:
        return np.nan
    
    # Calculate portfolio returns
    portfolio_returns = (asset_returns * position_values).sum(axis=1)
    
    if method == 'historical':
        var = -np.percentile(portfolio_returns, confidence_level * 100)
    
    elif method == 'parametric':
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        var = -(mean_return + stats.norm.ppf(confidence_level) * std_return)
    
    elif method == 'monte_carlo':
        # Monte Carlo simulation
        n_simulations = 10000
        cov_matrix = asset_returns.cov()
        mean_returns = asset_returns.mean()
        
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        simulated_portfolio_returns = (simulated_returns * position_values).sum(axis=1)
        var = -np.percentile(simulated_portfolio_returns, confidence_level * 100)
    
    else:
        raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
    
    return max(0, var)  # VaR should be positive

def calculate_component_var(positions, returns_data, confidence_level=0.05):
    """
    Calculate component VaR for each position
    
    Parameters:
    positions: dict, position sizes by asset
    returns_data: pandas DataFrame, historical returns
    confidence_level: float, confidence level
    
    Returns:
    dict, component VaR by asset
    """
    if not positions or returns_data.empty:
        return {}
    
    common_assets = set(positions.keys()).intersection(set(returns_data.columns))
    if not common_assets:
        return {}
    
    position_values = np.array([positions[asset] for asset in common_assets])
    asset_returns = returns_data[list(common_assets)].dropna()
    
    if len(asset_returns) == 0:
        return {}
    
    # Calculate portfolio VaR
    portfolio_var = calculate_portfolio_var(positions, returns_data, confidence_level)
    
    if np.isnan(portfolio_var) or portfolio_var == 0:
        return {asset: 0.0 for asset in common_assets}
    
    # Calculate marginal VaR for each asset
    component_vars = {}
    small_change = 0.001  # 0.1% change for numerical differentiation
    
    for i, asset in enumerate(common_assets):
        # Create modified positions
        modified_positions = positions.copy()
        modified_positions[asset] *= (1 + small_change)
        
        # Calculate new portfolio VaR
        new_var = calculate_portfolio_var(modified_positions, returns_data, confidence_level)
        
        if not np.isnan(new_var):
            marginal_var = (new_var - portfolio_var) / (positions[asset] * small_change)
            component_vars[asset] = positions[asset] * marginal_var
        else:
            component_vars[asset] = 0.0
    
    return component_vars

def calculate_expected_shortfall(returns, confidence_level=0.05):
    """
    Calculate Expected Shortfall (Conditional VaR)
    
    Parameters:
    returns: pandas Series or array-like, return data
    confidence_level: float, confidence level
    
    Returns:
    float, Expected Shortfall
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    
    returns = np.array(returns)
    
    if len(returns) == 0:
        return np.nan
    
    var_threshold = -np.percentile(returns, confidence_level * 100)
    tail_losses = returns[returns <= -var_threshold]
    
    if len(tail_losses) == 0:
        return var_threshold
    
    return -tail_losses.mean()

def calculate_risk_budgets(positions, returns_data, target_risk=None):
    """
    Calculate risk budgets for portfolio positions
    
    Parameters:
    positions: dict, position sizes by asset
    returns_data: pandas DataFrame, historical returns
    target_risk: float, target portfolio risk level
    
    Returns:
    dict, risk budget analysis
    """
    if not positions or returns_data.empty:
        return {}
    
    common_assets = set(positions.keys()).intersection(set(returns_data.columns))
    if not common_assets:
        return {}
    
    asset_returns = returns_data[list(common_assets)].dropna()
    
    if len(asset_returns) == 0:
        return {}
    
    # Calculate covariance matrix
    cov_matrix = asset_returns.cov() * 252  # Annualized
    
    # Position weights
    total_value = sum(abs(positions[asset]) for asset in common_assets)
    weights = np.array([positions[asset] / total_value for asset in common_assets])
    
    # Portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Marginal risk contributions
    marginal_contributions = np.dot(cov_matrix.values, weights) / portfolio_volatility
    
    # Risk contributions
    risk_contributions = weights * marginal_contributions
    risk_contribution_pct = risk_contributions / risk_contributions.sum() * 100
    
    risk_budget = {}
    for i, asset in enumerate(common_assets):
        risk_budget[asset] = {
            'weight': weights[i] * 100,
            'risk_contribution': risk_contributions[i],
            'risk_contribution_pct': risk_contribution_pct[i],
            'marginal_risk': marginal_contributions[i]
        }
    
    return {
        'risk_budgets': risk_budget,
        'portfolio_volatility': portfolio_volatility * 100,
        'total_risk_contribution': risk_contributions.sum()
    }

def calculate_stress_test_scenarios(positions, returns_data, scenarios=None):
    """
    Calculate stress test scenarios for portfolio
    
    Parameters:
    positions: dict, position sizes by asset
    returns_data: pandas DataFrame, historical returns
    scenarios: dict, custom stress scenarios
    
    Returns:
    dict, stress test results
    """
    if not positions or returns_data.empty:
        return {}
    
    common_assets = set(positions.keys()).intersection(set(returns_data.columns))
    if not common_assets:
        return {}
    
    asset_returns = returns_data[list(common_assets)].dropna()
    
    if len(asset_returns) == 0:
        return {}
    
    # Default stress scenarios if not provided
    if scenarios is None:
        scenarios = {
            'Market Crash (-20%)': -0.20,
            'Volatility Spike (+50%)': 'volatility_spike',
            'Correlation Breakdown': 'correlation_breakdown',
            '2008 Financial Crisis': 'historical_2008',
            'COVID-19 Pandemic': 'historical_2020'
        }
    
    stress_results = {}
    position_values = np.array([positions[asset] for asset in common_assets])
    
    for scenario_name, scenario_spec in scenarios.items():
        if isinstance(scenario_spec, (int, float)):
            # Simple market shock
            scenario_returns = np.full(len(common_assets), scenario_spec)
            scenario_pnl = np.sum(position_values * scenario_returns)
            
        elif scenario_spec == 'volatility_spike':
            # Increase volatility by 50%
            historical_vol = asset_returns.std()
            shocked_returns = np.random.normal(0, historical_vol * 1.5, len(common_assets))
            scenario_pnl = np.sum(position_values * shocked_returns)
            
        elif scenario_spec == 'correlation_breakdown':
            # Set all correlations to 0.95
            corr_matrix = asset_returns.corr()
            np.fill_diagonal(corr_matrix.values, 1.0)
            corr_matrix.values[corr_matrix.values != 1.0] = 0.95
            
            # Generate correlated shocks
            std_devs = asset_returns.std()
            shocked_returns = np.random.multivariate_normal(
                np.zeros(len(common_assets)),
                corr_matrix.values * np.outer(std_devs, std_devs)
            )
            scenario_pnl = np.sum(position_values * shocked_returns)
            
        elif scenario_spec.startswith('historical_'):
            # Use historical worst case scenarios
            if scenario_spec == 'historical_2008':
                # Find worst 22-day period (approximate month) in data
                rolling_returns = asset_returns.rolling(22).sum()
                worst_period_idx = rolling_returns.sum(axis=1).idxmin()
                if not pd.isna(worst_period_idx):
                    scenario_returns = rolling_returns.loc[worst_period_idx].values
                    scenario_pnl = np.sum(position_values * scenario_returns)
                else:
                    scenario_pnl = 0
            else:
                scenario_pnl = 0
        else:
            scenario_pnl = 0
        
        stress_results[scenario_name] = {
            'pnl_impact': scenario_pnl,
            'pnl_percentage': scenario_pnl / sum(abs(v) for v in position_values) * 100
        }
    
    return stress_results

def calculate_correlation_risk(returns_data, lookback_periods=[30, 90, 252]):
    """
    Calculate correlation risk metrics
    
    Parameters:
    returns_data: pandas DataFrame, historical returns
    lookback_periods: list, different lookback periods for correlation calculation
    
    Returns:
    dict, correlation risk analysis
    """
    if returns_data.empty or len(returns_data.columns) < 2:
        return {}
    
    correlation_analysis = {}
    
    for period in lookback_periods:
        if len(returns_data) >= period:
            # Rolling correlation
            period_returns = returns_data.tail(period)
            corr_matrix = period_returns.corr()
            
            # Extract upper triangle correlations (excluding diagonal)
            upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix.values[upper_tri_indices]
            correlations = correlations[~np.isnan(correlations)]
            
            if len(correlations) > 0:
                correlation_analysis[f'{period}D'] = {
                    'mean_correlation': np.mean(correlations),
                    'max_correlation': np.max(correlations),
                    'min_correlation': np.min(correlations),
                    'correlation_std': np.std(correlations),
                    'correlations_above_50pct': np.sum(correlations > 0.5) / len(correlations) * 100,
                    'correlations_above_80pct': np.sum(correlations > 0.8) / len(correlations) * 100
                }
    
    return correlation_analysis

def calculate_liquidity_risk(positions, volume_data, price_data):
    """
    Calculate liquidity risk metrics
    
    Parameters:
    positions: dict, position sizes by asset
    volume_data: pandas DataFrame, trading volume data
    price_data: pandas DataFrame, price data
    
    Returns:
    dict, liquidity risk analysis
    """
    if not positions or volume_data.empty or price_data.empty:
        return {}
    
    common_assets = set(positions.keys()).intersection(set(volume_data.columns))
    common_assets = common_assets.intersection(set(price_data.columns))
    
    if not common_assets:
        return {}
    
    liquidity_metrics = {}
    
    for asset in common_assets:
        position_size = abs(positions[asset])
        
        if asset in volume_data.columns and asset in price_data.columns:
            recent_volume = volume_data[asset].tail(20).mean()  # 20-day average
            recent_price = price_data[asset].tail(1).iloc[0]
            
            # Calculate liquidity metrics
            daily_dollar_volume = recent_volume * recent_price
            position_as_pct_of_volume = (position_size / daily_dollar_volume) * 100
            
            # Estimate days to liquidate (assuming 10% of daily volume)
            days_to_liquidate = position_size / (daily_dollar_volume * 0.1)
            
            # Calculate bid-ask spread estimate (simplified)
            price_volatility = price_data[asset].tail(20).pct_change().std()
            estimated_spread = price_volatility * 2  # Rough estimate
            
            liquidity_metrics[asset] = {
                'daily_dollar_volume': daily_dollar_volume,
                'position_vs_volume_pct': position_as_pct_of_volume,
                'days_to_liquidate': days_to_liquidate,
                'estimated_spread_pct': estimated_spread * 100,
                'liquidity_score': min(100, max(0, 100 - position_as_pct_of_volume * 10))
            }
    
    # Overall portfolio liquidity score
    if liquidity_metrics:
        position_weights = {asset: abs(positions[asset]) for asset in liquidity_metrics.keys()}
        total_position = sum(position_weights.values())
        
        weighted_liquidity_score = sum(
            metrics['liquidity_score'] * position_weights[asset] / total_position
            for asset, metrics in liquidity_metrics.items()
        )
        
        max_days_to_liquidate = max(
            metrics['days_to_liquidate'] for metrics in liquidity_metrics.values()
        )
        
        portfolio_liquidity = {
            'weighted_liquidity_score': weighted_liquidity_score,
            'max_days_to_liquidate': max_days_to_liquidate,
            'asset_liquidity': liquidity_metrics
        }
    else:
        portfolio_liquidity = {}
    
    return portfolio_liquidity

def calculate_tail_risk_metrics(returns_data, confidence_levels=[0.95, 0.99, 0.995]):
    """
    Calculate tail risk metrics including VaR, ES, and extreme value statistics
    
    Parameters:
    returns_data: pandas Series or DataFrame, return data
    confidence_levels: list, confidence levels for calculations
    
    Returns:
    dict, tail risk metrics
    """
    if isinstance(returns_data, pd.DataFrame):
        returns_data = returns_data.sum(axis=1)  # Portfolio returns
    
    returns_data = returns_data.dropna()
    
    if len(returns_data) == 0:
        return {}
    
    tail_metrics = {}
    
    for conf_level in confidence_levels:
        var = calculate_portfolio_var({'portfolio': 1.0}, 
                                    returns_data.to_frame('portfolio'), 
                                    confidence_level=1-conf_level)
        
        es = calculate_expected_shortfall(returns_data, confidence_level=1-conf_level)
        
        tail_metrics[f'{conf_level*100:.1f}%'] = {
            'var': var,
            'expected_shortfall': es,
            'es_var_ratio': es / var if var != 0 else np.nan
        }
    
    # Additional tail statistics
    returns_array = returns_data.values
    tail_metrics['statistics'] = {
        'skewness': stats.skew(returns_array),
        'excess_kurtosis': stats.kurtosis(returns_array),
        'jarque_bera_pvalue': stats.jarque_bera(returns_array)[1],
        'min_return': np.min(returns_array),
        'max_return': np.max(returns_array),
        'tail_ratio': abs(np.percentile(returns_array, 5)) / np.percentile(returns_array, 95)
    }
    
    return tail_metrics

def calculate_regime_risk(returns_data, regimes=['low_vol', 'high_vol', 'crisis']):
    """
    Calculate risk metrics under different market regimes
    
    Parameters:
    returns_data: pandas DataFrame, historical returns
    regimes: list, market regimes to analyze
    
    Returns:
    dict, regime-based risk analysis
    """
    if returns_data.empty:
        return {}
    
    # Calculate rolling volatility to identify regimes
    portfolio_returns = returns_data.sum(axis=1) if len(returns_data.columns) > 1 else returns_data.iloc[:, 0]
    rolling_vol = portfolio_returns.rolling(20).std() * np.sqrt(252)
    
    # Define regime thresholds
    vol_25th = rolling_vol.quantile(0.25)
    vol_75th = rolling_vol.quantile(0.75)
    vol_95th = rolling_vol.quantile(0.95)
    
    regime_data = {}
    
    # Low volatility regime
    low_vol_mask = rolling_vol <= vol_25th
    if low_vol_mask.sum() > 0:
        low_vol_returns = portfolio_returns[low_vol_mask]
        regime_data['low_vol'] = {
            'mean_return': low_vol_returns.mean() * 252,
            'volatility': low_vol_returns.std() * np.sqrt(252),
            'var_95': -np.percentile(low_vol_returns, 5),
            'max_drawdown': calculate_max_drawdown(low_vol_returns),
            'frequency': low_vol_mask.sum() / len(rolling_vol) * 100
        }
    
    # High volatility regime
    high_vol_mask = (rolling_vol > vol_75th) & (rolling_vol <= vol_95th)
    if high_vol_mask.sum() > 0:
        high_vol_returns = portfolio_returns[high_vol_mask]
        regime_data['high_vol'] = {
            'mean_return': high_vol_returns.mean() * 252,
            'volatility': high_vol_returns.std() * np.sqrt(252),
            'var_95': -np.percentile(high_vol_returns, 5),
            'max_drawdown': calculate_max_drawdown(high_vol_returns),
            'frequency': high_vol_mask.sum() / len(rolling_vol) * 100
        }
    
    # Crisis regime
    crisis_mask = rolling_vol > vol_95th
    if crisis_mask.sum() > 0:
        crisis_returns = portfolio_returns[crisis_mask]
        regime_data['crisis'] = {
            'mean_return': crisis_returns.mean() * 252,
            'volatility': crisis_returns.std() * np.sqrt(252),
            'var_95': -np.percentile(crisis_returns, 5),
            'max_drawdown': calculate_max_drawdown(crisis_returns),
            'frequency': crisis_mask.sum() / len(rolling_vol) * 100
        }
    
    return regime_data

def calculate_max_drawdown(returns):
    """
    Helper function to calculate maximum drawdown from returns
    
    Parameters:
    returns: pandas Series, return data
    
    Returns:
    float, maximum drawdown
    """
    if len(returns) == 0:
        return np.nan
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def calculate_returns(prices, periods=1):
    """
    Calculate returns for given price series
    
    Parameters:
    prices: pandas Series of prices
    periods: int, number of periods for return calculation
    
    Returns:
    pandas Series of returns
    """
    if isinstance(prices, pd.Series):
        return prices.pct_change(periods=periods)
    elif isinstance(prices, (list, np.ndarray)):
        prices = pd.Series(prices)
        return prices.pct_change(periods=periods)
    else:
        raise ValueError("Prices must be pandas Series, list, or numpy array")

def calculate_volatility(returns, annualize=True, trading_days=252):
    """
    Calculate volatility from returns
    
    Parameters:
    returns: pandas Series of returns
    annualize: bool, whether to annualize the volatility
    trading_days: int, number of trading days per year
    
    Returns:
    float, volatility
    """
    if len(returns.dropna()) == 0:
        return np.nan
    
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(trading_days)
    
    return vol

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, annualize=True, trading_days=252):
    """
    Calculate Sharpe ratio
    
    Parameters:
    returns: pandas Series of returns
    risk_free_rate: float, annual risk-free rate
    annualize: bool, whether to annualize the ratio
    trading_days: int, number of trading days per year
    
    Returns:
    float, Sharpe ratio
    """
    if len(returns.dropna()) == 0:
        return np.nan
    
    excess_returns = returns - (risk_free_rate / trading_days if annualize else risk_free_rate)
    
    if excess_returns.std() == 0:
        return np.nan
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    if annualize:
        sharpe *= np.sqrt(trading_days)
    
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.02, annualize=True, trading_days=252):
    """
    Calculate Sortino ratio (downside deviation)
    
    Parameters:
    returns: pandas Series of returns
    risk_free_rate: float, annual risk-free rate
    annualize: bool, whether to annualize the ratio
    trading_days: int, number of trading days per year
    
    Returns:
    float, Sortino ratio
    """
    if len(returns.dropna()) == 0:
        return np.nan
    
    excess_returns = returns - (risk_free_rate / trading_days if annualize else risk_free_rate)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = downside_returns.std()
    
    if downside_deviation == 0:
        return np.nan
    
    sortino = excess_returns.mean() / downside_deviation
    
    if annualize:
        sortino *= np.sqrt(trading_days)
    
    return sortino

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown
    
    Parameters:
    prices: pandas Series of prices
    
    Returns:
    float, maximum drawdown as percentage
    """
    if len(prices) == 0:
        return np.nan
    
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()

def calculate_calmar_ratio(returns, max_drawdown=None):
    """
    Calculate Calmar ratio (annual return / max drawdown)
    
    Parameters:
    returns: pandas Series of returns
    max_drawdown: float, optional maximum drawdown
    
    Returns:
    float, Calmar ratio
    """
    if len(returns.dropna()) == 0:
        return np.nan
    
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    
    if max_drawdown is None:
        max_drawdown = abs(calculate_max_drawdown(returns))
    else:
        max_drawdown = abs(max_drawdown)
    
    if max_drawdown == 0:
        return np.inf
    
    return annual_return / max_drawdown

def calculate_beta(returns, market_returns):
    """
    Calculate beta relative to market
    
    Parameters:
    returns: pandas Series of asset returns
    market_returns: pandas Series of market returns
    
    Returns:
    float, beta
    """
    if len(returns.dropna()) == 0 or len(market_returns.dropna()) == 0:
        return np.nan
    
    # Align the series
    aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
    
    if len(aligned_data) < 2:
        return np.nan
    
    asset_returns = aligned_data.iloc[:, 0]
    market_returns = aligned_data.iloc[:, 1]
    
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return np.nan
    
    return covariance / market_variance

def calculate_alpha(returns, market_returns, risk_free_rate=0.02):
    """
    Calculate alpha (Jensen's alpha)
    
    Parameters:
    returns: pandas Series of asset returns
    market_returns: pandas Series of market returns
    risk_free_rate: float, annual risk-free rate
    
    Returns:
    float, alpha
    """
    beta = calculate_beta(returns, market_returns)
    
    if np.isnan(beta):
        return np.nan
    
    asset_return = returns.mean() * 252
    market_return = market_returns.mean() * 252
    
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    alpha = asset_return - expected_return
    
    return alpha

def calculate_information_ratio(returns, benchmark_returns):
    """
    Calculate information ratio
    
    Parameters:
    returns: pandas Series of portfolio returns
    benchmark_returns: pandas Series of benchmark returns
    
    Returns:
    float, information ratio
    """
    active_returns = returns - benchmark_returns
    active_returns = active_returns.dropna()
    
    if len(active_returns) == 0:
        return np.nan
    
    tracking_error = active_returns.std() * np.sqrt(252)
    
    if tracking_error == 0:
        return np.nan
    
    active_return = active_returns.mean() * 252
    
    return active_return / tracking_error

def calculate_tracking_error(returns, benchmark_returns):
    """
    Calculate tracking error
    
    Parameters:
    returns: pandas Series of portfolio returns
    benchmark_returns: pandas Series of benchmark returns
    
    Returns:
    float, annualized tracking error
    """
    active_returns = returns - benchmark_returns
    active_returns = active_returns.dropna()
    
    if len(active_returns) == 0:
        return np.nan
    
    return active_returns.std() * np.sqrt(252)

def calculate_treynor_ratio(returns, market_returns, risk_free_rate=0.02):
    """
    Calculate Treynor ratio
    
    Parameters:
    returns: pandas Series of returns
    market_returns: pandas Series of market returns
    risk_free_rate: float, annual risk-free rate
    
    Returns:
    float, Treynor ratio
    """
    beta = calculate_beta(returns, market_returns)
    
    if np.isnan(beta) or beta == 0:
        return np.nan
    
    excess_return = returns.mean() * 252 - risk_free_rate
    
    return excess_return / beta

def calculate_omega_ratio(returns, threshold=0):
    """
    Calculate Omega ratio
    
    Parameters:
    returns: pandas Series of returns
    threshold: float, threshold return
    
    Returns:
    float, Omega ratio
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return np.nan
    
    excess_returns = returns - threshold
    gains = excess_returns[excess_returns > 0].sum()
    losses = abs(excess_returns[excess_returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    return gains / losses

def calculate_var(returns, confidence_level=0.05):
    """
    Calculate Value at Risk using historical simulation
    
    Parameters:
    returns: pandas Series of returns
    confidence_level: float, confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
    float, VaR as a positive number
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return np.nan
    
    return -np.percentile(returns, confidence_level * 100)

def calculate_cvar(returns, confidence_level=0.05):
    """
    Calculate Conditional Value at Risk (Expected Shortfall)
    
    Parameters:
    returns: pandas Series of returns
    confidence_level: float, confidence level
    
    Returns:
    float, CVaR as a positive number
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return np.nan
    
    var_threshold = -np.percentile(returns, confidence_level * 100)
    tail_losses = returns[returns <= -var_threshold]
    
    if len(tail_losses) == 0:
        return var_threshold
    
    return -tail_losses.mean()

def calculate_portfolio_metrics(weights, returns, cov_matrix=None):
    """
    Calculate portfolio metrics given weights and returns
    
    Parameters:
    weights: array-like, portfolio weights
    returns: pandas DataFrame of asset returns
    cov_matrix: pandas DataFrame, covariance matrix (optional)
    
    Returns:
    dict, portfolio metrics
    """
    weights = np.array(weights)
    
    if cov_matrix is None:
        cov_matrix = returns.cov() * 252
    
    # Portfolio return
    portfolio_return = np.sum(weights * returns.mean() * 252)
    
    # Portfolio volatility
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'weights': weights
    }

def optimize_portfolio(returns, objective='sharpe', constraints=None):
    """
    Optimize portfolio weights
    
    Parameters:
    returns: pandas DataFrame of asset returns
    objective: str, optimization objective ('sharpe', 'min_vol', 'max_return')
    constraints: dict, additional constraints
    
    Returns:
    dict, optimization results
    """
    n_assets = len(returns.columns)
    
    # Constraints
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
    
    # Bounds (no short selling by default)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Add custom constraints
    if constraints:
        if 'max_weight' in constraints:
            bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
        if 'min_weight' in constraints:
            bounds = tuple((constraints['min_weight'], bounds[i][1]) for i in range(n_assets))
    
    # Objective functions
    def negative_sharpe(weights):
        metrics = calculate_portfolio_metrics(weights, returns)
        return -metrics['sharpe_ratio']
    
    def portfolio_volatility(weights):
        metrics = calculate_portfolio_metrics(weights, returns)
        return metrics['volatility']
    
    def negative_return(weights):
        metrics = calculate_portfolio_metrics(weights, returns)
        return -metrics['expected_return']
    
    # Select objective function
    if objective == 'sharpe':
        objective_func = negative_sharpe
    elif objective == 'min_vol':
        objective_func = portfolio_volatility
    elif objective == 'max_return':
        objective_func = negative_return
    else:
        raise ValueError("Objective must be 'sharpe', 'min_vol', or 'max_return'")
    
    # Initial guess (equal weights)
    x0 = np.array([1/n_assets] * n_assets)
    
    # Optimize
    try:
        result = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            optimal_weights = result.x
            optimal_metrics = calculate_portfolio_metrics(optimal_weights, returns)
            
            return {
                'success': True,
                'weights': optimal_weights,
                'metrics': optimal_metrics,
                'message': 'Optimization successful'
            }
        else:
            return {
                'success': False,
                'weights': x0,
                'metrics': calculate_portfolio_metrics(x0, returns),
                'message': f'Optimization failed: {result.message}'
            }
    
    except Exception as e:
        return {
            'success': False,
            'weights': x0,
            'metrics': calculate_portfolio_metrics(x0, returns),
            'message': f'Optimization error: {str(e)}'
        }

def calculate_rolling_metrics(returns, window=252):
    """
    Calculate rolling performance metrics
    
    Parameters:
    returns: pandas Series of returns
    window: int, rolling window size
    
    Returns:
    pandas DataFrame of rolling metrics
    """
    rolling_metrics = pd.DataFrame(index=returns.index)
    
    rolling_metrics['return'] = returns.rolling(window).mean() * 252
    rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
    rolling_metrics['sharpe'] = rolling_metrics['return'] / rolling_metrics['volatility']
    
    # Rolling max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window).max()
    rolling_metrics['drawdown'] = (cumulative - rolling_max) / rolling_max
    
    return rolling_metrics

def calculate_factor_loadings(returns, factors):
    """
    Calculate factor loadings using linear regression
    
    Parameters:
    returns: pandas Series of asset returns
    factors: pandas DataFrame of factor returns
    
    Returns:
    dict, factor loadings and statistics
    """
    # Align data
    data = pd.concat([returns, factors], axis=1).dropna()
    
    if len(data) < 10:  # Need minimum observations
        return None
    
    y = data.iloc[:, 0]  # Asset returns
    X = data.iloc[:, 1:]  # Factor returns
    
    # Add constant for alpha
    X = pd.concat([pd.Series(1, index=X.index, name='Alpha'), X], axis=1)
    
    # Linear regression
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = model.predict(X)
        r_squared = r2_score(y, predictions)
        
        # Calculate residuals and statistics
        residuals = y - predictions
        residual_std = residuals.std()
        
        loadings = dict(zip(X.columns, model.coef_))
        
        return {
            'loadings': loadings,
            'r_squared': r_squared,
            'residual_volatility': residual_std * np.sqrt(252),
            'alpha': loadings['Alpha'] * 252,
            'factor_exposures': {k: v for k, v in loadings.items() if k != 'Alpha'}
        }
    
    except ImportError:
        # Fallback to basic correlation if sklearn not available
        correlations = {}
        for factor in factors.columns:
            correlations[factor] = data[factor].corr(y)
        
        return {
            'loadings': correlations,
            'r_squared': np.nan,
            'residual_volatility': np.nan,
            'alpha': np.nan,
            'factor_exposures': correlations
        }

def calculate_attribution(portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns):
    """
    Calculate performance attribution using Brinson model
    
    Parameters:
    portfolio_weights: dict or Series, portfolio weights by asset/sector
    portfolio_returns: dict or Series, portfolio returns by asset/sector
    benchmark_weights: dict or Series, benchmark weights by asset/sector
    benchmark_returns: dict or Series, benchmark returns by asset/sector
    
    Returns:
    dict, attribution analysis
    """
    # Convert to pandas Series if needed
    if isinstance(portfolio_weights, dict):
        portfolio_weights = pd.Series(portfolio_weights)
    if isinstance(portfolio_returns, dict):
        portfolio_returns = pd.Series(portfolio_returns)
    if isinstance(benchmark_weights, dict):
        benchmark_weights = pd.Series(benchmark_weights)
    if isinstance(benchmark_returns, dict):
        benchmark_returns = pd.Series(benchmark_returns)
    
    # Align all series
    common_index = portfolio_weights.index.intersection(portfolio_returns.index)
    common_index = common_index.intersection(benchmark_weights.index)
    common_index = common_index.intersection(benchmark_returns.index)
    
    pw = portfolio_weights.loc[common_index]
    pr = portfolio_returns.loc[common_index]
    bw = benchmark_weights.loc[common_index]
    br = benchmark_returns.loc[common_index]
    
    # Brinson attribution
    allocation_effect = (pw - bw) * br
    selection_effect = bw * (pr - br)
    interaction_effect = (pw - bw) * (pr - br)
    
    total_effect = allocation_effect + selection_effect + interaction_effect
    
    # Portfolio and benchmark returns
    portfolio_return = (pw * pr).sum()
    benchmark_return = (bw * br).sum()
    active_return = portfolio_return - benchmark_return
    
    attribution_results = pd.DataFrame({
        'Portfolio Weight': pw,
        'Benchmark Weight': bw,
        'Portfolio Return': pr,
        'Benchmark Return': br,
        'Active Weight': pw - bw,
        'Active Return': pr - br,
        'Allocation Effect': allocation_effect,
        'Selection Effect': selection_effect,
        'Interaction Effect': interaction_effect,
        'Total Effect': total_effect
    })
    
    return {
        'attribution_by_asset': attribution_results,
        'total_allocation_effect': allocation_effect.sum(),
        'total_selection_effect': selection_effect.sum(),
        'total_interaction_effect': interaction_effect.sum(),
        'total_active_return': active_return,
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return
    }

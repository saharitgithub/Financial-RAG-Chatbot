"""
calculations.py
================
- All calculations use real yfinance data
- Each result formatted cleanly
- LLM reasoning context prepared for app.py to append
"""

import yfinance as yf
import numpy as np
import pandas as pd
import math
import re
from scipy import stats
from datetime import datetime, timedelta


# =========================================
# HISTORICAL DATA FETCHERS
# =========================================

def get_historical_returns(ticker: str, period: str = "1y") -> pd.Series | None:
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or len(hist) < 5:
            return None
        return hist["Close"].pct_change().dropna()
    except Exception:
        return None


def get_historical_prices(ticker: str, period: str = "1y") -> pd.Series | None:
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None
        return hist["Close"]
    except Exception:
        return None


# =========================================
# KURTOSIS
# =========================================

def calculate_kurtosis(ticker: str, period: str = "1y") -> dict:
    returns = get_historical_returns(ticker, period)
    if returns is None or len(returns) < 10:
        return {"error": f"Insufficient data for {ticker}"}

    kurtosis = stats.kurtosis(returns, fisher=True)

    if kurtosis > 1:
        interpretation = "Leptokurtic (Fat Tails)"
        risk           = "⚠️ Higher tail risk — extreme events more likely than normal."
    elif kurtosis < -1:
        interpretation = "Platykurtic (Thin Tails)"
        risk           = "✅ Lower tail risk — extreme events less likely than normal."
    else:
        interpretation = "Mesokurtic (Near Normal)"
        risk           = "📊 Normal tail risk — similar to standard distribution."

    return {
        "calc_type":      "kurtosis",
        "ticker":         ticker,
        "kurtosis":       round(kurtosis, 4),
        "interpretation": interpretation,
        "risk":           risk,
        "data_points":    len(returns),
        "period":         period,
    }


# =========================================
# SKEWNESS
# =========================================

def calculate_skewness(ticker: str, period: str = "1y") -> dict:
    returns = get_historical_returns(ticker, period)
    if returns is None:
        return {"error": f"No data for {ticker}"}

    skew = stats.skew(returns)

    if skew > 0.5:
        interpretation = "Positive skew — more small losses, occasional large gains"
    elif skew < -0.5:
        interpretation = "Negative skew — more small gains, occasional large losses"
    else:
        interpretation = "Approximately symmetric distribution"

    return {
        "calc_type":      "skewness",
        "ticker":         ticker,
        "skewness":       round(skew, 4),
        "interpretation": interpretation,
        "period":         period,
    }


# =========================================
# VOLATILITY
# =========================================

def calculate_volatility(ticker: str, period: str = "1y") -> dict:
    returns = get_historical_returns(ticker, period)
    if returns is None:
        return {"error": f"No data for {ticker}"}

    daily_vol  = returns.std()
    annual_vol = daily_vol * np.sqrt(252)

    return {
        "calc_type":          "volatility",
        "ticker":             ticker,
        "daily_volatility":   round(daily_vol * 100, 4),
        "annual_volatility":  round(annual_vol * 100, 2),
        "period":             period,
    }


# =========================================
# MONTE CARLO VAR
# =========================================

def calculate_monte_carlo_var(
    ticker: str,
    investment: float = 10000,
    days: int = 10,
    confidence: float = 0.95,
    simulations: int = 10000,
) -> dict:
    returns = get_historical_returns(ticker, "2y")
    if returns is None or len(returns) < 20:
        return {"error": f"Insufficient data for {ticker}"}

    mu    = returns.mean()
    sigma = returns.std()

    np.random.seed(42)
    rand_returns     = np.random.normal(mu, sigma, (simulations, days))
    cum_returns      = np.exp(np.cumsum(rand_returns, axis=1))
    final_multipliers = cum_returns[:, -1]
    final_values     = investment * final_multipliers
    losses           = investment - final_values
    pos_losses       = losses[losses > 0]

    var_pct   = 1 - confidence
    var_value = np.percentile(losses, var_pct * 100)
    var_pct_  = (var_value / investment) * 100
    cvar_val  = pos_losses[pos_losses >= var_value].mean() if len(pos_losses[pos_losses >= var_value]) > 0 else var_value

    return {
        "calc_type":        "var",
        "ticker":           ticker,
        "investment":       investment,
        "days":             days,
        "confidence":       confidence,
        "confidence_pct":   int(confidence * 100),
        "mu_daily":         round(mu, 6),
        "sigma_daily":      round(sigma, 6),
        "volatility_annual": round(sigma * np.sqrt(252) * 100, 2),
        "var_amount":       round(var_value, 2),
        "var_percent":      round(var_pct_, 2),
        "cvar_amount":      round(cvar_val, 2),
        "cvar_percent":     round((cvar_val / investment) * 100, 2),
        "simulations":      simulations,
    }


# =========================================
# SHARPE RATIO
# =========================================

def calculate_sharpe_ratio(ticker: str, risk_free_rate: float = 0.02, period: str = "1y") -> dict:
    returns = get_historical_returns(ticker, period)
    if returns is None:
        return {"error": f"No data for {ticker}"}

    daily_vol    = returns.std()
    annual_vol   = daily_vol * np.sqrt(252)
    annual_return = returns.mean() * 252
    sharpe       = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    rating = "Excellent" if sharpe > 1 else "Good" if sharpe > 0.5 else "Average" if sharpe > 0 else "Poor"

    return {
        "calc_type":     "sharpe",
        "ticker":        ticker,
        "sharpe_ratio":  round(sharpe, 4),
        "rating":        rating,
        "annual_return": round(annual_return * 100, 2),
        "volatility":    round(annual_vol * 100, 2),
        "risk_free_rate": risk_free_rate,
        "period":        period,
    }


# =========================================
# SORTINO RATIO
# =========================================

def calculate_sortino_ratio(ticker: str, risk_free_rate: float = 0.02, period: str = "1y") -> dict:
    returns = get_historical_returns(ticker, period)
    if returns is None:
        return {"error": f"No data for {ticker}"}

    downside_returns   = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    annual_return      = returns.mean() * 252
    sortino            = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

    return {
        "calc_type":          "sortino",
        "ticker":             ticker,
        "sortino_ratio":      round(sortino, 4),
        "annual_return":      round(annual_return * 100, 2),
        "downside_deviation": round(downside_deviation * 100, 2),
        "period":             period,
    }


# =========================================
# MAX DRAWDOWN
# =========================================

def calculate_max_drawdown(ticker: str, period: str = "1y") -> dict:
    prices = get_historical_prices(ticker, period)
    if prices is None or len(prices) < 2:
        return {"error": f"No data for {ticker}"}

    cumulative   = (1 + prices.pct_change()).cumprod()
    running_max  = cumulative.expanding().max()
    drawdown     = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    max_dd_idx   = drawdown.idxmin()

    return {
        "calc_type":    "max_drawdown",
        "ticker":       ticker,
        "max_drawdown": round(max_drawdown * 100, 2),
        "peak_date":    max_dd_idx.strftime("%Y-%m-%d") if hasattr(max_dd_idx, "strftime") else str(max_dd_idx),
        "period":       period,
    }


# =========================================
# BETA
# =========================================

def calculate_beta(ticker: str, market_ticker: str = "SPY", period: str = "1y") -> dict:
    stock_returns  = get_historical_returns(ticker, period)
    market_returns = get_historical_returns(market_ticker, period)

    if stock_returns is None or market_returns is None:
        return {"error": "Could not fetch data"}

    combined   = pd.DataFrame({"stock": stock_returns, "market": market_returns}).dropna()
    covariance = combined["stock"].cov(combined["market"])
    variance   = combined["market"].var()
    beta       = covariance / variance if variance > 0 else 1

    if beta > 1.2:
        interpretation = f"High volatility ({beta:.2f}x market) — amplifies market moves"
    elif beta < 0.8:
        interpretation = f"Defensive ({beta:.2f}x market) — less sensitive to market"
    else:
        interpretation = f"Market-like ({beta:.2f}x market) — moves with the market"

    return {
        "calc_type":      "beta",
        "ticker":         ticker,
        "market":         market_ticker,
        "beta":           round(beta, 4),
        "interpretation": interpretation,
        "period":         period,
    }


# =========================================
# CAGR
# =========================================

def calculate_cagr(ticker: str, period_days: int = 365) -> dict:
    period_years = period_days / 365
    if period_days <= 30:      period_str = "1mo"
    elif period_days <= 90:    period_str = "3mo"
    elif period_days <= 180:   period_str = "6mo"
    elif period_days <= 365:   period_str = "1y"
    elif period_days <= 730:   period_str = "2y"
    else:                      period_str = "5y"

    prices = get_historical_prices(ticker, period_str)
    if prices is None or len(prices) < 2:
        return {"error": "Insufficient data"}

    start_price = prices.iloc[0]
    end_price   = prices.iloc[-1]
    cagr        = (end_price / start_price) ** (1 / period_years) - 1 if period_years > 0 else 0

    return {
        "calc_type":    "cagr",
        "ticker":       ticker,
        "start_price":  round(start_price, 2),
        "end_price":    round(end_price, 2),
        "period_days":  period_days,
        "period_years": round(period_years, 2),
        "cagr":         round(cagr * 100, 2),
        "total_return": round(((end_price / start_price) - 1) * 100, 2),
    }


# =========================================
# TOTAL RETURN
# =========================================

def calculate_total_return(ticker: str, period_days: int = 365, amount: float = 10000) -> dict:
    if period_days <= 30:     period_str = "1mo"
    elif period_days <= 90:   period_str = "3mo"
    elif period_days <= 180:  period_str = "6mo"
    elif period_days <= 365:  period_str = "1y"
    elif period_days <= 730:  period_str = "2y"
    else:                     period_str = "5y"

    prices = get_historical_prices(ticker, period_str)
    if prices is None or len(prices) < 2:
        return {"error": "Insufficient data"}

    start_price = prices.iloc[0]
    end_price   = prices.iloc[-1]
    pct_return  = ((end_price - start_price) / start_price) * 100
    final_value = amount * (1 + pct_return / 100)
    profit      = final_value - amount

    return {
        "calc_type":   "total_return",
        "ticker":      ticker,
        "amount":      amount,
        "start_price": round(start_price, 2),
        "end_price":   round(end_price, 2),
        "pct_return":  round(pct_return, 2),
        "final_value": round(final_value, 2),
        "profit":      round(profit, 2),
    }


# =========================================
# CORRELATION
# =========================================

def calculate_correlation(ticker1: str, ticker2: str, period: str = "1y") -> dict:
    returns1 = get_historical_returns(ticker1, period)
    returns2 = get_historical_returns(ticker2, period)

    if returns1 is None or returns2 is None:
        return {"error": "Could not fetch data"}

    combined    = pd.DataFrame({ticker1: returns1, ticker2: returns2}).dropna()
    correlation = combined[ticker1].corr(combined[ticker2])

    if correlation > 0.7:
        interpretation = f"Strong positive ({correlation:.2f}) — move together"
    elif correlation > 0.3:
        interpretation = f"Moderate positive ({correlation:.2f})"
    elif correlation > -0.3:
        interpretation = f"Low/no correlation ({correlation:.2f}) — move independently"
    elif correlation > -0.7:
        interpretation = f"Moderate negative ({correlation:.2f})"
    else:
        interpretation = f"Strong negative ({correlation:.2f}) — move opposite"

    return {
        "calc_type":      "correlation",
        "ticker1":        ticker1,
        "ticker2":        ticker2,
        "correlation":    round(correlation, 4),
        "interpretation": interpretation,
        "data_points":    len(combined),
        "period":         period,
    }


# =========================================
# PROBABILITY
# =========================================

def calculate_probability(ticker: str, target_return: float = 0.10, days: int = 252) -> dict:
    returns = get_historical_returns(ticker, "2y")
    if returns is None:
        return {"error": f"No data for {ticker}"}

    mu      = returns.mean() * np.sqrt(252)
    sigma   = returns.std() * np.sqrt(252)
    z_score = (target_return - mu) / sigma if sigma > 0 else 0
    prob    = 1 - stats.norm.cdf(z_score)

    return {
        "calc_type":       "probability",
        "ticker":          ticker,
        "target_return":   round(target_return * 100, 2),
        "expected_return": round(mu * 100, 2),
        "volatility":      round(sigma * 100, 2),
        "probability":     round(prob * 100, 2),
        "z_score":         round(z_score, 4),
    }


# =========================================
# CONFIDENCE INTERVAL
# =========================================

def calculate_confidence_interval(ticker: str, confidence: float = 0.95, days: int = 10) -> dict:
    try:
        info          = yf.Ticker(ticker).info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception:
        current_price = None

    if not current_price:
        return {"error": f"Cannot get current price for {ticker}"}

    returns = get_historical_returns(ticker, "1y")
    if returns is None:
        return {"error": f"No historical data for {ticker}"}

    sigma_daily  = returns.std()
    sigma_period = sigma_daily * np.sqrt(days)
    z_score      = stats.norm.ppf((1 + confidence) / 2)
    price_change = current_price * sigma_period * z_score

    return {
        "calc_type":     "confidence_interval",
        "ticker":        ticker,
        "current_price": round(current_price, 2),
        "confidence":    confidence,
        "confidence_pct": int(confidence * 100),
        "days":          days,
        "lower_bound":   round(current_price - price_change, 2),
        "upper_bound":   round(current_price + price_change, 2),
    }


# =========================================
# FORMAT RESULT FOR DISPLAY
# =========================================

def format_calculation_result(result: dict) -> str:
    """Format any calculation result into clean readable text."""
    if "error" in result:
        return f"⚠️ {result['error']}"

    ct = result.get("calc_type", "")

    if ct == "kurtosis":
        return (
            f"**📊 Kurtosis — {result['ticker']} ({result['period']})**\n\n"
            f"- Excess Kurtosis  : **{result['kurtosis']}**\n"
            f"- Distribution     : {result['interpretation']}\n"
            f"- Data Points      : {result['data_points']}\n\n"
            f"{result['risk']}"
        )

    if ct == "skewness":
        return (
            f"**📊 Skewness — {result['ticker']} ({result['period']})**\n\n"
            f"- Skewness         : **{result['skewness']}**\n"
            f"- Interpretation   : {result['interpretation']}"
        )

    if ct == "volatility":
        return (
            f"**📊 Volatility — {result['ticker']} ({result['period']})**\n\n"
            f"- Daily Volatility  : **{result['daily_volatility']}%**\n"
            f"- Annual Volatility : **{result['annual_volatility']}%**"
        )

    if ct == "var":
        return (
            f"**🎲 Monte Carlo VaR — {result['ticker']}**\n\n"
            f"**Parameters:**\n"
            f"- Investment   : ${result['investment']:,.2f}\n"
            f"- Horizon      : {result['days']} days\n"
            f"- Confidence   : {result['confidence_pct']}%\n"
            f"- Simulations  : {result['simulations']:,}\n\n"
            f"**Results:**\n"
            f"- Annual Vol   : {result['volatility_annual']}%\n"
            f"- **VaR**      : ${result['var_amount']:,.2f} ({result['var_percent']}%)\n"
            f"- **CVaR**     : ${result['cvar_amount']:,.2f} ({result['cvar_percent']}%)\n\n"
            f"*At {result['confidence_pct']}% confidence, max expected loss over "
            f"{result['days']} days is **${result['var_amount']:,.2f}***"
        )

    if ct == "sharpe":
        return (
            f"**📈 Sharpe Ratio — {result['ticker']} ({result['period']})**\n\n"
            f"- Sharpe Ratio  : **{result['sharpe_ratio']}**\n"
            f"- Rating        : {result['rating']}\n"
            f"- Annual Return : {result['annual_return']}%\n"
            f"- Volatility    : {result['volatility']}%\n"
            f"- Risk-Free Rate: {result['risk_free_rate']*100}%"
        )

    if ct == "sortino":
        return (
            f"**📈 Sortino Ratio — {result['ticker']} ({result['period']})**\n\n"
            f"- Sortino Ratio      : **{result['sortino_ratio']}**\n"
            f"- Annual Return      : {result['annual_return']}%\n"
            f"- Downside Deviation : {result['downside_deviation']}%"
        )

    if ct == "max_drawdown":
        return (
            f"**📉 Max Drawdown — {result['ticker']} ({result['period']})**\n\n"
            f"- Max Drawdown : **{result['max_drawdown']}%**\n"
            f"- Peak Date    : {result['peak_date']}"
        )

    if ct == "beta":
        return (
            f"**📉 Beta — {result['ticker']} vs {result['market']} ({result['period']})**\n\n"
            f"- Beta            : **{result['beta']}**\n"
            f"- Interpretation  : {result['interpretation']}"
        )

    if ct == "cagr":
        arrow = "🟢" if result["cagr"] >= 0 else "🔴"
        return (
            f"**📊 CAGR — {result['ticker']}**\n\n"
            f"- Start Price  : ${result['start_price']}\n"
            f"- End Price    : ${result['end_price']}\n"
            f"- Period       : {result['period_years']} years\n"
            f"- **CAGR**     : {arrow} {result['cagr']}%\n"
            f"- Total Return : {result['total_return']}%"
        )

    if ct == "total_return":
        arrow = "🟢" if result["profit"] >= 0 else "🔴"
        return (
            f"**📊 Total Return — {result['ticker']}**\n\n"
            f"- Amount Invested : ${result['amount']:,.2f}\n"
            f"- Start Price     : ${result['start_price']}\n"
            f"- End Price       : ${result['end_price']}\n"
            f"- Return          : {result['pct_return']}%\n"
            f"- Final Value     : **${result['final_value']:,.2f}**\n"
            f"- Profit / Loss   : {arrow} ${result['profit']:+,.2f}"
        )

    if ct == "correlation":
        return (
            f"**🔄 Correlation — {result['ticker1']} vs {result['ticker2']} ({result['period']})**\n\n"
            f"- Correlation  : **{result['correlation']}**\n"
            f"- Meaning      : {result['interpretation']}\n"
            f"- Data Points  : {result['data_points']}"
        )

    if ct == "probability":
        return (
            f"**📊 Probability of Return — {result['ticker']}**\n\n"
            f"- Target Return   : {result['target_return']}%\n"
            f"- Expected Return : {result['expected_return']}%\n"
            f"- Volatility      : {result['volatility']}%\n"
            f"- **Probability** : **{result['probability']}%**\n"
            f"- Z-Score         : {result['z_score']}"
        )

    if ct == "confidence_interval":
        return (
            f"**📊 Price Confidence Interval — {result['ticker']}**\n\n"
            f"- Current Price  : ${result['current_price']}\n"
            f"- Horizon        : {result['days']} days\n"
            f"- Confidence     : {result['confidence_pct']}%\n"
            f"- **Lower Bound**: ${result['lower_bound']}\n"
            f"- **Upper Bound**: ${result['upper_bound']}"
        )

    return str(result)


# =========================================
# DETECT CALCULATION FROM QUERY
# =========================================

CALCULATION_MAP = {
    "kurtosis":            calculate_kurtosis,
    "skewness":            calculate_skewness,
    "volatility":          calculate_volatility,
    "var":                 calculate_monte_carlo_var,
    "value at risk":       calculate_monte_carlo_var,
    "monte carlo":         calculate_monte_carlo_var,
    "sharpe":              calculate_sharpe_ratio,
    "sortino":             calculate_sortino_ratio,
    "max drawdown":        calculate_max_drawdown,
    "drawdown":            calculate_max_drawdown,
    "beta":                calculate_beta,
    "cagr":                calculate_cagr,
    "total return":        calculate_total_return,
    "correlation":         calculate_correlation,
    "probability":         calculate_probability,
    "confidence interval": calculate_confidence_interval,
}


def detect_and_run_calculation(query: str, tickers: list) -> str | None:
    """
    Detects calculation type from query, runs it, returns formatted result.
    Returns None if no calculation detected.
    """
    q = query.lower()

    for calc_name, calc_func in CALCULATION_MAP.items():
        if calc_name in q:

            # Extract params
            period_str = "1y"
            for pat, p in [(r'(\d+)\s*year', lambda n: f"{n}y"),
                           (r'(\d+)\s*month', lambda n: f"{n}mo")]:
                m = re.search(pat, q)
                if m:
                    period_str = p(m.group(1))
                    break

            amount = 10000
            amt_m  = re.search(r'\$?\s*([\d,]+)\s*(k|m|b)?', q)
            if amt_m:
                v = float(amt_m.group(1).replace(",",""))
                s = (amt_m.group(2) or "").lower()
                if s == "k": v *= 1000
                elif s == "m": v *= 1_000_000
                amount = v

            days = 10
            dm = re.search(r'(\d+)\s*day', q)
            if dm:
                days = int(dm.group(1))

            confidence = 0.95
            if "99%" in q or "99 percent" in q:
                confidence = 0.99

            period_days = 365
            for pat2, mult in [(r'(\d+)\s*year', 365), (r'(\d+)\s*month', 30),
                               (r'(\d+)\s*week', 7), (r'(\d+)\s*day', 1)]:
                m2 = re.search(pat2, q)
                if m2:
                    period_days = int(m2.group(1)) * mult
                    break

            target_return = 0.10
            tm = re.search(r'(\d+(?:\.\d+)?)\s*%', q)
            if tm:
                target_return = float(tm.group(1)) / 100

            # Run based on type
            try:
                if calc_name in ("correlation",):
                    if len(tickers) < 2:
                        return "⚠️ Correlation needs 2 tickers. Example: correlation AAPL and MSFT"
                    result = calc_func(tickers[0], tickers[1], period_str)

                elif calc_name in ("var", "value at risk", "monte carlo"):
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker. Example: calculate VaR for AAPL"
                    result = calc_func(ticker, amount, days, confidence)

                elif calc_name in ("cagr",):
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker."
                    result = calc_func(ticker, period_days)

                elif calc_name in ("total return",):
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker."
                    result = calc_func(ticker, period_days, amount)

                elif calc_name in ("probability",):
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker."
                    result = calc_func(ticker, target_return)

                elif calc_name in ("confidence interval",):
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker."
                    result = calc_func(ticker, confidence, days)

                elif calc_name in ("beta",):
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker."
                    market = tickers[1] if len(tickers) > 1 else "SPY"
                    result = calc_func(ticker, market, period_str)

                else:
                    ticker = tickers[0] if tickers else None
                    if not ticker:
                        return "⚠️ Please specify a ticker."
                    result = calc_func(ticker, period_str)

                return format_calculation_result(result)

            except Exception as e:
                return f"⚠️ Calculation error: {str(e)}"

    return None
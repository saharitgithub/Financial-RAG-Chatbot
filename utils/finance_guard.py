#finance guard
import re
from config import OPENROUTER_API_KEY
import requests

DOMAIN_KEYWORDS = [
    # Finance and investment
    "finance", "financial", "investment", "portfolio", "risk", "return", "asset", "liability",
    "equity", "debt", "bond", "stock", "share", "dividend", "cash flow", "valuation",
    "market", "capital", "capm", "beta", "alpha", "sharpe", "volatility", "liquidity",
    "derivative", "option", "future", "futures", "swap", "hedge", "arbitrage", "leverage",
    "asset allocation", "diversification", "fund", "etf", "mutual fund", "interest rate",
    "inflation", "credit", "default", "credit risk", "value at risk", "var",
    "ratio analysis", "financial statement", "balance sheet", "income statement", "cash flow statement",
    "accounting", "audit", "tax", "depreciation", "amortization", "book value", "earnings",
    "revenue", "expense", "profit", "loss", "cost accounting", "cost of capital",
    "exposure", "portfolio exposure", "sector exposure", "geographic exposure",
    "economics", "economic", "macroeconomics", "microeconomics", "fiscal policy", "monetary policy",
    "gdp", "cpi", "exchange rate", "yield", "bond yield", "herfindahl", "hhi", "hirschman",
    "market concentration", "concentration index", "market share", "competitive",
    # Statistics and analytics
    "statistics", "statistic", "probability", "distribution", "mean", "median", "mode",
    "variance", "covariance", "correlation", "regression", "standard deviation",
    "hypothesis", "sample", "population", "forecast", "analytics", "quantitative",
    "time series", "factor", "risk management", "performance", "benchmark",
    "capital asset pricing model", "correlation matrix", "covariance matrix", "coefficient matrix",
    "regression analysis", "linear regression", "portfolio variance", "expected return",
    # Investment metrics
    "bid ask spread", "bid_ask_spread", "sortino ratio", "sortino_ratio",
    "correlation_matrix", "skewness", "kurtosis",
    "leverage ratio", "leverage_ratio", "tracking error", "tracking_error",
    "downside return", "downside-adjusted", "non-normal risk", "exit cost",
    "margin", "derivatives", "sharpe ratio", "information ratio", "treynor ratio",
    # Risk & Volatility
    "portfolio_volatility", "downside_volatility", "value_at_risk",
    "conditional_var", "cvar", "expected shortfall", "max_drawdown", "drawdown",
    "annualized", "std dev", "standard deviation", "annualized std dev",
    "peak-to-trough", "peak to trough",
    # Factor & Sensitivity
    "beta_to_benchmark", "factor_exposures", "factor exposure",
    "size", "value", "momentum", "duration", "interest_rate_sensitivity",
    "interest rate sensitivity", "currency_exposure", "currency exposure", "fx bucket",
    # Concentration & Diversification
    "herfindahl_hirschman_index", "effective_number_of_positions",
    "effective number of positions", "top_5_exposure", "top_10_exposure",
    "top 5 exposure", "top 10 exposure", "diversification_ratio", "diversification ratio",
    "top_5_exposure_pct", "top_10_exposure_pct",
    # Liquidity & Execution Risk
    "liquidity_score", "liquidity score", "days_to_liquidate", "days to liquidate",
    "average daily volume", "adv", "bid_ask_spread_impact", "bid ask spread impact",
    "execution risk",
    # Stress & Tail Risk
    "stress_test", "stress test", "stress_loss", "stress loss",
    "scenario_exposure", "scenario exposure", "scenario",
    "tail_dependency", "tail dependency", "extreme moves",
    "rate shock", "oil spike", "equity crash", "stress_test_loss",
    # Performance Attribution
    "risk_adjusted", "risk adjusted", "downside_adjusted", "downside adjusted",
    "performance", "attribution", "information_ratio", "sharpe_ratio", "sortino_ratio",
    # Correlations
    "pairwise correlation", "non-normal", "correlation_matrix_summary",
    # General finance concepts
    "ebitda", "dollar cost averaging", "dollar cost average", "dca",
    "p/e ratio", "price to earnings", "dcf", "discounted cash flow",
    "net present value", "npv", "internal rate of return", "irr",
    "intrinsic value", "market cap", "enterprise value",
    "efficient market", "systematic risk", "unsystematic risk",
    "rebalancing", "compounding", "time value of money", "opportunity cost",
    "risk premium", "deflation", "recession", "bull market", "bear market",
    "fixed income", "commodity", "real estate", "capital gain", "coupon", "maturity",
    "hedge fund", "private equity", "venture capital", "ipo", "financial ratio",
    "liquidity ratio", "solvency", "working capital", "gross margin", "net margin",
    "operating margin", "roe", "roa", "earnings per share", "eps", "debt to equity",
    "index fund", "asset class",
    # Statistical tests and formulas
    "bartlett", "bartlett test", "chi square", "chi-square", "levene",
    "shapiro", "shapiro-wilk", "kolmogorov", "anova", "t-test", "f-test", "z-test",
    "hypothesis test", "null hypothesis", "p-value", "significance",
    "confidence interval", "normal distribution", "log", "logarithm",
    "matrix", "determinant", "eigenvalue",
    "formula", "calculate", "compute", "test statistic", "degrees of freedom",
    "correlation coefficient", "pearson", "spearman", "kendall",
    "autocorrelation", "heteroscedasticity", "multicollinearity",
    "stationarity", "adf test", "augmented dickey fuller",
    "monte carlo", "simulation", "bootstrap", "bayes", "bayesian",
]

NON_FINANCE_TERMS = [
    "movie", "film", "song", "music", "joke", "recipe", "cooking", "restaurant", "food",
    "cat", "dog", "animal", "pet", "bird", "fish", "zoo", "garden", "gardening",
    "sports", "football", "cricket", "soccer", "hockey", "game", "video game", "gaming",
    "programming", "coding", "computer science", "software", "app", "website",
    "history", "literature", "novel", "poem", "poetry", "art", "painting", "drawing",
    "politics", "government", "travel", "vacation", "fashion", "makeup", "beauty",
    "weather", "health", "medical", "biology", "chemistry", "physics",
]


def _contains_term(query: str, terms):
    for term in terms:
        if term in query:
            return True
    return False


def _contains_non_finance_term(query: str, terms):
    for term in terms:
        if len(term) <= 3:
            if re.search(r"\b" + re.escape(term) + r"\b", query):
                return True
        else:
            if term in query:
                return True
    return False


def is_finance_query(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if _contains_non_finance_term(q, NON_FINANCE_TERMS):
        return False
    if _contains_term(q, DOMAIN_KEYWORDS):
        return True
    return False


def call_openrouter(prompt: str) -> str:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
    )
    return response.json()["choices"][0]["message"]["content"]


def is_finance_query_llm(query: str) -> bool:
    FOLLOWUP_TERMS = [
    "yes",
    "ok",
    "okay",
    "continue",
    "do it",
    "calculate it",
    "show steps",
    "go ahead",
]

    q = query.strip().lower()

    if q in FOLLOWUP_TERMS:
       return True

    if "=" in q:
       return True
   
    prompt = f"""
You are a classifier for a finance and quantitative analysis chatbot.

Answer YES if the query relates to ANY of:
- Finance, investment, banking, accounting, economics
- Statistics, probability, mathematics used in finance
- Risk analytics, portfolio analysis, quantitative methods
- Financial formulas, tests, matrices, coefficients
- Stock markets, tickers, companies, trading

Answer NO only for completely unrelated topics like movies, food, sports, cooking.

Query: {query}

Answer only YES or NO.
"""
    response = call_openrouter(prompt)
    return "YES" in response.upper()
"""
finance_data.py
================
- detect_switch_query: standalone export for app.py
- Investment switch: full amount to target, NO splitting
- Period detected from query (default current/instant for switch)
- Spelling corrector: fuzzy word-by-word
- Weekly OHLC always shown
- Crypto detection supported (BTC-USD, ETH-USD, etc.)
"""

import yfinance as yf
import streamlit as st
import re
import pandas as pd
from datetime import datetime, timedelta
from difflib import get_close_matches

# =========================================
# TICKER UNIVERSE
# =========================================

SECTOR_TICKERS = {
    "Tech":       ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD", "INTC", "QCOM", "ORCL"],
    "Finance":    ["JPM", "GS", "MS", "BAC", "BRK-B", "C", "WFC"],
    "Energy":     ["XOM", "CVX", "SLB", "COP", "PSX"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH"],
    "Consumer":   ["WMT", "COST", "HD", "NKE"],
    "Utilities":  ["NEE", "DUK", "SO"],
    "Industrial": ["BA", "CAT", "GE", "HON", "LMT"],
    "Media":      ["NFLX", "DIS", "SPOT"],
    "Staples":    ["KO", "PEP", "PG", "CL"],
    "Telecom":    ["T", "VZ", "TMUS"],
}

ALL_TICKERS = sorted({t for tickers in SECTOR_TICKERS.values() for t in tickers})

NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN",
    "meta": "META", "facebook": "META", "amd": "AMD",
    "intel": "INTC", "qualcomm": "QCOM", "oracle": "ORCL",
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman": "GS",
    "goldman sachs": "GS", "morgan stanley": "MS", "bank of america": "BAC",
    "berkshire": "BRK-B", "citigroup": "C", "citi": "C",
    "wells fargo": "WFC", "exxon": "XOM", "chevron": "CVX",
    "schlumberger": "SLB", "conocophillips": "COP", "phillips 66": "PSX",
    "johnson": "JNJ", "pfizer": "PFE", "merck": "MRK",
    "abbvie": "ABBV", "unitedhealth": "UNH", "walmart": "WMT",
    "costco": "COST", "home depot": "HD", "nike": "NKE",
    "nextera": "NEE", "duke": "DUK", "southern": "SO",
    "boeing": "BA", "caterpillar": "CAT", "general electric": "GE",
    "honeywell": "HON", "lockheed": "LMT", "netflix": "NFLX",
    "disney": "DIS", "spotify": "SPOT", "coca cola": "KO",
    "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "procter": "PG", "p&g": "PG", "colgate": "CL",
    "at&t": "T", "verizon": "VZ", "t-mobile": "TMUS", "tmobile": "TMUS",
    "tesla": "TSLA", "tsla": "TSLA",
    # Lowercase ticker symbols for natural language queries
    "amzn": "AMZN", "aapl": "AAPL", "msft": "MSFT", "nvda": "NVDA",
    "googl": "GOOGL", "goog": "GOOGL", "nflx": "NFLX",
    "dis": "DIS", "spot": "SPOT", "intc": "INTC",
    "qcom": "QCOM", "orcl": "ORCL", "jpm": "JPM", "gs": "GS",
    "bac": "BAC", "wfc": "WFC", "xom": "XOM", "cvx": "CVX",
    "jnj": "JNJ", "pfe": "PFE", "mrk": "MRK", "unh": "UNH",
    "wmt": "WMT", "ko": "KO", "pep": "PEP", "pg": "PG",
}

# Crypto tickers — yfinance uses BTC-USD format
CRYPTO_NAME_MAP = {
    "btc": "BTC-USD", "bitcoin": "BTC-USD",
    "eth": "ETH-USD", "ethereum": "ETH-USD",
    "bnb": "BNB-USD", "binance": "BNB-USD",
    "sol": "SOL-USD", "solana": "SOL-USD",
    "xrp": "XRP-USD", "ripple": "XRP-USD",
    "ada": "ADA-USD", "cardano": "ADA-USD",
    "doge": "DOGE-USD", "dogecoin": "DOGE-USD",
    "ltc": "LTC-USD", "litecoin": "LTC-USD",
    "dot": "DOT-USD", "polkadot": "DOT-USD",
    "avax": "AVAX-USD", "avalanche": "AVAX-USD",
}

CRYPTO_DESCRIPTIONS = {
    "BTC-USD":  "Bitcoin (BTC) is the world's first and largest cryptocurrency by market cap. It is a decentralized digital currency that operates without a central bank, using blockchain technology for peer-to-peer transactions.",
    "ETH-USD":  "Ethereum (ETH) is a decentralized blockchain platform that enables smart contracts and decentralized applications (dApps). It is the second largest cryptocurrency by market cap.",
    "DOGE-USD": "Dogecoin (DOGE) started as a meme cryptocurrency in 2013 but gained massive popularity, largely driven by social media and endorsements from Elon Musk. It uses a Proof-of-Work consensus and is widely used for tipping and micro-transactions.",
    "SOL-USD":  "Solana (SOL) is a high-performance blockchain supporting smart contracts and DeFi applications, known for extremely fast transaction speeds and low fees.",
    "BNB-USD":  "Binance Coin (BNB) is the native cryptocurrency of the Binance exchange ecosystem, used for trading fee discounts, token sales, and powering the BNB Smart Chain.",
    "XRP-USD":  "XRP is the native token of the Ripple network, designed for fast, low-cost international money transfers between financial institutions.",
    "ADA-USD":  "Cardano (ADA) is a proof-of-stake blockchain platform built with a research-driven approach, focused on security, scalability, and sustainability.",
    "LTC-USD":  "Litecoin (LTC) is one of the earliest altcoins, designed as a faster and lighter version of Bitcoin with lower transaction fees.",
    "DOT-USD":  "Polkadot (DOT) is a multi-chain protocol that enables different blockchains to transfer messages and value in a trust-free fashion.",
    "AVAX-USD": "Avalanche (AVAX) is a smart contract platform competing with Ethereum, known for fast finality and a unique consensus mechanism.",
}

TICKER_SPELLING = {
    "appl": "AAPL", "aple": "AAPL", "appel": "AAPL", "aplle": "AAPL",
    "mcrsft": "MSFT", "mcrsoft": "MSFT", "mircosoft": "MSFT", "microsft": "MSFT",
    "nvdia": "NVDA", "nvidea": "NVDA", "nivda": "NVDA",
    "gogle": "GOOGL", "gooogle": "GOOGL",
    "amazn": "AMZN", "amazom": "AMZN", "amzon": "AMZN",
    "testa": "TSLA", "teslab": "TSLA",
    "jpmorgen": "JPM", "jpmorgn": "JPM",
    "netflx": "NFLX", "netfix": "NFLX", "netlfix": "NFLX",
    "disny": "DIS", "disnep": "DIS",
}

FINANCE_SPELLING = {
    "sharpe ratio": "Sharpe Ratio", "sharpee ratio": "Sharpe Ratio", "sharp ratio": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "ebitda": "EBITDA (Earnings Before Interest, Taxes, Depreciation & Amortization)",
    "capm": "CAPM (Capital Asset Pricing Model)", "dcf": "DCF (Discounted Cash Flow)",
    "pe ratio": "P/E Ratio (Price-to-Earnings)", "roe": "ROE (Return on Equity)",
    "roa": "ROA (Return on Assets)", "eps": "EPS (Earnings Per Share)",
    "hhi": "HHI (Herfindahl-Hirschman Index)", "npv": "NPV (Net Present Value)",
    "irr": "IRR (Internal Rate of Return)", "dca": "DCA (Dollar Cost Averaging)",
    "etf": "ETF (Exchange-Traded Fund)", "ipo": "IPO (Initial Public Offering)",
    "var": "VaR (Value at Risk)", "cvar": "CVaR (Conditional Value at Risk)",
}


# =========================================
# SPELLING DETECTOR
# =========================================

def _spelling_card(wrote: str, shown_name: str, ticker: str) -> str:
    return (
        f"**Spelling Correction** ✅\n\n"
        f"- You wrote : `{wrote}`\n"
        f"- Correct   : **{shown_name}** (ticker: **{ticker}**)"
    )


def detect_spelling_query(query: str):
    q = query.strip().lower()
    triggers = [
        "spelling", "spell", "correct spelling", "how to spell",
        "what is spelling", "what is the spelling", "speling", "spelng", "correct spell",
    ]
    if not any(t in q for t in triggers):
        return None

    clean = q
    for t in sorted(triggers, key=len, reverse=True):
        clean = clean.replace(t, " ")
    clean = re.sub(r'\b(of|for|the|what|is|a|an|correct|please|tell|me)\b', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if not clean:
        return None

    words = clean.split()

    # 1. Already a valid ticker
    for word in words:
        upper = word.upper()
        if upper in ALL_TICKERS:
            info = get_ticker_info(upper)
            name = info.get("longName", upper) if "error" not in info else upper
            return f"**Spelling Confirmed** ✅\n\n- Ticker : **{upper}**\n- Name   : {name}\n- Status : Correct!"

    # 2. Known misspelling table
    for word in words:
        if word in TICKER_SPELLING:
            correct = TICKER_SPELLING[word]
            info = get_ticker_info(correct)
            name = info.get("longName", correct) if "error" not in info else correct
            return _spelling_card(word, name, correct)       # FIX: name first, input as typed

    # 3. Company-name fuzzy match BEFORE ticker fuzzy match: the user is asking how to
    #    spell a word, so the answer should be the company name, not just a ticker.
    for word in words:
        nm = get_close_matches(word, list(NAME_TO_TICKER.keys()), n=1, cutoff=0.6)
        if nm:
            ct = NAME_TO_TICKER[nm[0]]
            return _spelling_card(word, nm[0].title(), ct)

    # 4. Ticker fuzzy match
    for word in words:
        matches = get_close_matches(word.upper(), ALL_TICKERS, n=1, cutoff=0.6)
        if matches:
            correct = matches[0]
            info = get_ticker_info(correct)
            name = info.get("longName", correct) if "error" not in info else correct
            return _spelling_card(word, name, correct)

    # 5. Finance terms
    tm = get_close_matches(clean, list(FINANCE_SPELLING.keys()), n=1, cutoff=0.6)
    if tm:
        full = FINANCE_SPELLING[tm[0]]
        return f"**Finance Term** ✅\n\n- You wrote : `{clean}`\n- Correct   : **{full}**\n\nWould you like me to explain **{full}**?"

    return (
        f"I couldn't find a close match for `{clean}`. "
        f"Known tickers: {', '.join(ALL_TICKERS[:15])}..."
    )


# =========================================
# TICKER DETECTION
# =========================================

def detect_tickers(query: str) -> list:
    """All tickers mentioned in the query, in the order they were mentioned
    ("google vs microsoft" -> ["GOOGL", "MSFT"])."""
    q       = query.strip()
    q_lower = q.lower()
    hits    = []   # (position in query, ticker)

    # Stock tickers — uppercase symbols
    for m in re.finditer(r'\b[A-Z]{1,5}(?:-[A-Z]{2,3})?\b', q):
        if m.group(0) in ALL_TICKERS:
            hits.append((m.start(), m.group(0)))

    # Company names, misspelled tickers, crypto names
    for table in (NAME_TO_TICKER, TICKER_SPELLING, CRYPTO_NAME_MAP):
        for name, ticker in table.items():
            pos = q_lower.find(name)
            if pos != -1:
                hits.append((pos, ticker))

    found = []
    for _, ticker in sorted(hits, key=lambda h: h[0]):   # stable: ties keep detection order
        if ticker not in found:
            found.append(ticker)
    return found


def typo_correction_note(query: str) -> str:
    """One-line notice when a known misspelling (e.g. 'nvdia') was silently corrected."""
    for word in re.findall(r"[a-z&\-]+", query.lower()):
        if word in TICKER_SPELLING:
            return f"*Showing results for **{TICKER_SPELLING[word]}** (corrected from '{word}').*\n\n"
    return ""


def detect_ticker(query: str):
    """Returns first detected ticker, or None."""
    t = detect_tickers(query)
    return t[0] if t else None


# =========================================
# GRAPH PERIOD DETECTION
# =========================================

def detect_graph_period(query: str) -> str:
    q = query.lower()
    m = re.search(r'(\d+)\s*month', q)
    if m:
        n = int(m.group(1))
        if n <= 1: return "1mo"
        if n <= 3: return "3mo"
        if n <= 6: return "6mo"
        return "1y"
    w = re.search(r'(\d+)\s*week', q)
    if w:
        n = int(w.group(1))
        if n <= 1: return "5d"
        if n <= 4: return "1mo"
        return "3mo"
    y = re.search(r'(\d+)\s*year', q)
    if y:
        n = int(y.group(1))
        if n == 1: return "1y"
        if n == 2: return "2y"
        return "5y"
    d = re.search(r'(\d+)\s*day', q)
    if d:
        n = int(d.group(1))
        if n <= 5:  return "5d"
        if n <= 30: return "1mo"
        return "3mo"
    if "ytd" in q: return "ytd"
    if "max" in q: return "max"
    return "5y"


# =========================================
# AMOUNT PARSER
# =========================================

def parse_amount(raw: str, suffix: str) -> float | None:
    try:
        v = float(raw.replace(",", ""))
    except Exception:
        return None
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
    return v * mult.get((suffix or "").lower().strip(), 1)


# =========================================
# RESOLVE COMPANY NAME TO TICKER
# =========================================

def _resolve(name: str) -> str | None:
    if not name:
        return None
    name = re.sub(
        r'\b(the|a|an|my|all|it|stock|stocks|shares|money|funds|in|into|to|from|and|for|of|with|on|at|by|some|any)\b',
        '', name.lower()
    ).strip()
    if not name:
        return None
    if name in NAME_TO_TICKER:
        return NAME_TO_TICKER[name]
    matches = get_close_matches(name, list(NAME_TO_TICKER.keys()), n=1, cutoff=0.55)
    if matches:
        return NAME_TO_TICKER[matches[0]]
    upper = name.upper().strip()
    if upper in ALL_TICKERS:
        return upper
    return None


# =========================================
# SWITCH DETECTION — STANDALONE EXPORT
# =========================================

def detect_switch_query(query: str) -> dict | None:
    """
    Detects switch/move/pump money queries.
    Returns dict: {amount, source, target, is_switch, period_days, period_label}
    or None if not a switch query.
    """
    q = query.lower()

    switch_kw = ["from", "to", "into", "switch", "move", "transfer", "shift", "pump", "sell", "reinvest"]
    if not any(kw in q for kw in switch_kw):
        return None

    # Must have amount
    amt_match = re.search(
        r'\$?\s*([\d,]+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?',
        q, re.IGNORECASE
    )
    if not amt_match:
        return None

    amount = parse_amount(amt_match.group(1), amt_match.group(2) or "")
    if not amount:
        return None

    # Patterns to extract FROM and TO
    patterns = [
        # "invest/put/move X from A to B"
        r'(?:invest|put|move|switch|transfer|shift|pump|take)\s+(?:my\s+)?(?:\$?[\d,k m b]+\s+)?(?:from|of)\s+([\w\s]+?)\s+(?:to|into)\s+([\w\s]+?)(?:\s+\d|\s*[?\.]|$)',
        # "X from A to B"  / "from A to B"
        r'(?:from|out of)\s+([\w\s]+?)\s+(?:to|into)\s+([\w\s]+?)(?:\s+\d|\s*[?\.]|$)',
        # "move X to B from A"
        r'(?:move|switch|pump)\s+.*?(?:to|into)\s+([\w\s]+?)\s+(?:from|of)\s+([\w\s]+?)(?:\s+\d|\s*[?\.]|$)',
        # "sell A and buy B"
        r'sell\s+([\w\s]+?)\s+and\s+(?:buy|invest in|put in)\s+([\w\s]+?)(?:\s+\d|\s*[?\.]|$)',
        # "what if i invested X to B from A"
        r'(?:what if|if i)\s+(?:invest|put|move)\s+.*?(?:to|into)\s+([\w\s]+?)\s+from\s+([\w\s]+?)(?:\s+\d|\s*[?\.]|$)',
    ]

    source_ticker = None
    target_ticker = None

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            g1 = match.group(1).strip()
            g2 = match.group(2).strip()

            # Determine which is source and which is target based on pattern
            if "to" in pattern and "from" in pattern[:pattern.index("to")]:
                source_ticker = _resolve(g1)
                target_ticker = _resolve(g2)
            elif "move" in pattern and "to" in pattern and "from" in pattern[pattern.index("to"):]:
                target_ticker = _resolve(g1)
                source_ticker = _resolve(g2)
            elif "sell" in pattern:
                source_ticker = _resolve(g1)
                target_ticker = _resolve(g2)
            else:
                t1 = _resolve(g1)
                t2 = _resolve(g2)
                if t1 and t2:
                    source_ticker = t1
                    target_ticker = t2
                elif t2:
                    target_ticker = t2
                elif t1:
                    target_ticker = t1

            if target_ticker:
                break

    if not target_ticker:
        return None

    # Period
    period_days, period_label = 0, "current/instant"
    for pattern, mult, unit in [
        (r'(\d+)\s*year', 365, "year"), (r'(\d+)\s*month', 30, "month"),
        (r'(\d+)\s*week', 7, "week"),   (r'(\d+)\s*day', 1, "day"),
    ]:
        m = re.search(pattern, q)
        if m:
            num          = int(m.group(1))
            period_days  = num * mult
            period_label = f"{num} {unit}{'s' if num > 1 else ''}"
            break

    return {
        "amount":       amount,
        "source":       source_ticker,
        "target":       target_ticker,
        "is_switch":    True,
        "period_days":  period_days,
        "period_label": period_label,
    }


# =========================================
# INVESTMENT DETECTOR (regular, non-switch)
# =========================================

def detect_investment_query(query: str) -> dict | None:
    q = query.lower()

    invest_kw = [
        "invested", "invest", "put", "bought", "purchase", "if i invest",
        "if i put", "what if i", "how much would", "returns", "profit",
        "gain", "worth now", "portfolio",
    ]
    if not any(kw in q for kw in invest_kw):
        return None

    amt_match = re.search(
        r'\$?\s*([\d,]+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?',
        q, re.IGNORECASE
    )
    if not amt_match:
        return None

    amount = parse_amount(amt_match.group(1), amt_match.group(2) or "")
    if not amount:
        return None

    tickers = detect_tickers(query)
    if not tickers:
        return None

    # Period — default 1 year for regular investment
    period_days, period_label = 365, "1 year"
    for pattern, mult, unit in [
        (r'(\d+)\s*year', 365, "year"), (r'(\d+)\s*month', 30, "month"),
        (r'(\d+)\s*week', 7, "week"),   (r'(\d+)\s*day', 1, "day"),
    ]:
        m = re.search(pattern, q)
        if m:
            num          = int(m.group(1))
            period_days  = num * mult
            period_label = f"{num} {unit}{'s' if num > 1 else ''}"
            break

    return {
        "amount":       amount,
        "tickers":      tickers,
        "per_ticker":   amount / len(tickers),
        "period_days":  period_days,
        "period_label": period_label,
        "is_switch":    False,
        "source":       None,
    }


# =========================================
# DATA FETCHERS
# =========================================

@st.cache_data(ttl=300, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def get_price_history(ticker: str, period: str = "5y"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist if not hist.empty else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_weekly_data(ticker: str) -> dict | None:
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
        if hist.empty:
            return None
        return {
            "week_open":  round(float(hist["Open"].iloc[0]),  2),
            "week_high":  round(float(hist["High"].max()),    2),
            "week_low":   round(float(hist["Low"].min()),     2),
            "week_close": round(float(hist["Close"].iloc[-1]),2),
            "volume":     int(hist["Volume"].sum()),
        }
    except Exception:
        return None


# =========================================
# HELPERS
# =========================================

def _safe(info, key, default="N/A"):
    val = info.get(key)
    return default if (val is None or val == "" or val == 0) else val


def _fmt(val):
    try:
        val = float(val)
        if val >= 1e12: return f"${val/1e12:.2f}T"
        if val >= 1e9:  return f"${val/1e9:.2f}B"
        if val >= 1e6:  return f"${val/1e6:.2f}M"
        return f"${val:,.2f}"
    except:
        return str(val)


def _pct(val):
    try:    return f"{float(val)*100:.2f}%"
    except: return "N/A"


def _pe(info) -> str:
    """Safely format P/E ratio to 2 decimals — avoids crash when value is N/A."""
    try:
        val = info.get("trailingPE")
        if val is None or val == "" or val == 0:
            return "N/A"
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _dy(info) -> str:
    """Dividend yield as a percent string. Computed from rate/price so it
    doesn't depend on yfinance's unit convention for dividendYield."""
    try:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        rate = info.get("dividendRate")

        if rate is not None and price:
            pct = float(rate) / float(price) * 100
        else:
            val = info.get("dividendYield")
            if val in (None, ""):
                return "N/A"
            pct = float(val)  # this yfinance version returns percent already

        if pct < 0 or pct > 25:   # sanity guard against bad data
            return "N/A"
        return f"{pct:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _usd(v, escape: bool = False) -> str:
    """$1,234.50 with 2 decimals, or N/A. escape=True writes \\$ so that two prices on one line
    (a range) are not read as a LaTeX formula by Streamlit's markdown."""
    try:
        s = f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "N/A"
    return s.replace("$", "\\$") if escape else s


def _range_position(info: dict) -> str:
    """Where the price sits between the 52-week low (0%) and high (100%)."""
    try:
        price = float(info.get("currentPrice") or info.get("regularMarketPrice"))
        lo = float(info.get("fiftyTwoWeekLow"))
        hi = float(info.get("fiftyTwoWeekHigh"))
        return f"{(price - lo) / (hi - lo) * 100:.0f}% of the way from low to high"
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"


def _money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _num(v, digits: int = 2) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct_field(info: dict, key: str) -> str:
    """yfinance returns margins / ROE / payout as fractions (0.1744 means 17.44%)."""
    try:
        return f"{float(info.get(key)) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _count(n) -> str:
    """Plain count (shares, employees) with K/M/B/T suffix and NO dollar sign."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.2f}{suffix}"
    return f"{n:,.0f}"


def _shares_text(info: dict) -> str:
    """yfinance's sharesOutstanding can cover one share class only (e.g. GOOGL).
    If it disagrees with marketCap / price by >5%, show the implied figure instead."""
    so = info.get("sharesOutstanding")
    cap = info.get("marketCap")
    px = info.get("currentPrice") or info.get("regularMarketPrice")
    implied = (cap / px) if cap and px else None
    if implied and (not so or abs(so / implied - 1) > 0.05):
        return f"{_count(implied)} (implied: market cap ÷ price)"
    return _count(so)


_TYPO_MAP = {
    "yeild": "yield", "yeid": "yield", "yld": "yield",
    "divident": "dividend", "divedend": "dividend", "dividned": "dividend",
    "revnue": "revenue", "revenu": "revenue",
    "earnigns": "earnings", "ernings": "earnings",
}


_LONG_VOCAB = [
    "dividend", "dividends", "revenue", "earnings", "financials", "capitalization",
    "valuation", "performance", "headquarters", "management", "executive", "leadership",
]


def _fix_typos(q: str) -> str:
    """Fix typos in intent words only. Short words use the explicit map (fuzzy matching on
    5-letter words causes false hits like field->yield); long words use difflib at 0.8."""
    fixed = []
    for w in q.split():
        if w in _TYPO_MAP:
            fixed.append(_TYPO_MAP[w])
        elif len(w) >= 7 and w not in _LONG_VOCAB:
            m = get_close_matches(w, _LONG_VOCAB, n=1, cutoff=0.8)
            fixed.append(m[0] if m else w)
        else:
            fixed.append(w)
    return " ".join(fixed)


# =========================================
# STOCK INFO BLOCK — always appended
# =========================================

def _stock_info_block(ticker: str) -> str:
    info   = get_ticker_info(ticker)
    weekly = get_weekly_data(ticker)
    if "error" in info:
        return f"\n*Could not fetch live data for {ticker}.*"

    name  = _safe(info, "longName", ticker)
    sym   = _safe(info, "symbol",   ticker)
    price = _safe(info, "currentPrice", _safe(info, "regularMarketPrice"))
    prev  = _safe(info, "previousClose")

    try:
        chg   = float(price) - float(prev)
        pct   = (chg / float(prev)) * 100
        chg_s = f"{'🟢' if chg >= 0 else '🔴'} {chg:+.2f} ({pct:+.2f}%)"
    except (TypeError, ValueError, ZeroDivisionError):
        chg_s = "N/A"

    weekly_block = ""
    if weekly:
        wk_chg = weekly["week_close"] - weekly["week_open"]
        weekly_block = (
            f"\n**📅 Weekly OHLC (Last 5 Trading Days)**\n"
            f"- Weekly Open  : {_usd(weekly['week_open'])}\n"
            f"- Weekly High  : **{_usd(weekly['week_high'])}**\n"
            f"- Weekly Low   : **{_usd(weekly['week_low'])}**\n"
            f"- Weekly Close : {_usd(weekly['week_close'])}\n"
            f"- Weekly Chg   : {'🟢' if wk_chg >= 0 else '🔴'} ${wk_chg:+.2f}\n"
            f"- Weekly Vol   : {weekly['volume']:,}\n"
        )

    return (
        f"\n\n---\n\n"
        f"**📊 {name} ({sym}) — Live Stock Data**\n\n"
        f"**💰 Price**\n"
        f"- Current Price : **{_usd(price)}**\n"
        f"- Change Today  : {chg_s}\n"
        f"- Prev Close    : {_usd(prev)}\n"
        f"- Open          : {_usd(info.get('open'))}\n"
        f"- Day Range     : {_usd(info.get('dayLow'), True)} - {_usd(info.get('dayHigh'), True)}\n"
        f"{weekly_block}\n"
        f"**📈 Fundamentals**\n"
        f"- Market Cap    : {_fmt(_safe(info,'marketCap'))}\n"
        f"- P/E (TTM)     : {_pe(info)}\n"
        f"- Div Yield     : {_dy(info)}\n"
        f"- 52W Range     : {_usd(info.get('fiftyTwoWeekLow'), True)} - {_usd(info.get('fiftyTwoWeekHigh'), True)}\n"
        f"- 52W Position  : {_range_position(info)}\n"
        f"- Revenue (TTM) : {_fmt(_safe(info,'totalRevenue'))}\n"
        f"- EPS (TTM)     : {_usd(info.get('trailingEps'))}\n"
        f"- Sector        : {_safe(info,'sector')} | {_safe(info,'industry')}\n"
        f"- Employees     : {_num(info.get('fullTimeEmployees'), 0)}"
    )


# =========================================
# INVESTMENT CALCULATOR
# =========================================

def calculate_investment_return(amount: float, ticker: str, period_days: int) -> dict:
    """Return of `amount` invested `period_days` ago.
    - current price = the same live quote the rest of the app shows (one source of truth)
    - buy price     = close on the first trading day on/after (today - period_days)
    """
    try:
        info = get_ticker_info(ticker)
        live = None
        if "error" not in info:
            live = info.get("currentPrice") or info.get("regularMarketPrice")
        try:
            live = float(live) if live else None
        except (TypeError, ValueError):
            live = None

        if period_days == 0:
            if not live or live <= 0:
                return {"error": f"Cannot get price for {ticker}"}
            sh = amount / live
            return {
                "ticker": ticker, "buy_price": live, "current_price": live,
                "shares": sh, "invested": amount, "current_value": amount,
                "profit": 0, "pct_return": 0, "is_current": True, "buy_date": None,
            }

        today  = datetime.today()
        target = today - timedelta(days=period_days)
        start  = target - timedelta(days=7)          # a week of slack for weekends / holidays
        end    = today + timedelta(days=1)           # yfinance 'end' is exclusive: +1 keeps today's bar
        hist   = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return {"error": f"No historical data for {ticker}"}

        dates = hist.index.tz_localize(None) if getattr(hist.index, "tz", None) is not None else hist.index
        on_or_after = dates >= pd.Timestamp(target.date())
        pos = int(on_or_after.argmax()) if on_or_after.any() else 0

        bp  = float(hist["Close"].iloc[pos])
        cp  = live if live else float(hist["Close"].iloc[-1])
        sh  = amount / bp
        cv  = sh * cp
        pnl = cv - amount
        pct = (pnl / amount) * 100
        return {
            "ticker": ticker, "buy_price": bp, "current_price": cp,
            "shares": sh, "invested": amount, "current_value": cv,
            "profit": pnl, "pct_return": pct, "is_current": False,
            "buy_date": dates[pos].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        return {"error": str(e)}


# =========================================
# BUILD INVESTMENT ANSWER
# =========================================

def _on(r: dict) -> str:
    """' (on 2025-10-27)' when the buy date is known."""
    return f" (on {r['buy_date']})" if r.get("buy_date") else ""


def build_investment_answer(inv: dict) -> str:
    amount = inv["amount"]
    period_days = inv.get("period_days", 365)
    period_label = inv.get("period_label", "1 year")
    is_switch = inv.get("is_switch", False)
    source = inv.get("source")

    # ── SWITCH ──
    if is_switch and "target" in inv:
        target = inv["target"]

        src_info = get_ticker_info(source) if source else {}
        src_name = src_info.get("longName", source) if source and "error" not in src_info else (source or "your holding")
        src_text = f"{src_name} ({source})" if source and src_name != source else src_name

        tgt_info = get_ticker_info(target)
        tgt_name = tgt_info.get("longName", target) if "error" not in tgt_info else target

        r = calculate_investment_return(amount, target, period_days)
        if "error" in r:
            return f"⚠️ Could not calculate for {target}: {r['error']}"

        # FIX: the "from" side now goes through the SAME calculator as the "to" side,
        # so both legs use the same price source.
        s = calculate_investment_return(amount, source, period_days) if source else None
        s_ok = bool(s) and "error" not in s

        if period_days == 0:
            lines = [
                "**🔄 Investment Switch — Current Analysis**\n",
                f"- Amount : **${amount:,.2f}**",
                f"- From   : {src_text}",
                f"- To     : **{tgt_name} ({target})**\n",
                "**If you switch right now:**",
            ]
            if s_ok:
                lines.append(f"- Sell {source} : {s['shares']:.4f} shares @ ${s['current_price']:.2f}")
            lines += [
                f"- Buy {target}  : {r['shares']:.4f} shares @ ${r['current_price']:.2f}",
                f"- Value moved  : **${r['current_value']:,.2f}**",
            ]
            return "\n".join(lines)

        arrow = "🟢" if r["profit"] >= 0 else "🔴"
        lines = [
            f"**🔄 Investment Switch — {period_label} ago**\n",
            f"- Amount switched : **${amount:,.2f}** (full amount, no split)",
            f"- From            : {src_text}",
            f"- To              : **{tgt_name} ({target})**\n",
            f"**If you had switched {period_label} ago:**",
            f"- {target} buy price{_on(r)} : ${r['buy_price']:.2f}",
            f"- Shares purchased   : {r['shares']:.4f}",
            f"- Current Price      : ${r['current_price']:.2f}",
            f"- Current Value      : **${r['current_value']:,.2f}**",
            f"- Profit / Loss      : {arrow} ${r['profit']:+,.2f} ({r['pct_return']:+.2f}%)",
        ]
        if s_ok:
            s_arrow = "🟢" if s["profit"] >= 0 else "🔴"
            lines += [
                "",
                f"**If you had stayed in {source}:**",
                f"- Current Value : **${s['current_value']:,.2f}**",
                f"- Profit / Loss : {s_arrow} ${s['profit']:+,.2f} ({s['pct_return']:+.2f}%)",
                "",
                f"**Switch vs stay : ${r['current_value'] - s['current_value']:+,.2f}**",
            ]
        lines += ["", "*Price return only — dividends are not included.*"]
        return "\n".join(lines)

    # ── REGULAR INVESTMENT ──
    tickers = inv["tickers"]
    per_ticker = inv.get("per_ticker", amount / len(tickers))

    if len(tickers) == 1:
        header = (
            f"**Investment Return Analysis — {period_label} ago**\n\n"
            f"- Amount Invested : **${amount:,.2f}**\n"
            f"- Stock           : **{tickers[0]}**\n\n"
        )
    else:
        header = (
            f"**Investment Return Analysis — {period_label} ago**\n\n"
            f"- Total Invested : **${amount:,.2f}**\n"
            f"- Split across   : {', '.join(tickers)} (${per_ticker:,.2f} each)\n\n"
        )

    lines = [header]
    tc = ti = 0.0

    for t in tickers:
        r = calculate_investment_return(per_ticker, t, period_days)
        if "error" in r:
            lines.append(f"**{t}**: Could not fetch — {r['error']}")
            continue
        arrow = "🟢" if r["profit"] >= 0 else "🔴"
        lines.extend([
            f"**{t}**",
            f"  - Buy Price{_on(r)} : ${r['buy_price']:.2f}",
            f"  - Current Price : ${r['current_price']:.2f}",
            f"  - Shares Bought : {r['shares']:.4f}",
            f"  - Current Value : **${r['current_value']:,.2f}**",
            f"  - Profit / Loss : {arrow} ${r['profit']:+,.2f} ({r['pct_return']:+.2f}%)",
            "",
        ])
        tc += r["current_value"]
        ti += per_ticker

    if len(tickers) > 1:
        tp = tc - ti
        tpc = (tp / ti) * 100
        lines.extend([
            "---", "**Portfolio Total**",
            f"  - Total Invested : ${ti:,.2f}",
            f"  - Total Value    : **${tc:,.2f}**",
            f"  - Total P&L      : {'🟢' if tp >= 0 else '🔴'} ${tp:+,.2f} ({tpc:+.2f}%)",
        ])

    lines.append("\n*Price return only — dividends are not included.*")
    return "\n".join(lines)


# =========================================
# SINGLE TICKER ANSWER
# =========================================

def build_ticker_answer(ticker: str, query: str) -> str:
    q = _fix_typos(query.strip().lower())      # FIX: "divident yeild" now reaches the dividend block
    info = get_ticker_info(ticker)
    if "error" in info:
        return f"Could not fetch data for **{ticker}**: {info['error']}"

    name = _safe(info, "longName", ticker)
    sym = _safe(info, "symbol", ticker)
    specific = ""

    # ── CEO / Leadership ──   ("head" -> "head of": "headquarters" used to land here)
    if any(w in q for w in ["ceo", "leader", "leading", "head of", "chief", "founder",
                            "who runs", "management", "officer", "executive"]):
        officers = info.get("companyOfficers", [])
        ceo = next((o for o in officers if "CEO" in o.get("title", "").upper()), None)
        ceo_name = ceo.get("name", "N/A") if ceo else "N/A"
        pay = ceo.get("totalPay") if ceo else None
        ceo_pay = _fmt(pay) if pay else "N/A"             # missing pay is not $0.00
        emp = info.get("fullTimeEmployees")
        emp_str = f"{int(emp):,}" if emp else "N/A"
        top = "\n".join([f"  - {o.get('name', 'N/A')} — {o.get('title', 'N/A')}" for o in officers[:5]])
        specific = (
            f"**{name} ({sym}) — Leadership**\n\n"
            f"- CEO           : **{ceo_name}**\n"
            f"- CEO Total Pay : {ceo_pay}\n\n"
            f"**Top Executives:**\n{top}\n\n"
            f"- Employees     : {emp_str}"
        )

    # ── Business Description ──
    elif any(w in q for w in ["what does", "business", "description", "about", "makes", "products", "services"]):
        desc = _safe(info, "longBusinessSummary", "")
        if not desc:
            desc = CRYPTO_DESCRIPTIONS.get(sym, f"{sym} is a cryptocurrency traded on major exchanges.")
        specific = f"**{name} ({sym}) — Business Overview**\n\n{desc}"

    # ── Market Cap ──
    elif any(w in q for w in ["market cap", "capitalization", "worth", "valuation"]):
        specific = (
            f"**{name} ({sym}) — Market Cap**\n\n"
            f"- Market Cap         : **{_fmt(_safe(info, 'marketCap'))}**\n"
            f"- Enterprise Value   : {_fmt(_safe(info, 'enterpriseValue'))}\n"
            f"- Shares Outstanding : {_shares_text(info)}"      # FIX: was formatted as dollars
        )

    # ── P/E Ratio ──
    elif any(w in q for w in ["p/e", "pe ratio", "price to earnings"]):
        specific = (
            f"**{name} ({sym}) — Valuation Ratios**\n\n"
            f"- Trailing P/E : **{_pe(info)}**\n"
            f"- Forward P/E  : {_num(info.get('forwardPE'))}\n"
            f"- PEG Ratio    : {_num(info.get('pegRatio'))}\n"
            f"- Price/Sales  : {_num(info.get('priceToSalesTrailing12Months'))}\n"
            f"- Price/Book   : {_num(info.get('priceToBook'))}"
        )

    # ── Dividend ──
    elif any(w in q for w in ["dividend", "yield", "payout"]):
        dy_str = _dy(info)
        rate = info.get("dividendRate")
        specific = (
            f"**{name} ({sym}) — Dividends**\n\n"
            f"- Dividend Yield : **{dy_str}**\n"
            f"- Annual Rate    : {_money(rate) if rate else 'N/A'}\n"   # FIX: was "$N/A"
            f"- Payout Ratio   : {_pct_field(info, 'payoutRatio')}"
        )

    # ── 52-Week Range ──
    elif any(w in q for w in ["52 week", "52-week", "range", "performance", "year high", "year low"]):
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        hi = info.get("fiftyTwoWeekHigh")
        lo = info.get("fiftyTwoWeekLow")
        try:
            fl = (float(price) - float(lo)) / float(lo) * 100
            fh = (float(price) - float(hi)) / float(hi) * 100
            perf = f"- From 52W Low  : {fl:+.1f}%\n- From 52W High : {fh:+.1f}%\n"
        except (TypeError, ValueError, ZeroDivisionError):
            perf = ""
        specific = (
            f"**{name} ({sym}) — 52-Week Performance**\n\n"
            f"- 52W High      : **{_money(hi)}**\n"
            f"- 52W Low       : **{_money(lo)}**\n"
            f"- Current Price : {_money(price)}\n"
            f"{perf}"
            f"- 50-Day MA     : {_money(info.get('fiftyDayAverage'))}\n"     # FIX: 2 decimals
            f"- 200-Day MA    : {_money(info.get('twoHundredDayAverage'))}"
        )

    # ── Financials ──
    elif any(w in q for w in ["financial", "revenue", "profit", "earnings", "ebitda", "income", "margin"]):
        specific = (
            f"**{name} ({sym}) — Key Financials**\n\n"
            f"- Revenue (TTM) : **{_fmt(_safe(info, 'totalRevenue'))}**\n"
            f"- Gross Profit  : {_fmt(_safe(info, 'grossProfits'))}\n"
            f"- EBITDA        : {_fmt(_safe(info, 'ebitda'))}\n"
            f"- Net Income    : {_fmt(_safe(info, 'netIncomeToCommon'))}\n"
            f"- EPS (TTM)     : {_money(info.get('trailingEps'))}\n"
            f"- Profit Margin : {_pct_field(info, 'profitMargins')}\n"          # FIX: 0.1744 -> 17.44%
            f"- ROE           : {_pct_field(info, 'returnOnEquity')}"           # FIX: 0.3056 -> 30.56%
        )

    # ── Sector / Corporate Info ──
    elif any(w in q for w in ["sector", "industry", "country", "where", "based", "headquarters"]):
        specific = (
            f"**{name} ({sym}) — Corporate Info**\n\n"
            f"- Sector   : **{_safe(info, 'sector')}**\n"
            f"- Industry : {_safe(info, 'industry')}\n"
            f"- Country  : {_safe(info, 'country')}\n"
            f"- City     : {_safe(info, 'city')}\n"
            f"- Website  : {_safe(info, 'website')}"
        )

    stock_block = _stock_info_block(sym)
    if specific:
        return specific + stock_block
    return stock_block.lstrip("\n\n---\n\n")
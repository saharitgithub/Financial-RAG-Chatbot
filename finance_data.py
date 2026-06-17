"""
finance_data.py
================
- detect_switch_query: standalone export for app.py
- Investment switch: full amount to target, NO splitting
- Period detected from query (default current/instant for switch)
- Spelling corrector: fuzzy word-by-word
- Weekly OHLC always shown
"""

import yfinance as yf
import streamlit as st
import re
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

def detect_spelling_query(query: str):
    q        = query.strip().lower()
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

    for word in words:
        upper = word.upper()
        if upper in ALL_TICKERS:
            info = get_ticker_info(upper)
            name = info.get("longName", upper) if "error" not in info else upper
            return f"**Spelling Confirmed** ✅\n\n- Ticker : **{upper}**\n- Name   : {name}\n- Status : Correct!"

    for word in words:
        if word in TICKER_SPELLING:
            correct = TICKER_SPELLING[word]
            info    = get_ticker_info(correct)
            name    = info.get("longName", correct) if "error" not in info else correct
            return f"**Spelling Correction** ✅\n\n- You wrote : `{word.upper()}`\n- Correct   : **{correct}**\n- Full Name : {name}"

    for word in words:
        matches = get_close_matches(word.upper(), ALL_TICKERS, n=1, cutoff=0.6)
        if matches:
            correct = matches[0]
            info    = get_ticker_info(correct)
            name    = info.get("longName", correct) if "error" not in info else correct
            return f"**Spelling Correction** ✅\n\n- You wrote : `{word.upper()}`\n- Correct   : **{correct}**\n- Full Name : {name}"

    for word in words:
        nm = get_close_matches(word, list(NAME_TO_TICKER.keys()), n=1, cutoff=0.6)
        if nm:
            ct   = NAME_TO_TICKER[nm[0]]
            info = get_ticker_info(ct)
            fn   = info.get("longName", ct) if "error" not in info else ct
            return f"**Spelling Correction** ✅\n\n- You wrote : `{word}`\n- Correct   : **{nm[0].title()}** (Ticker: **{ct}**)\n- Full Name : {fn}"

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
    q, found = query.strip(), []

    # Stock tickers — uppercase symbols
    for word in re.findall(r'\b[A-Z]{1,5}(?:-[A-Z])?\b', q):
        if word in ALL_TICKERS and word not in found:
            found.append(word)

    q_lower = q.lower()

    # Company names
    for name, ticker in NAME_TO_TICKER.items():
        if name in q_lower and ticker not in found:
            found.append(ticker)

    # Misspelled tickers
    for wrong, correct in TICKER_SPELLING.items():
        if wrong in q_lower and correct not in found:
            found.append(correct)

    # ── Crypto detection — MUST be before return ──
    for name, crypto_ticker in CRYPTO_NAME_MAP.items():
        if name in q_lower and crypto_ticker not in found:
            found.append(crypto_ticker)

    return found  # ← return is LAST, after all detection


def detect_ticker(query: str):
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
                # "from A to B" — g1=source, g2=target
                source_ticker = _resolve(g1)
                target_ticker = _resolve(g2)
            elif "move" in pattern and "to" in pattern and "from" in pattern[pattern.index("to"):]:
                # "move to B from A" — g1=target, g2=source
                target_ticker = _resolve(g1)
                source_ticker = _resolve(g2)
            elif "sell" in pattern:
                # "sell A and buy B" — g1=source, g2=target
                source_ticker = _resolve(g1)
                target_ticker = _resolve(g2)
            else:
                # Default: try both
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

    # Period
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
        "amount":      amount,
        "tickers":     tickers,
        "per_ticker":  amount / len(tickers),
        "period_days": period_days,
        "period_label": period_label,
        "is_switch":   False,
        "source":      None,
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
    except: return str(val)

def _pct(val):
    try:    return f"{float(val)*100:.2f}%"
    except: return "N/A"


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
    dy    = info.get("dividendYield")

    try:
        chg   = float(price) - float(prev)
        pct   = (chg / float(prev)) * 100
        chg_s = f"{'🟢' if chg >= 0 else '🔴'} {chg:+.2f} ({pct:+.2f}%)"
    except: chg_s = "N/A"

    weekly_block = ""
    if weekly:
        wk_chg   = weekly["week_close"] - weekly["week_open"]
        weekly_block = (
            f"\n**📅 Weekly OHLC (Last 5 Trading Days)**\n"
            f"- Weekly Open  : ${weekly['week_open']}\n"
            f"- Weekly High  : **${weekly['week_high']}**\n"
            f"- Weekly Low   : **${weekly['week_low']}**\n"
            f"- Weekly Close : ${weekly['week_close']}\n"
            f"- Weekly Chg   : {'🟢' if wk_chg >= 0 else '🔴'} ${wk_chg:+.2f}\n"
            f"- Weekly Vol   : {weekly['volume']:,}\n"
        )

    return (
        f"\n\n---\n\n"
        f"**📊 {name} ({sym}) — Live Stock Data**\n\n"
        f"**💰 Price**\n"
        f"- Current Price : **${price}**\n"
        f"- Change Today  : {chg_s}\n"
        f"- Prev Close    : ${prev}\n"
        f"- Open          : ${_safe(info,'open')}\n"
        f"- Day Range     : ${_safe(info,'dayLow')} – ${_safe(info,'dayHigh')}\n"
        f"{weekly_block}\n"
        f"**📈 Fundamentals**\n"
        f"- Market Cap    : {_fmt(_safe(info,'marketCap'))}\n"
        f"- P/E (TTM)     : {_safe(info,'trailingPE')}\n"
        f"- Div Yield     : {_pct(dy) if dy else 'N/A'}\n"
        f"- 52W Range     : ${_safe(info,'fiftyTwoWeekLow')} – ${_safe(info,'fiftyTwoWeekHigh')}\n"
        f"- Revenue (TTM) : {_fmt(_safe(info,'totalRevenue'))}\n"
        f"- EPS (TTM)     : ${_safe(info,'trailingEps')}\n"
        f"- Sector        : {_safe(info,'sector')} | {_safe(info,'industry')}\n"
        f"- Employees     : {int(_safe(info,'fullTimeEmployees',0)):,}"
    )


# =========================================
# INVESTMENT CALCULATOR
# =========================================

def calculate_investment_return(amount: float, ticker: str, period_days: int) -> dict:
    try:
        if period_days == 0:
            info = get_ticker_info(ticker)
            cp   = _safe(info, "currentPrice", _safe(info, "regularMarketPrice", 0))
            cp   = float(cp) if cp != "N/A" else 0
            if cp <= 0:
                return {"error": f"Cannot get price for {ticker}"}
            sh = amount / cp
            return {
                "ticker": ticker, "buy_price": cp, "current_price": cp,
                "shares": sh, "invested": amount, "current_value": amount,
                "profit": 0, "pct_return": 0, "is_current": True,
            }

        end   = datetime.today()
        start = end - timedelta(days=period_days + 5)
        hist  = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return {"error": f"No historical data for {ticker}"}
        bp  = float(hist["Close"].iloc[0])
        cp  = float(hist["Close"].iloc[-1])
        sh  = amount / bp
        cv  = sh * cp
        pnl = cv - amount
        pct = (pnl / amount) * 100
        return {
            "ticker": ticker, "buy_price": bp, "current_price": cp,
            "shares": sh, "invested": amount, "current_value": cv,
            "profit": pnl, "pct_return": pct, "is_current": False,
        }
    except Exception as e:
        return {"error": str(e)}


# =========================================
# BUILD INVESTMENT ANSWER
# =========================================

def build_investment_answer(inv: dict) -> str:
    amount       = inv["amount"]
    period_days  = inv.get("period_days", 365)
    period_label = inv.get("period_label", "1 year")
    is_switch    = inv.get("is_switch", False)
    source       = inv.get("source")

    # ── SWITCH ──
    if is_switch and "target" in inv:
        target = inv["target"]

        src_info = get_ticker_info(source) if source else {}
        src_name = src_info.get("longName", source) if source and "error" not in src_info else (source or "your holding")

        tgt_info = get_ticker_info(target)
        tgt_name = tgt_info.get("longName", target) if "error" not in tgt_info else target

        r = calculate_investment_return(amount, target, period_days)
        if "error" in r:
            return f"⚠️ Could not calculate for {target}: {r['error']}"

        if period_days == 0:
            return (
                f"**🔄 Investment Switch — Current Analysis**\n\n"
                f"- Amount         : **${amount:,.2f}**\n"
                f"- From           : {src_name}\n"
                f"- To             : **{tgt_name} ({target})**\n\n"
                f"**If you switch right now:**\n"
                f"- {target} Current Price : ${r['current_price']:.2f}\n"
                f"- Shares you get        : **{r['shares']:.4f}**\n"
                f"- Total value           : **${r['current_value']:,.2f}**\n"
            )
        else:
            arrow = "🟢" if r["profit"] >= 0 else "🔴"
            return (
                f"**🔄 Investment Switch — {period_label} ago**\n\n"
                f"- Amount switched : **${amount:,.2f}** (full amount, no split)\n"
                f"- From            : {src_name}\n"
                f"- To              : **{tgt_name} ({target})**\n\n"
                f"**Historical Result (if switched {period_label} ago):**\n"
                f"- Buy Price at entry : ${r['buy_price']:.2f}\n"
                f"- Shares purchased   : {r['shares']:.4f}\n"
                f"- Current Price      : ${r['current_price']:.2f}\n"
                f"- Current Value      : **${r['current_value']:,.2f}**\n"
                f"- Profit / Loss      : {arrow} ${r['profit']:+,.2f} ({r['pct_return']:+.2f}%)\n"
            )

    # ── REGULAR INVESTMENT ──
    tickers    = inv["tickers"]
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
            f"  - Buy Price     : ${r['buy_price']:.2f}",
            f"  - Current Price : ${r['current_price']:.2f}",
            f"  - Shares Bought : {r['shares']:.4f}",
            f"  - Current Value : **${r['current_value']:,.2f}**",
            f"  - Profit / Loss : {arrow} ${r['profit']:+,.2f} ({r['pct_return']:+.2f}%)",
            "",
        ])
        tc += r["current_value"]
        ti += per_ticker

    if len(tickers) > 1:
        tp  = tc - ti
        tpc = (tp / ti) * 100
        lines.extend([
            "---", "**Portfolio Total**",
            f"  - Total Invested : ${ti:,.2f}",
            f"  - Total Value    : **${tc:,.2f}**",
            f"  - Total P&L      : {'🟢' if tp >= 0 else '🔴'} ${tp:+,.2f} ({tpc:+.2f}%)",
        ])

    return "\n".join(lines)


# =========================================
# SINGLE TICKER ANSWER
# =========================================

def build_ticker_answer(ticker: str, query: str) -> str:
    q    = query.strip().lower()
    info = get_ticker_info(ticker)
    if "error" in info:
        return f"Could not fetch data for **{ticker}**: {info['error']}"

    name = _safe(info, "longName", ticker)
    sym  = _safe(info, "symbol",   ticker)
    specific = ""

    if any(w in q for w in ["ceo", "leader", "leading", "head", "chief", "founder",
                             "who runs", "management", "officer", "executive"]):
        officers = info.get("companyOfficers", [])
        ceo      = next((o for o in officers if "CEO" in o.get("title","").upper()), None)
        ceo_name = ceo.get("name","N/A") if ceo else "N/A"
        ceo_pay  = _fmt(ceo.get("totalPay",0)) if ceo else "N/A"
        top      = "\n".join([f"  - {o.get('name','N/A')} — {o.get('title','N/A')}" for o in officers[:5]])
        specific = (
            f"**{name} ({sym}) — Leadership**\n\n"
            f"- CEO           : **{ceo_name}**\n"
            f"- CEO Total Pay : {ceo_pay}\n\n"
            f"**Top Executives:**\n{top}\n\n"
            f"- Employees     : {int(_safe(info,'fullTimeEmployees',0)):,}"
        )
    elif any(w in q for w in ["what does", "business", "description", "about", "makes", "products", "services"]):
        desc     = _safe(info, "longBusinessSummary", "")
        # Crypto fallback descriptions
        CRYPTO_DESCRIPTIONS = {
        "BTC-USD":  "Bitcoin (BTC) is the world's first and largest cryptocurrency by market cap. It is a decentralized digital currency that operates without a central bank, using blockchain technology for peer-to-peer transactions.",
        "ETH-USD":  "Ethereum (ETH) is a decentralized blockchain platform that enables smart contracts and decentralized applications (dApps). It is the second largest cryptocurrency by market cap.",
        "DOGE-USD": "Dogecoin (DOGE) started as a meme cryptocurrency in 2013 but gained massive popularity, largely driven by social media and endorsements from Elon Musk. It uses a Proof-of-Work consensus and is widely used for tipping and micro-transactions.",
        "SOL-USD":  "Solana (SOL) is a high-performance blockchain supporting smart contracts and DeFi applications, known for extremely fast transaction speeds and low fees.",
        "BNB-USD":  "Binance Coin (BNB) is the native cryptocurrency of the Binance exchange ecosystem, used for trading fee discounts, token sales, and powering the BNB Smart Chain.",
        "XRP-USD":  "XRP is the native token of the Ripple network, designed for fast, low-cost international money transfers between financial institutions.",
        "ADA-USD":  "Cardano (ADA) is a proof-of-stake blockchain platform built with a research-driven approach, focused on security, scalability, and sustainability.",
        "DOGE-USD": "Dogecoin (DOGE) is a meme-inspired cryptocurrency that became a cultural phenomenon. Elon Musk has repeatedly endorsed it on social media, calling it 'the people's crypto'.",
        "LTC-USD":  "Litecoin (LTC) is one of the earliest altcoins, designed as a faster and lighter version of Bitcoin with lower transaction fees.",
        "AVAX-USD": "Avalanche (AVAX) is a smart contract platform competing with Ethereum, known for fast finality and a unique consensus mechanism.",
        }
    
        if not desc:
            sym  = _safe(info, "symbol", ticker)
            desc = CRYPTO_DESCRIPTIONS.get(sym, f"{sym} is a cryptocurrency traded on major exchanges. Detailed business description not available from data provider.")
    
        specific = f"**{name} ({sym}) — Business Overview**\n\n{desc}"
    elif any(w in q for w in ["market cap", "capitalization", "worth", "valuation"]):
        specific = (
            f"**{name} ({sym}) — Market Cap**\n\n"
            f"- Market Cap         : **{_fmt(_safe(info,'marketCap'))}**\n"
            f"- Enterprise Value   : {_fmt(_safe(info,'enterpriseValue'))}\n"
            f"- Shares Outstanding : {_fmt(_safe(info,'sharesOutstanding'))}"
        )
    elif any(w in q for w in ["p/e", "pe ratio", "price to earnings"]):
        specific = (
            f"**{name} ({sym}) — Valuation Ratios**\n\n"
            f"- Trailing P/E : **{_safe(info,'trailingPE')}**\n"
            f"- Forward P/E  : {_safe(info,'forwardPE')}\n"
            f"- PEG Ratio    : {_safe(info,'pegRatio')}\n"
            f"- Price/Sales  : {_safe(info,'priceToSalesTrailing12Months')}\n"
            f"- Price/Book   : {_safe(info,'priceToBook')}"
        )
    elif any(w in q for w in ["dividend", "yield", "payout"]):
        dy       = info.get("dividendYield")
        specific = (
            f"**{name} ({sym}) — Dividends**\n\n"
            f"- Dividend Yield : **{_pct(dy) if dy else 'N/A'}**\n"
            f"- Annual Rate    : ${_safe(info,'dividendRate')}\n"
            f"- Payout Ratio   : {_safe(info,'payoutRatio')}"
        )
    elif any(w in q for w in ["52 week", "52-week", "range", "performance", "year high", "year low"]):
        price = _safe(info, "currentPrice", _safe(info, "regularMarketPrice"))
        hi    = _safe(info, "fiftyTwoWeekHigh")
        lo    = _safe(info, "fiftyTwoWeekLow")
        try:
            fl   = ((float(price)-float(lo))/float(lo))*100
            fh   = ((float(price)-float(hi))/float(hi))*100
            perf = f"- From 52W Low  : +{fl:.1f}%\n- From 52W High : {fh:.1f}%"
        except: perf = ""
        specific = (
            f"**{name} ({sym}) — 52-Week Performance**\n\n"
            f"- 52W High      : **${hi}**\n- 52W Low       : **${lo}**\n"
            f"- Current Price : ${price}\n{perf}\n"
            f"- 50-Day MA     : ${_safe(info,'fiftyDayAverage')}\n"
            f"- 200-Day MA    : ${_safe(info,'twoHundredDayAverage')}"
        )
    elif any(w in q for w in ["financial", "revenue", "profit", "earnings", "ebitda", "income", "margin"]):
        specific = (
            f"**{name} ({sym}) — Key Financials**\n\n"
            f"- Revenue (TTM) : **{_fmt(_safe(info,'totalRevenue'))}**\n"
            f"- Gross Profit  : {_fmt(_safe(info,'grossProfits'))}\n"
            f"- EBITDA        : {_fmt(_safe(info,'ebitda'))}\n"
            f"- Net Income    : {_fmt(_safe(info,'netIncomeToCommon'))}\n"
            f"- EPS (TTM)     : ${_safe(info,'trailingEps')}\n"
            f"- Profit Margin : {_safe(info,'profitMargins')}\n"
            f"- ROE           : {_safe(info,'returnOnEquity')}"
        )
    elif any(w in q for w in ["sector", "industry", "country", "where", "based", "headquarters"]):
        specific = (
            f"**{name} ({sym}) — Corporate Info**\n\n"
            f"- Sector   : **{_safe(info,'sector')}**\n"
            f"- Industry : {_safe(info,'industry')}\n"
            f"- Country  : {_safe(info,'country')}\n"
            f"- City     : {_safe(info,'city')}\n"
            f"- Website  : {_safe(info,'website')}"
        )

    stock_block = _stock_info_block(sym)
    if specific:
        return specific + stock_block
    return stock_block.lstrip("\n\n---\n\n")
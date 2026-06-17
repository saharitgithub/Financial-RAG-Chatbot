"""
app.py — Financial RAG ChatBot
================================
- Switch queries: full amount to target, no splitting, LLM reasoning
- Graph queries: chart + professional LLM commentary
- Calculations: real yfinance data + LLM reasoning appended
- Ticker queries: stock data + weekly OHLC + analysis
- Comparison: full fundamentals for each ticker + LLM analysis
- No calculate_switch_investment import (removed)
"""

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pickle
import re

from utils.finance_guard import is_finance_query_llm as is_finance_query
from rag.generator import generate_answer
from embeddings.embedder import get_embedding
from rag.retriever import retrieve
from finance_data import (
    detect_tickers, detect_investment_query, detect_switch_query,
    build_investment_answer, detect_spelling_query,
    build_ticker_answer, detect_graph_period,
)
from calculations import detect_and_run_calculation
from visualization import build_3d_chart, build_comparison_chart

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="📊 FinRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# STYLES
# =========================================

st.markdown("""
<style>
.main, .stApp { background-color: #0E1117; color: white; }
h1, h2, h3, h4, p, div, label, span { color: white !important; }
section[data-testid="stSidebar"] { background-color: #111827; }
.user-msg {
    background-color: #1565C0; padding: 12px 16px;
    border-radius: 14px 14px 4px 14px; margin-bottom: 10px;
    color: white !important; font-size: 15px; line-height: 1.6;
    word-wrap: break-word; max-width: 85%; margin-left: auto;
}
.bot-msg {
    background-color: #1E1E2E; border-left: 3px solid #1E88E5;
    padding: 12px 16px; border-radius: 4px 14px 14px 14px;
    margin-bottom: 10px; color: white !important; font-size: 15px;
    line-height: 1.7; word-wrap: break-word; max-width: 95%;
}
.source-badge {
    font-size: 12px; color: #90CAF9 !important;
    margin-top: -6px; margin-bottom: 16px; padding-left: 6px;
}
.sidebar-msg {
    font-size: 12px; padding: 4px 2px; color: #bbbbbb !important;
    border-bottom: 1px solid #2a2a3a; word-wrap: break-word; line-height: 1.4;
}
.stButton > button {
    background-color: #1E88E5; color: white; font-weight: 600;
    border-radius: 8px; padding: 0.4em 1em; border: none; width: 100%;
}
.stButton > button:hover { background-color: #1565C0; }
.stChatInputContainer {
    position: fixed !important; bottom: 0 !important;
    background-color: #0E1117 !important; padding: 12px 0 !important;
    z-index: 999 !important; border-top: 1px solid #2a2a3a !important;
}
.chat-scroll-area { padding-bottom: 90px; }
hr { border-color: #2a2a3a; }
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD VECTOR STORE
# =========================================

@st.cache_resource
def load_store():
    with open("faiss_store.pkl", "rb") as f:
        return pickle.load(f)

store = load_store()

# =========================================
# SESSION STATE
# =========================================

if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "anim_playing" not in st.session_state:
    st.session_state.anim_playing = {}

# =========================================
# HELPERS
# =========================================

CALC_KEYWORDS = [
    "calculate", "compute", "kurtosis", "skewness", "volatility",
    "sharpe", "sortino", "drawdown", "beta", "cagr", "total return",
    "correlation", "probability", "confidence interval",
    "var", "value at risk", "monte carlo",
]

def is_calc_query(q: str) -> bool:
    return any(kw in q.lower() for kw in CALC_KEYWORDS)

GRAPH_KEYWORDS = ["graph", "chart", "plot", "show me", "visualize", "price history", "trend"]

def is_graph_query(q: str) -> bool:
    return any(kw in q.lower() for kw in GRAPH_KEYWORDS)

STRATEGY_KEYWORDS = [
    "should i", "worth it", "good idea", "bad idea", "recommend",
    "advice", "opinion", "better option", "wise", "smart move",
    "make sense", "is it worth",
]

def is_strategy_query(q: str) -> bool:
    return any(kw in q.lower() for kw in STRATEGY_KEYWORDS)

COMPARE_KEYWORDS = ["compare", "vs", "versus", "comparison", "which is better", "difference between"]

def is_compare_query(q: str) -> bool:
    return any(kw in q.lower() for kw in COMPARE_KEYWORDS)

PERIOD_LABEL_MAP = {
    "5d": "5 days", "1mo": "1 month", "3mo": "3 months",
    "6mo": "6 months", "ytd": "year to date", "1y": "1 year",
    "2y": "2 years", "5y": "5 years", "max": "all time",
}

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.markdown("## 📊 FinRAG")
    st.caption("Finance • Investment • Risk • Analytics")
    st.divider()
    st.markdown("#### 📜 Chat History")
    if not st.session_state.messages:
        st.caption("No chats yet.")
    else:
        for msg in st.session_state.messages[-15:]:
            icon    = "🧑" if msg["role"] == "user" else "🤖"
            preview = msg["content"][:50].replace("\n", " ")
            st.markdown(
                f'<div class="sidebar-msg">{icon} {preview}...</div>',
                unsafe_allow_html=True,
            )
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑 Clear"):
            st.session_state.messages     = []
            st.session_state.anim_playing = {}
            st.rerun()
    with c2:
        if st.button("❌ Exit"):
            st.stop()

# =========================================
# MAIN AREA
# =========================================

st.markdown(
    "<h2 style='margin-bottom:4px;'>💹 Financial Intelligence Assistant using Retrieval Augmented Generation and LLMs</h2>",
    unsafe_allow_html=True,
)
st.caption("Ask about stocks, finance, formulas, investment returns, or comparisons")
st.divider()

st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)

for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg"><b>🧑 You:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-msg"><b>🤖 Assistant:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("source"):
            st.markdown(
                f'<div class="source-badge">{msg["source"]}</div>',
                unsafe_allow_html=True,
            )
        if msg.get("chart_tickers"):
            ct     = msg["chart_tickers"]
            period = msg.get("chart_period", "1y")
            lbl    = (
                f"📉 {ct[0]} — {PERIOD_LABEL_MAP.get(period, period)} Chart"
                if len(ct) == 1
                else f"📊 {' vs '.join(ct)} — {PERIOD_LABEL_MAP.get(period, period)} Comparison"
            )
            st.markdown(f"**{lbl}**")
            with st.spinner("Loading chart..."):
                fig = (
                    build_3d_chart(ct[0], period=period)
                    if len(ct) == 1
                    else build_comparison_chart(ct, period=period)
                )
            st.plotly_chart(
                fig, use_container_width=True,
                config={"displayModeBar": False},
                key=f"chart_{idx}",
            )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# CHAT INPUT
# =========================================

query = st.chat_input("Ask anything — stocks, finance, formulas, calculations...")

# =========================================
# CHATBOT LOGIC
# =========================================

if query:

    answer        = ""
    source_msg    = ""
    show_source   = False
    chart_tickers = []
    chart_period  = "1y"

    st.session_state.messages.append({"role": "user", "content": query})

    # ── Step 1: Spelling check ──
    spelling = detect_spelling_query(query)
    if spelling:
        answer      = spelling
        source_msg  = "📖 *Source: Ticker / finance term database.*"
        show_source = True

    else:
        detected     = detect_tickers(query)
        chart_period = detect_graph_period(query)

        # ── Step 2: Graph query ──
        if is_graph_query(query) and detected:
            chart_tickers = detected
            period_label  = PERIOD_LABEL_MAP.get(chart_period, chart_period)

            # Fetch data for ALL detected tickers
            all_data = []
            for t in detected:
                with st.spinner(f"📡 Fetching {t}..."):
                    all_data.append(build_ticker_answer(t, query))
            combined_data = "\n\n---\n\n".join(all_data)

            # Single ticker chart commentary
            if len(detected) == 1:
                with st.spinner("🤖 Generating chart analysis..."):
                    answer = generate_answer(
                        query,
                        f"User wants a {period_label} price chart for {detected[0]}.\n\n"
                        f"Live market data:\n{combined_data}\n\n"
                        f"Provide professional technical analysis:\n"
                        f"1. Current price and {period_label} trend direction\n"
                        f"2. Key support and resistance levels from 52W range\n"
                        f"3. Volume and momentum\n"
                        f"4. Short-term outlook\n\n"
                        f"IMPORTANT: The 3D animated chart is already displayed below. "
                        f"Do NOT say you cannot show graphs. Start directly with the analysis.",
                    )
            else:
                # Multi-ticker comparison chart + analysis
                with st.spinner("🤖 Generating comparison analysis..."):
                    answer = generate_answer(
                        query,
                        f"User wants a {period_label} comparison chart for {' vs '.join(detected)}.\n\n"
                        f"Live market data:\n{combined_data}\n\n"
                        f"Provide detailed comparison:\n"
                        f"1. Price performance over {period_label} for each stock\n"
                        f"2. Which performed better and by how much\n"
                        f"3. Fundamental comparison (market cap, P/E, revenue)\n"
                        f"4. Risk comparison and outlook\n\n"
                        f"IMPORTANT: The 3D comparison chart is displayed below. "
                        f"Do NOT say you cannot show graphs.",
                    )
            source_msg  = "📊🤖 *Source: Live market data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 3: Switch query ("invest X from A to B") ──
        elif detect_switch_query(query):
            sw = detect_switch_query(query)

            with st.spinner(f"🔄 Calculating switch to {sw['target']}..."):
                # Build investment answer using switch data
                inv_data = {
                    "amount":       sw["amount"],
                    "target":       sw["target"],
                    "source":       sw["source"],
                    "is_switch":    True,
                    "period_days":  sw["period_days"],
                    "period_label": sw["period_label"],
                }
                calc_result = build_investment_answer(inv_data)

            # Fetch target stock data
            with st.spinner(f"📡 Fetching {sw['target']} data..."):
                target_data = build_ticker_answer(sw["target"], query)

            chart_tickers = [sw["target"]]

            # LLM reasoning on the switch
            with st.spinner("🤖 Generating investment analysis..."):
                llm_part = generate_answer(
                    query,
                    f"Investment switch calculation result:\n{calc_result}\n\n"
                    f"Target stock live data:\n{target_data}\n\n"
                    f"Provide professional analysis:\n"
                    f"1. Assessment of this switch decision\n"
                    f"2. Historical performance of the target stock\n"
                    f"3. Risk factors to consider\n"
                    f"4. Whether this switch makes financial sense\n"
                    f"5. Recommendation",
                )

            answer      = f"{calc_result}\n\n{target_data}\n\n---\n\n**💡 Investment Analysis:**\n{llm_part}"
            source_msg  = "📊🤖 *Source: Live market data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 4: Quantitative calculations ──
        elif is_calc_query(query) and detected:
            with st.spinner("📊 Running calculation..."):
                calc_result = detect_and_run_calculation(query, detected)

            if calc_result:
                chart_tickers = [detected[0]]

                # Append LLM reasoning to calculation result
                with st.spinner("🤖 Generating analysis..."):
                    llm_part = generate_answer(
                        query,
                        f"Quantitative calculation result for {', '.join(detected)}:\n\n"
                        f"{calc_result}\n\n"
                        f"Provide professional financial interpretation:\n"
                        f"1. What this metric means for investors\n"
                        f"2. How this compares to industry benchmarks\n"
                        f"3. Risk implications\n"
                        f"4. Actionable insights",
                    )
                answer      = f"{calc_result}\n\n---\n\n**💡 Analysis & Insights:**\n{llm_part}"
                source_msg  = "📊🤖 *Source: Real-time calculations (yfinance) + AI analysis.*"
                show_source = True
            else:
                # Fallback to ticker info if calc not detected
                with st.spinner(f"📡 Fetching {detected[0]}..."):
                    yf_answer = build_ticker_answer(detected[0], query)
                chart_tickers = [detected[0]]
                with st.spinner("🤖 Generating analysis..."):
                    llm_part = generate_answer(query, f"Data:\n{yf_answer}")
                answer      = f"{yf_answer}\n\n---\n\n**🤖 Analysis:**\n{llm_part}"
                source_msg  = "📊🤖 *Source: Live market data + AI analysis.*"
                show_source = True

        # ── Step 5: Comparison query ──
        elif is_compare_query(query) and len(detected) >= 2:
            chart_tickers = detected
            summaries     = []
            for t in detected:
                with st.spinner(f"📡 Fetching {t}..."):
                    summaries.append(build_ticker_answer(t, query))
            combined = "\n\n---\n\n".join(summaries)

            with st.spinner("🤖 Generating detailed comparison..."):
                answer = generate_answer(
                    query,
                    f"Compare these stocks: {', '.join(detected)}\n\n"
                    f"Live market data for each:\n{combined}\n\n"
                    f"Provide a comprehensive comparison:\n"
                    f"1. Current price and recent performance of each\n"
                    f"2. Valuation comparison (P/E, market cap, price/book)\n"
                    f"3. Financial health (revenue, earnings, margins)\n"
                    f"4. Growth prospects and competitive position\n"
                    f"5. Risk factors for each\n"
                    f"6. Which is the better investment right now and why\n"
                    f"Be specific, data-driven, and professional.",
                )
            source_msg  = "📊🤖 *Source: Live market data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 6: Strategy query ──
        elif is_strategy_query(query) and detected:
            summaries = []
            for t in detected:
                with st.spinner(f"📡 Fetching {t}..."):
                    summaries.append(build_ticker_answer(t, query))
            combined      = "\n\n---\n\n".join(summaries)
            chart_tickers = detected

            with st.spinner("🤖 Generating strategic analysis..."):
                answer = generate_answer(
                    query,
                    f"Strategic investment question about: {', '.join(detected)}\n\n"
                    f"Live market data:\n{combined}\n\n"
                    f"Provide professional strategic analysis and recommendation.",
                )
            source_msg  = "📊🤖 *Source: Live market data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 7: Regular investment query ──
        elif detect_investment_query(query):
            inv = detect_investment_query(query)
            with st.spinner("📈 Calculating investment returns..."):
                calc_result = build_investment_answer(inv)

            chart_tickers = inv["tickers"]

            # LLM commentary on the result
            with st.spinner("🤖 Generating investment commentary..."):
                llm_part = generate_answer(
                    query,
                    f"Investment calculation:\n{calc_result}\n\n"
                    f"Provide brief professional commentary:\n"
                    f"1. Was this a good investment decision?\n"
                    f"2. Risk assessment\n"
                    f"3. Forward-looking recommendation",
                )
            answer      = f"{calc_result}\n\n---\n\n**💡 Commentary:**\n{llm_part}"
            source_msg  = "📊🤖 *Source: Live historical data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 8: Single ticker info ──
        elif len(detected) == 1:
            ticker        = detected[0]
            chart_tickers = [ticker]

            with st.spinner(f"📡 Fetching live data for {ticker}..."):
                yf_answer = build_ticker_answer(ticker, query)

            with st.spinner("🤖 Generating analysis..."):
                llm_part = generate_answer(
                    query,
                    f"User is asking about {ticker} stock.\n"
                    f"Live market data:\n{yf_answer}\n\n"
                    f"Provide:\n"
                    f"1. Direct answer to the user's specific question\n"
                    f"2. Financial analysis and current market outlook\n"
                    f"3. Key risks and opportunities\n"
                    f"4. Buy / Hold / Sell consideration with reasoning\n"
                    f"Be concise, data-driven, and professional.",
                )
            answer      = f"{yf_answer}\n\n---\n\n**🤖 Analysis & Insights:**\n{llm_part}"
            source_msg  = "📊🤖 *Source: Live market data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 9: Multi-ticker without explicit compare keyword ──
        elif len(detected) > 1:
            chart_tickers = detected
            summaries     = []
            for t in detected:
                with st.spinner(f"📡 Fetching {t}..."):
                    summaries.append(build_ticker_answer(t, query))
            combined = "\n\n---\n\n".join(summaries)

            with st.spinner("🤖 Generating analysis..."):
                llm_part = generate_answer(
                    query,
                    f"Multiple tickers detected: {', '.join(detected)}\n\n"
                    f"Live data:\n{combined}\n\n"
                    f"Answer the user's question and provide relevant analysis.",
                )
            answer      = f"{combined}\n\n---\n\n**🤖 Analysis:**\n{llm_part}"
            source_msg  = "📊🤖 *Source: Live market data (yfinance) + AI analysis.*"
            show_source = True

        # ── Step 10: Domain check → RAG + LLM ──
        elif not is_finance_query(query):
            answer = (
                "I am a finance-focused assistant and can only help with "
                "finance, investment, economics, accounting, statistics, "
                "and related analytical topics."
            )

        else:
            with st.spinner("🔎 Searching financial documents..."):
                query_emb = get_embedding(query)
                results   = retrieve(store, query_emb, k=5)

            pdf_found     = bool(results)
            final_context = (
                "According to the uploaded financial PDFs:\n\n" + "\n".join(results)
                if pdf_found
                else "No PDF context found. Answer using financial knowledge only."
            )

            recent_history = st.session_state.messages[-6:]
            history_text   = ""
            for m in recent_history:
                role          = "User" if m["role"] == "user" else "Assistant"
                history_text += f"{role}: {m['content']}\n"

            with st.spinner("🤖 Generating response..."):
                answer = generate_answer(
                    query=query,
                    context=final_context,
                    chat_history=history_text,
                )

            if pdf_found and len(results) >= 3:
                source_msg  = "📚 *Source: Retrieved from uploaded financial PDFs.*"
                show_source = True
            elif pdf_found:
                source_msg  = "📚🤖 *Source: Retrieved from PDFs + enhanced by AI.*"
                show_source = True
            else:
                source_msg  = "🤖 *Source: Answer generated by AI.*"
                show_source = True

    # ── Save + rerun ──
    st.session_state.messages.append({
        "role":          "assistant",
        "content":       answer,
        "source":        source_msg if show_source else "",
        "chart_tickers": chart_tickers,
        "chart_period":  chart_period,
    })

    st.rerun()
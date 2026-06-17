"""
visualization.py
=================
- Dynamic period: uses whatever period is passed (1mo, 6mo, 1y, 5y etc.)
- Animation works: starts from empty, draws progressively
- Play/Pause via Plotly updatemenus
- No toolbar buttons
- Single ticker: animated 3D line
- Multi ticker: animated comparison
"""

import plotly.graph_objects as go
import numpy as np
import yfinance as yf

TICKER_COLORS = [
    "#00E5FF",  # cyan
    "#FFD600",  # yellow
    "#76FF03",  # green
    "#FF6D00",  # orange
    "#FF4081",  # pink
]

PERIOD_LABELS = {
    "5d":  "5 Days",
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "ytd": "Year to Date",
    "1y":  "1 Year",
    "2y":  "2 Years",
    "5y":  "5 Years",
    "max": "All Time",
}


# =========================================
# FALLBACK PRICE HISTORY FUNCTION
# =========================================

def get_price_history(ticker: str, period: str = "5y"):
    """Fetch historical price data for charts - direct yfinance call"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist if not hist.empty else None
    except Exception:
        return None


def build_3d_chart(ticker: str, period: str = "5y") -> go.Figure:
    """Single ticker animated 3D chart with dynamic period."""
    return _build_single(ticker, period)


def build_comparison_chart(tickers: list, period: str = "5y") -> go.Figure:
    """Multi-ticker animated comparison 3D chart with dynamic period."""
    return _build_comparison(tickers, period)


# =========================================
# SINGLE TICKER — animated
# =========================================

def _build_single(ticker: str, period: str = "5y") -> go.Figure:
    hist = get_price_history(ticker, period=period)

    if hist is None or hist.empty:
        return _empty_fig(f"No data for {ticker}")

    hist            = hist.reset_index()
    hist["DateStr"] = hist["Date"].dt.strftime("%b %d, %Y")
    hist["Year"]    = hist["Date"].dt.year

    prices = hist["Close"].values.astype(float)
    n      = len(prices)
    x_vals = np.linspace(0, 100, n)
    y_vals = np.zeros(n)
    z_vals = prices
    color  = TICKER_COLORS[0]

    # X axis labels — year for long periods, month for short
    axis_label = "Year"
    tick_labels = {}

    if period in ("5d", "1mo"):
        # Daily ticks
        axis_label = "Date"
        for i, row in hist.iterrows():
            norm_idx = (i / n) * 100
            tick_labels[round(norm_idx, 1)] = row["DateStr"]
    elif period in ("3mo", "6mo", "ytd", "1y"):
        # Monthly ticks
        axis_label = "Month"
        hist["Month"] = hist["Date"].dt.strftime("%b %Y")
        seen = set()
        for i, row in hist.iterrows():
            label = row["Month"]
            if label not in seen:
                norm_idx = (i / n) * 100
                tick_labels[round(norm_idx, 1)] = label
                seen.add(label)
    else:
        # Year ticks for 2y, 5y, max
        axis_label = "Year"
        for yr in sorted(hist["Year"].unique()):
            idx      = hist[hist["Year"] == yr].index[0]
            norm_idx = (idx / n) * 100
            tick_labels[round(norm_idx, 1)] = str(yr)

    # Animation frames
    step        = max(1, n // 120)
    frame_steps = list(range(1, n, step))
    if not frame_steps or frame_steps[-1] != n - 1:
        frame_steps.append(n - 1)

    frames = []
    for fi in frame_steps:
        xi = x_vals[:fi + 1]
        yi = y_vals[:fi + 1]
        zi = z_vals[:fi + 1]
        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=xi, y=yi, z=zi,
                    mode="lines",
                    line=dict(color=color, width=5),
                    marker=dict(size=0),
                ),
                go.Scatter3d(
                    x=[xi[-1]], y=[yi[-1]], z=[zi[-1]],
                    mode="markers+text",
                    marker=dict(size=7, color="#FFFFFF"),
                    text=[f"  ${zi[-1]:.0f}"],
                    textfont=dict(color="white", size=11),
                ),
            ],
            name=str(fi),
        ))

    initial_line = go.Scatter3d(
        x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
        mode="lines+markers",
        line=dict(color=color, width=5),
        marker=dict(size=6, color="#FFFFFF"),
        name=ticker,
        hovertext=[hist["DateStr"].iloc[0]],
        hovertemplate="<b>%{hovertext}</b><br>$%{z:.2f}<extra></extra>",
    )

    period_label = PERIOD_LABELS.get(period, period)

    layout = go.Layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=10, r=10, t=50, b=60),
        title=dict(
            text=f"<b>{ticker}</b> — {period_label} Price History",
            font=dict(color="white", size=15),
            x=0.5, y=0.97,
        ),
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.2, y=-1.8, z=0.5),
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=0.3, z=1.0),
            xaxis=dict(
                title=dict(text=axis_label, font=dict(color="white", size=11)),
                tickvals=list(tick_labels.keys()),
                ticktext=list(tick_labels.values()),
                tickfont=dict(color="white", size=10),
                gridcolor="#2a2a2a",
                backgroundcolor="#0D0D1A",
                showbackground=True,
                zerolinecolor="#444",
                showspikes=False,
                range=[0, 100],
            ),
            yaxis=dict(
                showticklabels=False, showgrid=False,
                backgroundcolor="#0E1117", showbackground=False,
                showspikes=False, visible=False,
            ),
            zaxis=dict(
                title=dict(text="Price (USD)", font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=10),
                gridcolor="#2a2a2a",
                backgroundcolor="#0D0D1A",
                showbackground=True,
                zerolinecolor="#444",
                showspikes=False,
            ),
            bgcolor="#0E1117",
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(14,17,23,0.8)",
            bordercolor="#2a2a3a", borderwidth=1,
            x=0.01, y=0.99,
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.01, y=-0.08,
            xanchor="left", yanchor="top",
            bgcolor="#1565C0",
            bordercolor="#0d47a1",
            font=dict(color="white", size=13),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=30, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate",
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
    )

    return go.Figure(data=[initial_line], layout=layout, frames=frames)


# =========================================
# MULTI-TICKER — animated comparison
# =========================================

def _build_comparison(tickers: list, period: str = "5y") -> go.Figure:

    all_hist = {}
    for t in tickers:
        h = get_price_history(t, period=period)
        if h is not None and not h.empty:
            all_hist[t] = h.reset_index()

    if not all_hist:
        return _empty_fig("No data available")

    traces   = []
    frames   = []
    prepared = {}
    max_len  = max(len(h) for h in all_hist.values())

    for i, (ticker, hist) in enumerate(all_hist.items()):
        hist["DateStr"] = hist["Date"].dt.strftime("%b %d, %Y")
        hist["Year"]    = hist["Date"].dt.year
        prices = hist["Close"].values.astype(float)
        n      = len(prices)
        x_vals = np.linspace(0, 100, n)
        y_vals = np.full(n, i * 0.4)
        z_vals = prices
        color  = TICKER_COLORS[i % len(TICKER_COLORS)]

        prepared[ticker] = {"x": x_vals, "y": y_vals, "z": z_vals, "color": color}

        traces.append(go.Scatter3d(
            x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
            mode="lines+markers",
            line=dict(color=color, width=5),
            marker=dict(size=5, color="#FFFFFF"),
            name=ticker,
        ))

    first_hist         = list(all_hist.values())[0]
    first_hist["Year"] = first_hist["Date"].dt.year
    n_first            = len(first_hist)
    year_ticks         = {}
    for yr in sorted(first_hist["Year"].unique()):
        idx      = first_hist[first_hist["Year"] == yr].index[0]
        norm_idx = (idx / n_first) * 100
        year_ticks[round(norm_idx, 1)] = str(yr)

    step        = max(1, max_len // 120)
    frame_steps = list(range(1, max_len, step))
    if not frame_steps or frame_steps[-1] != max_len - 1:
        frame_steps.append(max_len - 1)

    for fi in frame_steps:
        frame_data = []
        for ticker, data in prepared.items():
            max_idx = min(fi, len(data["x"]) - 1)
            frame_data.append(go.Scatter3d(
                x=data["x"][:max_idx + 1],
                y=data["y"][:max_idx + 1],
                z=data["z"][:max_idx + 1],
                mode="lines",
                line=dict(color=data["color"], width=5),
                name=ticker,
            ))
        frames.append(go.Frame(data=frame_data, name=str(fi)))

    period_label = PERIOD_LABELS.get(period, period)

    layout = go.Layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=10, r=10, t=50, b=60),
        title=dict(
            text=f"<b>{' vs '.join(tickers)}</b> — {period_label} Comparison",
            font=dict(color="white", size=15),
            x=0.5, y=0.97,
        ),
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.2, y=-1.8, z=0.5),
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=0.5, z=1.0),
            xaxis=dict(
                title=dict(text="Year", font=dict(color="white", size=11)),
                tickvals=list(year_ticks.keys()),
                ticktext=list(year_ticks.values()),
                tickfont=dict(color="white", size=10),
                gridcolor="#2a2a2a",
                backgroundcolor="#0D0D1A",
                showbackground=True,
                zerolinecolor="#444",
                showspikes=False,
                range=[0, 100],
            ),
            yaxis=dict(
                showticklabels=False, showgrid=False,
                backgroundcolor="#0E1117", showbackground=False,
                showspikes=False, visible=False,
            ),
            zaxis=dict(
                title=dict(text="Price (USD)", font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=10),
                gridcolor="#2a2a2a",
                backgroundcolor="#0D0D1A",
                showbackground=True,
                zerolinecolor="#444",
                showspikes=False,
            ),
            bgcolor="#0E1117",
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(14,17,23,0.8)",
            bordercolor="#2a2a3a", borderwidth=1,
            x=0.01, y=0.99,
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.01, y=-0.08,
            xanchor="left", yanchor="top",
            bgcolor="#1565C0",
            bordercolor="#0d47a1",
            font=dict(color="white", size=13),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=30, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate",
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
    )

    return go.Figure(data=traces, layout=layout, frames=frames)


# =========================================
# EMPTY FIGURE
# =========================================

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="white"),
    )
    fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    return fig
"""
Pragyam v7.0.5 — Reusable UI components: metric cards, signal badges, headers, section headers.

UI — Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import html as html_mod

import streamlit as st


# ── SVG Icons (inline, no external deps) — with ARIA labels for accessibility

ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "chevron-right": '<svg aria-label="Expand icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>',
    "download":   '<svg aria-label="Download icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "briefcase":  '<svg aria-label="Portfolio icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "compass":    '<svg aria-label="Regime icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
}


def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a styled section header with icon, title, and optional description.

    Args:
        title: Section title (rendered uppercase).
        description: Optional one-line description below title.
        icon: Key from ICONS dict.
        accent: CSS color class — "", "cyan", "emerald", "violet", "rose".
    """
    svg = ICONS.get(icon, ICONS["chart"])
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text">'
        f'<h3>{html_mod.escape(title)}</h3>'
        f'{desc_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_gap() -> None:
    """Insert vertical spacing between major sections."""
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
) -> None:
    """Render a terminal-styled metric card with optional tooltip.

    Args:
        label: Card label (rendered uppercase).
        value: Primary metric value.
        subtext: Optional secondary description below value.
        color_class: Semantic color — "neutral", "success", "danger", "warning", "info", "violet".
        tooltip: Optional hover explanation text.
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{html_mod.escape(label)}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f'{f"<div class=\"sub-metric\">{html_mod.escape(subtext)}</div>" if subtext else ""}'
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_conviction_signal(
    symbol: str,
    conviction: float,
    rsi: str = "—",
    osc: str = "—",
    zscore: str = "—",
    ma: str = "—",
) -> None:
    """Render a conviction signal row for position guide.

    Args:
        symbol: Stock symbol.
        conviction: Conviction score (0-100).
        rsi: RSI value formatted.
        osc: Oscillator value formatted.
        zscore: Z-score value formatted.
        ma: Moving average alignment formatted.
    """
    if conviction >= 65:
        signal_class = "buy"
        signal_text = "Strong Buy"
        emoji = "🟢"
        conviction_bar_width = min(100, conviction)
        conviction_bar_color = "var(--emerald)"
    elif conviction >= 50:
        signal_class = "buy"
        signal_text = "Buy"
        emoji = "🟩"
        conviction_bar_width = min(100, conviction)
        conviction_bar_color = "var(--emerald-bright)"
    elif conviction >= 35:
        signal_class = "hold"
        signal_text = "Hold"
        emoji = "🟡"
        conviction_bar_width = min(100, conviction)
        conviction_bar_color = "var(--amber)"
    else:
        signal_class = "sell"
        signal_text = "Caution"
        emoji = "🔴"
        conviction_bar_width = min(100, conviction)
        conviction_bar_color = "var(--rose)"

    st.markdown(
        f"""
        <div class="signal-row" style="display:flex; align-items:center; gap:0.75rem; padding:0.75rem 0; border-bottom:1px solid var(--border-subtle); position:relative; overflow:hidden;">
            <div style="position:absolute; left:0; top:0; bottom:0; width:{conviction_bar_width} * 0.3%; background: linear-gradient(90deg, {conviction_bar_color}08, {conviction_bar_color}03); pointer-events:none;"></div>
            <div style="flex:1; font-family:var(--data); font-weight:600; color:var(--ink-primary); position:relative; z-index:1;">{html_mod.escape(symbol)}</div>
            <div style="font-family:var(--data); font-size:0.7rem; color:var(--ink-tertiary); position:relative; z-index:1;">
                <span style="color:var(--ink-secondary); font-weight:500;">RSI</span> {rsi}
            </div>
            <div style="font-family:var(--data); font-size:0.7rem; color:var(--ink-tertiary); position:relative; z-index:1;">
                <span style="color:var(--ink-secondary); font-weight:500;">Osc</span> {osc}
            </div>
            <div style="font-family:var(--data); font-size:0.7rem; color:var(--ink-tertiary); position:relative; z-index:1;">
                <span style="color:var(--ink-secondary); font-weight:500;">Z</span> {zscore}
            </div>
            <div style="font-family:var(--data); font-size:0.7rem; color:var(--ink-tertiary); position:relative; z-index:1;">
                <span style="color:var(--ink-secondary); font-weight:500;">MA</span> {ma}
            </div>
            <div style="position:relative; z-index:1;">
                <div style="width:60px; height:4px; background:var(--bg-elevated); border-radius:2px; overflow:hidden;">
                    <div style="width:{conviction_bar_width}%; height:100%; background:{conviction_bar_color}; border-radius:2px; transition:width 0.6s cubic-bezier(0.16, 1, 0.3, 1);"></div>
                </div>
            </div>
            <div style="font-family:var(--data); font-size:0.75rem; font-weight:700; color:var(--ink-primary); min-width:40px; text-align:right; position:relative; z-index:1;">{int(conviction)}</div>
            <div class="signal-pill {signal_class}" style="display:inline-flex; align-items:center; gap:0.3rem; padding:0.3rem 0.75rem; border-radius:20px; font-size:0.72rem; font-weight:600; position:relative; z-index:1;">
                {emoji} {signal_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, tagline: str) -> None:
    """Render the terminal masthead."""
    st.markdown(
        f'<div class="premium-header">'
        f"<h1>{html_mod.escape(title)}</h1>"
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "cyan") -> None:
    """Render an info box."""
    st.markdown(
        f'<div class="info-box">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_system_card(
    title: str,
    description: str,
    specs: list[tuple[str, str]],
    card_class: str = "portfolio",
) -> None:
    """Render a system feature card for landing page.

    Args:
        title: Card title.
        description: Card description.
        specs: List of (label, value) tuples for specifications.
        card_class: CSS class — "portfolio", "regime", "strategies".
    """
    spec_html = "".join(
        f'<span>{html_mod.escape(label)}</span> {html_mod.escape(value)}<br>'
        for label, value in specs
    )

    st.markdown(
        f"""
        <div class='system-card {html_mod.escape(card_class)}'>
            <h3>{html_mod.escape(title)}</h3>
            <p>{html_mod.escape(description)}</p>
            <div class='spec'>{spec_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_box(title: str, content: str) -> None:
    """Render a themed alert/warning box."""
    st.markdown(
        f"""
        <div class="warning-box">
            <div class="icon"></div>
            <div>
                <div class="title">{html_mod.escape(title)}</div>
                <div class="content">{html_mod.escape(content)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_skeleton(height: int = 280) -> None:
    """Render a loading skeleton placeholder for charts.

    Provides visual feedback while chart data is being computed.
    Uses CSS shimmer animation for a polished loading experience.
    """
    st.markdown(
        f'<div class="skeleton-chart" style="min-height:{height}px;">'
        f'<div class="skeleton-line skeleton-pulse"></div>'
        f'<div class="skeleton-block skeleton-pulse"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_collapsible_section(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
    default_open: bool = False,
):
    """Render a collapsible section header with chevron toggle.

    Returns a context manager that yields content when expanded.
    Uses Streamlit's container + checkbox pattern for state management.

    Args:
        title: Section title (rendered uppercase).
        description: Optional one-line description below title.
        icon: Key from ICONS dict.
        accent: CSS color class — "", "cyan", "emerald", "violet", "rose".
        default_open: Whether section starts expanded or collapsed.
    """
    svg = ICONS.get(icon, ICONS["chart"])
    section_id = f"collapsible_{html_mod.escape(title.lower().replace(' ', '_'))}"
    is_open = st.checkbox(
        f"toggle_{section_id}",
        value=default_open,
        label_visibility="collapsed",
        key=f"_{section_id}_state",
    )

    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    open_class = "open" if is_open else ""

    st.markdown(
        f'<div class="collapsible-section {open_class}" id="{section_id}">'
        f'<div class="collapsible-header" data-target="{section_id}">'
        f'<span class="chevron">{ICONS["chevron-right"]}</span>'
        f'<div class="{hdr_class}" style="margin:0;padding:0;border:none;flex:1;">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text">'
        f'<h3>{html_mod.escape(title)}</h3>'
        f'{desc_html}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="collapsible-body">'
        f'<div class="collapsible-body-inner">',
        unsafe_allow_html=True,
    )

    return is_open


def render_collapsible_section_close() -> None:
    """Close a collapsible section opened by render_collapsible_section."""
    st.markdown(
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_theme_toggle() -> None:
    """Render a fixed-position theme toggle button (dark/light mode).

    Uses JavaScript to toggle data-theme attribute on the html element.
    Persists preference in localStorage.
    """
    from streamlit.components.v1 import html as st_html
    st_html(
        """
        <div class="theme-toggle" id="theme-toggle" title="Toggle theme" onclick="toggleTheme()">
            <svg id="theme-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
            <svg id="theme-icon-moon" style="display:none" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
            <span id="theme-label">Light</span>
        </div>
        <script>
        (function() {
            var html = document.documentElement;
            var saved = localStorage.getItem('pragyam-theme');
            var theme = saved || 'dark';
            html.setAttribute('data-theme', theme);
            updateUI(theme);

            window.toggleTheme = function() {
                var current = html.getAttribute('data-theme');
                var next = current === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', next);
                localStorage.setItem('pragyam-theme', next);
                updateUI(next);
            };

            function updateUI(theme) {
                var sun = document.getElementById('theme-icon-sun');
                var moon = document.getElementById('theme-icon-moon');
                var label = document.getElementById('theme-label');
                if (theme === 'light') {
                    if (sun) sun.style.display = 'none';
                    if (moon) moon.style.display = 'block';
                    if (label) label.textContent = 'Dark';
                } else {
                    if (sun) sun.style.display = 'block';
                    if (moon) moon.style.display = 'none';
                    if (label) label.textContent = 'Light';
                }
            }
        })();
        </script>
        """,
        height=0,
    )


def render_export_button_row(
    label: str = "Export",
    icon: str = "download",
    data: bytes = b"",
    file_name: str = "export.csv",
    mime: str = "text/csv",
) -> None:
    """Render a right-aligned export button with icon.

    Args:
        label: Button label text.
        icon: Key from ICONS dict (defaults to "download").
        data: Binary data to export.
        file_name: Default download filename.
        mime: MIME type for the download.
    """
    svg = ICONS.get(icon, ICONS["download"])
    st.markdown(
        f'<div class="export-btn-row">'
        f'{svg}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        label=f"{svg}  {label}",
        data=data,
        file_name=file_name,
        mime=mime,
        key=f"export_{file_name}",
    )


def render_interpretation_card(
    title: str,
    body: str,
    color: str = "neutral",
) -> None:
    """Render a state-aware interpretation card — terminal readout style.

    Args:
        title: Short state label (e.g. "NEUTRAL", "STRONG OVERSOLD").
        body: One-paragraph explanation.
        color: Semantic color — "neutral", "success", "danger", "warning", "info".
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(title)}</div>'
        f'<div class="interp-body">{html_mod.escape(body)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

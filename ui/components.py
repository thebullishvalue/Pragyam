"""
PRAGYAM — UI Components
══════════════════════════════════════════════════════════════════════════════

Reusable UI primitives — metric cards, headers, section headers, system cards,
interpretation cards, and key/value tables — in the Obsidian Quant Terminal
design language.

Author: @thebullishvalue
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
    "rocket":     '<svg aria-label="Strong Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-3 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4.5c1.62-1.63 5-2.5 5-2.5"/><path d="M12 15v5s3.03-.55 4.5-2c1.63-1.62 2.5-5 2.5-5"/></svg>',
    "trending-up": '<svg aria-label="Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "arrow-up-right": '<svg aria-label="Weak Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>',
    "arrow-down-right": '<svg aria-label="Weak Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/></svg>',
    "arrow-up": '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "move-horizontal": '<svg aria-label="Chop icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert-triangle": '<svg aria-label="Crisis icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "help-circle": '<svg aria-label="Unknown icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle": '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "scale":      '<svg aria-label="Weighting icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
    "settings":   '<svg aria-label="Settings icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
}


def get_icon(name: str, size: int = 18, stroke_width: float = 1.5) -> str:
    """Return an SVG icon string with custom size and stroke width."""
    import re
    base_svg = ICONS.get(name, ICONS["chart"])
    
    # Clean existing attributes to avoid duplicates or stale values
    base_svg = re.sub(r'\s+width="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+height="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+stroke-width="[^"]*"', '', base_svg)
    
    # Inject standardized attributes
    return base_svg.replace('<svg', f'<svg width="{size}" height="{size}" stroke-width="{stroke_width}"')


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
    svg = get_icon(icon, size=16, stroke_width=1.8)
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
    icon: str = "",
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

    sub_metric_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12, stroke_width=2)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f"<h4>{icon_html}{html_mod.escape(label)}</h4>"
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
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


def render_system_card(
    title: str,
    description: str,
    specs: list[tuple[str, str]],
    card_class: str = "portfolio",
    icon: str = "briefcase"
) -> None:
    """Render a system feature card for landing page.

    Args:
        title: Card title.
        description: Card description.
        specs: List of (label, value) tuples for specifications.
        card_class: CSS class — "portfolio", "regime", "strategies".
        icon: Key from ICONS dict.
    """
    spec_html = "".join(
        f'<span>{html_mod.escape(label)}</span> {html_mod.escape(value)}<br>'
        for label, value in specs
    )
    svg = get_icon(icon, size=16, stroke_width=1.8)

    st.markdown(
        f"""
        <div class='system-card {html_mod.escape(card_class)}'>
            <h3>
                {svg}
                {html_mod.escape(title)}
            </h3>
            <p>{html_mod.escape(description)}</p>
            <div class='spec'>{spec_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
        f'<div class="interp-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kv_table(data: dict[str, str], header_left: str = "Setting", header_right: str = "Value") -> None:
    """Render a professional KV table in Obsidian Quant style.
    
    Args:
        data: Dictionary of keys and values to display.
        header_left: Header for the left column.
        header_right: Header for the right column.
    """
    import html as html_module
    rows = []
    for k, v in data.items():
        rows.append(
            f'<tr>'
            f'<td class="key">{html_module.escape(k)}</td>'
            f'<td class="value">{html_module.escape(v)}</td>'
            f'</tr>'
        )
    
    rows_html = "\n".join(rows)
    table_html = f'''
    <div class="kv-table-container">
        <table class="kv-table">
            <thead>
                <tr>
                    <th>{html_module.escape(header_left)}</th>
                    <th style="text-align:right;">{html_module.escape(header_right)}</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    '''
    st.markdown(table_html, unsafe_allow_html=True)

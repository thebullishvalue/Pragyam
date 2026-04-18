"""
PRAGYAM Universe Selection Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dynamic universe definitions and fetching functions for portfolio analysis.

Supports:
- ETF Universe (fixed list of 30 NSE ETFs)
- India Indices (NIFTY 50, NIFTY 500, F&O Stocks, sectoral indices)
- US Indices (S&P 500, DOW JONES, NASDAQ 100)
- Commodities (24 futures)
- Currency (24 pairs)

Adapted from Nirnay system.
"""

import streamlit as st
import pandas as pd
import requests
import io
from typing import List, Tuple, Optional, Dict

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── ETF Universe (Fixed) ─────────────────────────────────────────────────────
ETF_UNIVERSE = [
    "SENSEXIETF.NS", "NIFTYIETF.NS", "MON100.NS", "MAKEINDIA.NS", "SILVERIETF.NS",
    "HEALTHIETF.NS", "CONSUMIETF.NS", "GOLDIETF.NS", "INFRAIETF.NS", "CPSEETF.NS",
    "TNIDETF.NS", "COMMOIETF.NS", "MODEFENCE.NS", "MOREALTY.NS", "PSUBNKIETF.NS",
    "MASPTOP50.NS", "FMCGIETF.NS", "BANKIETF.NS", "ITIETF.NS", "EVINDIA.NS",
    "MNC.NS", "FINIETF.NS", "AUTOIETF.NS", "PVTBANIETF.NS", "MONIFTY500.NS",
    "ECAPINSURE.NS", "MIDCAPIETF.NS", "MOSMALL250.NS", "OILIETF.NS", "METALIETF.NS"
]

# ── India Index Universe ─────────────────────────────────────────────────────
INDIA_INDEX_LIST = [
    "NIFTY 50",
    "F&O Stocks",
    "NIFTY NEXT 50",
    "NIFTY 100",
    "NIFTY 200",
    "NIFTY 500",
    "NIFTY MIDCAP 50",
    "NIFTY MIDCAP 100",
    "NIFTY SMLCAP 100",
    "NIFTY BANK",
    "NIFTY AUTO",
    "NIFTY FIN SERVICE",
    "NIFTY FMCG",
    "NIFTY IT",
    "NIFTY MEDIA",
    "NIFTY METAL",
    "NIFTY PHARMA"
]

# ── US Index Universe ────────────────────────────────────────────────────────
US_INDEX_LIST = ["S&P 500", "DOW JONES", "NASDAQ 100"]

# ── Universe Options for Dropdown ────────────────────────────────────────────
UNIVERSE_OPTIONS = [
    "ETF Universe",
    "India Indexes",
    "US Indexes",
    "Commodities",
    "Currency"
]

# ── Index URL Map for fetching constituents ──────────────────────────────────
BASE_URL = "https://www.niftyindices.com/IndexConstituent/"
INDEX_URL_MAP = {
    "NIFTY 50": f"{BASE_URL}ind_nifty50list.csv",
    "NIFTY NEXT 50": f"{BASE_URL}ind_niftynext50list.csv",
    "NIFTY 100": f"{BASE_URL}ind_nifty100list.csv",
    "NIFTY 200": f"{BASE_URL}ind_nifty200list.csv",
    "NIFTY 500": f"{BASE_URL}ind_nifty500list.csv",
    "NIFTY MIDCAP 50": f"{BASE_URL}ind_niftymidcap50list.csv",
    "NIFTY MIDCAP 100": f"{BASE_URL}ind_niftymidcap100list.csv",
    "NIFTY SMLCAP 100": f"{BASE_URL}ind_niftysmallcap100list.csv",
    "NIFTY BANK": f"{BASE_URL}ind_niftybanklist.csv",
    "NIFTY AUTO": f"{BASE_URL}ind_niftyautolist.csv",
    "NIFTY FIN SERVICE": f"{BASE_URL}ind_niftyfinancelist.csv",
    "NIFTY FMCG": f"{BASE_URL}ind_niftyfmcglist.csv",
    "NIFTY IT": f"{BASE_URL}ind_niftyitlist.csv",
    "NIFTY MEDIA": f"{BASE_URL}ind_niftymedialist.csv",
    "NIFTY METAL": f"{BASE_URL}ind_niftymetallist.csv",
    "NIFTY PHARMA": f"{BASE_URL}ind_niftypharmalist.csv"
}

# ── Commodity Futures (Yahoo Finance) ─────────────────────────────────────────
COMMODITY_TICKERS = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "PL=F": "Platinum",
    "PA=F": "Palladium",
    "HG=F": "Copper",
    "CL=F": "Crude Oil WTI",
    "BZ=F": "Brent Crude",
    "NG=F": "Natural Gas",
    "RB=F": "Gasoline RBOB",
    "HO=F": "Heating Oil",
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
    "ZS=F": "Soybeans",
    "ZM=F": "Soybean Meal",
    "ZL=F": "Soybean Oil",
    "CT=F": "Cotton",
    "KC=F": "Coffee",
    "SB=F": "Sugar",
    "CC=F": "Cocoa",
    "OJ=F": "Orange Juice",
    "LBS=F": "Lumber",
    "LE=F": "Live Cattle",
    "HE=F": "Lean Hogs",
    "GF=F": "Feeder Cattle",
}

# ── Currency Pairs (Yahoo Finance) ────────────────────────────────────────────
CURRENCY_TICKERS = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    "USDINR=X": "USD/INR",
    "EURGBP=X": "EUR/GBP",
    "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    "AUDJPY=X": "AUD/JPY",
    "EURCHF=X": "EUR/CHF",
    "EURAUD=X": "EUR/AUD",
    "GBPCHF=X": "GBP/CHF",
    "GBPAUD=X": "GBP/AUD",
    "USDSGD=X": "USD/SGD",
    "USDHKD=X": "USD/HKD",
    "USDCNH=X": "USD/CNH",
    "USDZAR=X": "USD/ZAR",
    "USDMXN=X": "USD/MXN",
    "USDTRY=X": "USD/TRY",
    "USDBRL=X": "USD/BRL",
    "USDKRW=X": "USD/KRW",
}

# ── Hardcoded Dow Jones 30 components ─────────────────────────────────────────
DOW_JONES_TICKERS = [
    "AMZN", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
    "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
    "CRM", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "DIS", "DOW", "NVDA"
]

# ── Wikipedia URLs for India Index fallback ───────────────────────────────────
INDIA_INDEX_WIKI_MAP = {
    "NIFTY 50": "https://en.wikipedia.org/wiki/NIFTY_50",
    "NIFTY NEXT 50": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
    "NIFTY 500": "https://en.wikipedia.org/wiki/NIFTY_500",
}


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE FETCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_etf_universe() -> Tuple[List[str], str]:
    """Return the fixed ETF universe for analysis"""
    return ETF_UNIVERSE, f"✓ Loaded {len(ETF_UNIVERSE)} ETFs"


@st.cache_data(ttl=3600, show_spinner=False)
def get_fno_stock_list() -> Tuple[Optional[List[str]], str]:
    """Fetch F&O stock list from NSE. NSE requires a warm session cookie before
    the JSON API returns data, so we seed cookies via the homepage first."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
        }
        session = requests.Session()
        session.headers.update(headers)
        # Warm cookies — NSE returns empty {} without a valid session
        session.get("https://www.nseindia.com/", timeout=15)
        session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)

        url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get('data'):
            symbols = [item.get('symbol', '') for item in data['data'] if item.get('symbol')]
            # Drop the index row itself (e.g. "NIFTY 50") if NSE includes it
            symbols = [s for s in symbols if s and not s.startswith('NIFTY')]
            symbols_ns = [str(s) + ".NS" for s in symbols if str(s).strip()]
            if symbols_ns:
                return symbols_ns, f"✓ Fetched {len(symbols_ns)} F&O securities from NSE"

        return None, "Could not extract F&O symbols from NSE API"

    except Exception as e:
        return None, f"Error fetching F&O list: {e}"


def _parse_wiki_table(url: str, min_count: int = 10) -> Optional[List[str]]:
    """Parse a Wikipedia page and extract NSE symbols from the constituent table"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        for tbl in tables:
            if 'Symbol' in tbl.columns:
                symbols = tbl['Symbol'].dropna().astype(str).str.strip().tolist()
                symbols = [s for s in symbols if s and len(s) <= 20 and s != 'nan']
                if len(symbols) >= min_count:
                    return symbols
        return None
    except Exception:
        return None


def _fetch_india_index_from_wikipedia(index: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Fallback: Fetch Indian index constituents from Wikipedia when niftyindices.com is unreachable"""
    try:
        # NIFTY 100 is constructed from NIFTY 50 + NIFTY NEXT 50
        if index == "NIFTY 100":
            n50 = _parse_wiki_table(INDIA_INDEX_WIKI_MAP["NIFTY 50"], min_count=40)
            nn50 = _parse_wiki_table(INDIA_INDEX_WIKI_MAP["NIFTY NEXT 50"], min_count=40)
            if n50 and nn50:
                combined = list(dict.fromkeys(n50 + nn50))  # deduplicate preserving order
                symbols_ns = [s + ".NS" for s in combined]
                return symbols_ns, f"⚠ niftyindices.com unavailable → Loaded {len(symbols_ns)} NIFTY 100 constituents from Wikipedia (NIFTY 50 + Next 50)"
            return None, "Wikipedia fallback failed for NIFTY 100"

        # NIFTY 200 — use NIFTY 500 Wikipedia page (first 200 by order)
        if index == "NIFTY 200":
            symbols = _parse_wiki_table(INDIA_INDEX_WIKI_MAP["NIFTY 500"], min_count=100)
            if symbols:
                symbols_200 = symbols[:200]
                symbols_ns = [s + ".NS" for s in symbols_200]
                return symbols_ns, f"⚠ niftyindices.com unavailable → Loaded {len(symbols_ns)} NIFTY 200 constituents from Wikipedia (top 200 of NIFTY 500)"
            return None, "Wikipedia fallback failed for NIFTY 200"

        # Direct Wikipedia lookup for NIFTY 50, NIFTY NEXT 50, NIFTY 500
        wiki_url = INDIA_INDEX_WIKI_MAP.get(index)
        if wiki_url:
            min_expected = {"NIFTY 50": 40, "NIFTY NEXT 50": 40, "NIFTY 500": 400}.get(index, 10)
            symbols = _parse_wiki_table(wiki_url, min_count=min_expected)
            if symbols:
                symbols_ns = [s + ".NS" for s in symbols]
                return symbols_ns, f"⚠ niftyindices.com unavailable → Loaded {len(symbols_ns)} {index} constituents from Wikipedia"
            return None, f"Wikipedia fallback: could not parse {index} table"

        # No Wikipedia fallback available for this index (sectoral/midcap)
        return None, None

    except Exception as e:
        return None, f"Wikipedia fallback error: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_index_stock_list(index: str) -> Tuple[Optional[List[str]], str]:
    """Fetch index constituents from NSE Indices with Wikipedia fallback, or US Indices"""
    # Route US indices to separate handler
    if index in US_INDEX_LIST:
        return get_us_index_stock_list(index)

    url = INDEX_URL_MAP.get(index)
    if not url:
        return None, f"No URL for {index}"

    # ── Primary: niftyindices.com CSV ──
    primary_error = None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()

        csv_file = io.StringIO(response.text)
        stock_df = pd.read_csv(csv_file)

        if 'Symbol' in stock_df.columns:
            symbols = stock_df['Symbol'].tolist()
            symbols_ns = [str(s) + ".NS" for s in symbols if s and str(s).strip()]
            return symbols_ns, f"✓ Fetched {len(symbols_ns)} constituents from {index}"
        else:
            primary_error = "No Symbol column found in CSV"

    except Exception as e:
        primary_error = str(e)

    # ── Fallback: Wikipedia ──
    wiki_result, wiki_msg = _fetch_india_index_from_wikipedia(index)
    if wiki_result:
        return wiki_result, wiki_msg

    # Both failed — return informative error
    fallback_note = ""
    if wiki_msg is None:
        fallback_note = " (no Wikipedia fallback available for this index — try NIFTY 50/100/500 or retry later)"
    elif wiki_msg:
        fallback_note = f" | {wiki_msg}"

    return None, f"Error: {primary_error}{fallback_note}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_us_index_stock_list(index: str) -> Tuple[Optional[List[str]], str]:
    """Fetch US index constituents from Wikipedia with hardcoded fallback for Dow Jones"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        if index == "S&P 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            # Scan tables for the constituents table — column name/order on Wikipedia
            # has shifted historically ('Symbol' vs 'Ticker symbol'), so don't trust tables[0]
            for tbl in tables:
                cols = [str(c) for c in tbl.columns]
                sym_col = next((c for c in cols if c.strip().lower() in ('symbol', 'ticker', 'ticker symbol')), None)
                if not sym_col:
                    continue
                symbols = tbl[sym_col].dropna().astype(str).str.strip().tolist()
                # Yahoo uses '-' in place of '.' for class-share tickers (BRK.B → BRK-B)
                symbols = [s.replace('.', '-') for s in symbols if s and s.lower() != 'nan']
                if len(symbols) >= 400:
                    return symbols, f"✓ Fetched {len(symbols)} S&P 500 constituents from Wikipedia"
            return None, "Could not parse S&P 500 table from Wikipedia"

        elif index == "NASDAQ 100":
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            for tbl in tables:
                if 'Symbol' in tbl.columns or 'Ticker' in tbl.columns:
                    col = 'Symbol' if 'Symbol' in tbl.columns else 'Ticker'
                    symbols = tbl[col].dropna().astype(str).tolist()
                    symbols = [s.replace('.', '-') for s in symbols if s.strip()]
                    if len(symbols) > 50:
                        return symbols, f"✓ Fetched {len(symbols)} NASDAQ 100 constituents from Wikipedia"
            return None, "Could not parse NASDAQ 100 table"

        elif index == "DOW JONES":
            return DOW_JONES_TICKERS, f"✓ Loaded {len(DOW_JONES_TICKERS)} Dow Jones components"

        return None, f"Unknown US index: {index}"

    except Exception as e:
        return None, f"Error fetching {index}: {e}"


def get_commodity_list() -> Tuple[List[str], str]:
    """Return all commodity futures tickers for analysis"""
    tickers = list(COMMODITY_TICKERS.keys())
    return tickers, f"✓ Loaded {len(tickers)} commodity futures"


def get_currency_list() -> Tuple[List[str], str]:
    """Return all currency pair tickers for analysis"""
    tickers = list(CURRENCY_TICKERS.keys())
    return tickers, f"✓ Loaded {len(tickers)} currency pairs"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RESOLVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_universe(
    universe: str,
    index: Optional[str] = None
) -> Tuple[List[str], str]:
    """
    Resolve a universe selection to a list of symbols.

    Args:
        universe: One of UNIVERSE_OPTIONS ("ETF Universe", "India Indexes", etc.)
        index: Sub-selection (e.g., "NIFTY 50", "S&P 500") — required for India/US Indexes

    Returns:
        Tuple of (symbol_list, status_message)

    Raises:
        ValueError: If universe is unknown or index is missing when required
    """
    if universe == "ETF Universe":
        return get_etf_universe()

    elif universe == "India Indexes":
        if not index:
            raise ValueError("Index selection is required for India Indexes universe")
        return get_index_stock_list(index)

    elif universe == "US Indexes":
        if not index:
            raise ValueError("Index selection is required for US Indexes universe")
        return get_us_index_stock_list(index)

    elif universe == "Commodities":
        return get_commodity_list()

    elif universe == "Currency":
        return get_currency_list()

    else:
        raise ValueError(f"Unknown universe: {universe}. Choose from: {UNIVERSE_OPTIONS}")


def get_index_options(universe: str) -> List[str]:
    """Return the list of index options for a given universe"""
    if universe == "India Indexes":
        return INDIA_INDEX_LIST
    elif universe == "US Indexes":
        return US_INDEX_LIST
    return []


def get_default_index(universe: str) -> Optional[str]:
    """Return the default index for a given universe"""
    if universe == "India Indexes":
        return "NIFTY 50"
    elif universe == "US Indexes":
        return "DOW JONES"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# UI RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_universe_selector() -> Tuple[str, Optional[str]]:
    """
    Render the universe selection UI inputs in the sidebar (title rendered externally).

    Returns:
        Tuple of (universe, selected_index) where selected_index may be None
    """
    universe = st.selectbox(
        "Analysis Universe",
        UNIVERSE_OPTIONS,
        help="Choose the universe of securities to analyze"
    )

    selected_index = None

    # Show index dropdown only for India/US Indexes
    if universe in ("India Indexes", "US Indexes"):
        index_options = get_index_options(universe)
        default_index = get_default_index(universe)
        default_idx = index_options.index(default_index) if default_index in index_options else 0

        label = "Select Index" if universe == "India Indexes" else "Select US Index"
        help_text = "Select the index for constituent analysis"

        selected_index = st.selectbox(
            label,
            index_options,
            index=default_idx,
            help=help_text
        )

    return universe, selected_index


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Universe definitions
    'ETF_UNIVERSE',
    'INDIA_INDEX_LIST',
    'US_INDEX_LIST',
    'UNIVERSE_OPTIONS',
    'COMMODITY_TICKERS',
    'CURRENCY_TICKERS',
    'DOW_JONES_TICKERS',
    'INDEX_URL_MAP',
    # Fetching functions
    'get_etf_universe',
    'get_fno_stock_list',
    'get_index_stock_list',
    'get_us_index_stock_list',
    'get_commodity_list',
    'get_currency_list',
    # Resolver
    'resolve_universe',
    'get_index_options',
    'get_default_index',
    # UI
    'render_universe_selector',
]

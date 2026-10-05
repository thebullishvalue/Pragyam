"""
PRAGYAM — Market Data & Indicator Engine
══════════════════════════════════════════════════════════════════════════════

Fetches OHLCV history from yfinance (behind a circuit breaker) and computes the
per-symbol indicator panel that every downstream layer consumes.

Produces, per symbol per day (daily and weekly timeframes where noted):
  • Liquidity Oscillator (+ 9/21 EMAs) and its 20-period z-score
  • RSI (Wilder), moving averages (20/90/200) and their deviations
  • Volume-profile features (daily): point-of-control (POC), value-area position
    (``vap`` — a volatility-normalised premium/discount to accepted value) and
    in-value position (``va_pos``). See ``compute_volume_profile``.
  • Pragati's two tapes — conviction on Ladder down (intraday.py; D · W before
    intraday history) (pragati.py) and value on D · W
    (samanvaya.py, hedged against a macro basket fetched once per panel by
    ``fetch_macro_drivers``) — and the state they place each name in. See
    ``cvgrid.compute_readings``.

The canonical column set is ``COLUMN_ORDER``; ``generate_historical_data``
returns a chronological list of ``(date, snapshot_df)`` tuples in that shape,
which the regime detector and the covariance curation both read.

Author: @thebullishvalue
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import warnings
import os
from typing import List, Tuple, Dict, Any, Optional, cast

# Import circuit breaker, metrics and the console
from circuit_breaker import yfinance_circuit, RetryWithBackoff
from cvgrid import COLUMNS as CVG_COLUMNS, compute_readings
from samanvaya import DRIVER_TICKERS, _close_utc
from logger_config import console
from metrics import get_metrics

warnings.filterwarnings("ignore", category=FutureWarning)


class LiquidityOscillator:
    """Calculates the Liquidity Oscillator indicator."""

    def __init__(self, length: int = 20, impact_window: int = 3):
        if length <= 0 or impact_window <= 0:
            raise ValueError("length and impact_window must be positive integers.")
        self.length = length
        self.impact_window = impact_window

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            return pd.Series(dtype=float)

        # FX and some futures report volume as 0/NaN on Yahoo; the oscillator
        # is volume-weighted and cannot be computed in that case.
        if data['volume'].fillna(0).sum() == 0:
            return pd.Series(index=data.index, dtype=float, name='liquidity_oscillator')

        df = data.copy()
        df['spread'] = (df['high'] + df['low']) / 2 - df['open']
        df['vol_ma'] = df['volume'].rolling(window=self.length).mean()
        # Divide-by-zero guard: during the first `length-1` bars vol_ma is NaN
        # (rolling warmup), and occasionally a window can sum to exactly 0
        # volume. Filling either case with 1.0 does NOT make the ratio "safe" —
        # it turns spread*volume/1.0 into a value at ~volume's natural scale
        # (often 10^5-10^6x too large), which then sits inside the next
        # 20-bar rolling means and contaminates roughly bars [length, 2*length)
        # of every symbol's oscillator with garbage instead of "no signal yet".
        # NaN-propagate instead: where vol_ma isn't a valid positive average,
        # the ratio is undefined and must stay NaN so it's dropped by the
        # rolling mean exactly like the other warm-up NaNs downstream already
        # rely on (see compute_volume_profile's docstring for the convention).
        safe_vol_ma = df['vol_ma'].where(df['vol_ma'] > 0)
        df['vwap_spread'] = (df['spread'] * df['volume'] / safe_vol_ma).rolling(window=self.length).mean()
        close_shifted = df['close'].shift(self.impact_window)
        df['price_impact'] = ((df['close'] - close_shifted) * df['volume'] / safe_vol_ma).rolling(window=self.length).mean()
        df['liquidity_score'] = df['vwap_spread'] - df['price_impact']
        df['source_value'] = df['close'] + df['liquidity_score']
        df['lowest_value'] = df['source_value'].rolling(window=self.length).min()
        df['highest_value'] = df['source_value'].rolling(window=self.length).max()
        range_value = df['highest_value'] - df['lowest_value']
        # Same principle: a zero (or NaN, from warmup) high-low source range
        # means the 200*(x-lo)/range formula is undefined, not "1.0-wide" —
        # filling with 1.0 previously made a flat-price window explode to
        # +/-Infinity-adjacent values instead of correctly reporting NaN.
        safe_range_value = range_value.where(range_value > 0)
        oscillator = 200 * (df['source_value'] - df['lowest_value']) / safe_range_value - 100
        return oscillator.rename('liquidity_oscillator')

def resample_data(df: pd.DataFrame, rule: str = 'W-FRI') -> pd.DataFrame:
    """Resample daily OHLCV data to a different timeframe."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    agg_map = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample(rule).agg(agg_map).dropna()


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
    if data.empty or 'close' not in data.columns or len(data) < period:
        return pd.Series(index=data.index, dtype=float)

    delta = data['close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 (all-gains window) is a genuine RSI=100 case, but a blanket
    # fillna(100.0) also stamps the first `period` warm-up rows (NaN from
    # min_periods=period, not from an all-gains window) as a maximally
    # overbought 100 — a phantom signal where there is actually no data yet.
    # Only backfill the true all-gains case; leave warm-up NaNs as NaN so
    # downstream consumers treat them as "no signal" like every other
    # indicator's warmup.
    all_gains = avg_loss.eq(0) & avg_gain.notna() & (avg_gain > 0)
    rsi = rsi.where(~all_gains, 100.0)
    return rsi

def compute_volume_profile(
    df: pd.DataFrame,
    window: int = 90,
    bins: int = 60,
    value_area: float = 0.70,
) -> pd.DataFrame:
    """Rolling EOD volume profile → POC / value-area position features.

    The system has no notion of *where volume actually traded*. Its only
    "value" anchor is the rolling mean baked into the oscillator z-score. This
    builds, per bar, a trailing volume profile (the proxy/binned model from the
    Inferred-Delta volume-profile indicator, adapted to a cross-sectional EOD
    panel) and returns three measured, no-inference features:

      • ``poc``        – point of control: the modal price (price bin that
                         accumulated the most volume) over the trailing window.
      • ``vap``        – Value-Area Position: where the latest close sits inside
                         the developing volume profile, volatility-normalised by
                         the value-area half-width and clamped to roughly
                         [-3, +3]. Positive = price trades at a *discount* to
                         accepted value (mean-reversion long), negative = at a
                         *premium*. This is the cross-sectional signal feeding
                         the regime detector's acceptance factor.
      • ``va_pos``     – raw position of close within [VAL, VAH] mapped to
                         [-1, +1] (inside-value vs at-an-edge), used by the
                         portfolio selection layer for structural hold/rotate.

    Volume binning replicates the indicator's profile loop: each bar's volume is
    spread across the price bins its high–low range touches, so a bar that
    straddles many levels contributes a thin slice to each. The value area is
    grown outward from the POC, always taking the heavier adjacent bin, until it
    holds ``value_area`` of the windowed volume — the same expansion the
    indicator uses for VAH/VAL.

    Per-window histogram construction is vectorized via a difference-array +
    cumsum (each bar's "add v/spread to bins [b_lo, b_hi]" range-update becomes
    two O(1) writes and one O(bins) cumulative sum instead of an O(window)
    Python loop per window) — validated bit-identical against the original
    per-bar loop, ~4-5x faster end to end. The outer per-window loop remains
    (the bin edges genuinely shift every window since [lo, hi] is a trailing
    min/max), so this is not a full incremental/streaming histogram.

    Returns a DataFrame indexed like ``df`` with columns ``poc``/``vap``/``va_pos``.
    Bars before the window is warm, or with no usable volume, are NaN — they are
    dropped downstream exactly like the other indicators' warmup NaNs.
    """
    import numpy as np
    out = pd.DataFrame(index=df.index, columns=['poc', 'vap', 'va_pos'], dtype=float)
    n = len(df)
    if n < window or not {'high', 'low', 'close', 'volume'}.issubset(df.columns):
        return out

    highs = df['high'].to_numpy(dtype=float)
    lows = df['low'].to_numpy(dtype=float)
    closes = df['close'].to_numpy(dtype=float)
    vols = df['volume'].to_numpy(dtype=float)

    for end in range(window - 1, n):
        start = end - window + 1
        wl = lows[start:end + 1]
        wh = highs[start:end + 1]
        wv = vols[start:end + 1]
        price = closes[end]

        lo = np.nanmin(wl)
        hi = np.nanmax(wh)
        # Need a real price range and some real volume to build a profile.
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        total_vol = np.nansum(wv)
        if not np.isfinite(total_vol) or total_vol <= 0:
            continue

        step = (hi - lo) / bins
        if step <= 0:
            continue

        # ── Vectorized histogram build ──────────────────────────────────────
        # The original per-window loop did `for j in range(window)`, each
        # iteration slicing hist[b_lo:b_hi+1] += slice_v — 90*90 = 8,100
        # Python-level operations per symbol per window*, the dominant cost
        # of the whole data-fetch phase at universe scale (~0.2s/symbol here,
        # ~100s for a 500-symbol universe). Every bar's contribution is
        # "add `v/spread` to each bin in [b_lo, b_hi]" — a range-update, which
        # a difference array turns into two O(1) writes (+v/spread at b_lo,
        # -v/spread at b_hi+1) followed by ONE cumulative sum over `bins`
        # (60) instead of up to `bins` writes per bar. This produces the
        # IDENTICAL histogram (validated against the original loop) at a
        # fraction of the cost.
        valid = np.isfinite(wv) & (wv > 0) & np.isfinite(wl) & np.isfinite(wh)
        if not valid.any():
            continue
        wl_v, wh_v, wv_v = wl[valid], wh[valid], wv[valid]

        b_lo_arr = np.floor((wl_v - lo) / step).astype(np.int64)
        b_hi_arr = np.floor((wh_v - lo) / step).astype(np.int64)
        np.clip(b_lo_arr, 0, bins - 1, out=b_lo_arr)
        np.clip(b_hi_arr, 0, bins - 1, out=b_hi_arr)
        spread_arr = (b_hi_arr - b_lo_arr + 1).astype(np.float64)
        slice_v_arr = wv_v / spread_arr

        diff = np.zeros(bins + 1, dtype=np.float64)
        np.add.at(diff, b_lo_arr, slice_v_arr)
        np.add.at(diff, b_hi_arr + 1, -slice_v_arr)
        hist = np.cumsum(diff[:-1])

        # ── POC: modal price bin ──────────────────────────────────────────────
        poc_bin = int(np.argmax(hist))
        poc_price = lo + (poc_bin + 0.5) * step

        # ── Value area: grow outward from POC, heavier-side first, to value_area
        acc = hist[poc_bin]
        target = total_vol * value_area
        up = poc_bin
        dn = poc_bin
        while acc < target and (dn > 0 or up < bins - 1):
            v_up = hist[up + 1] if up < bins - 1 else -1.0
            v_dn = hist[dn - 1] if dn > 0 else -1.0
            if v_up >= v_dn:
                acc += v_up
                up += 1
            else:
                acc += v_dn
                dn -= 1
        vah = lo + (up + 1) * step
        val = lo + dn * step

        # ── va_pos: close inside [VAL, VAH] mapped to [-1, +1] ────────────────
        va_mid = (vah + val) / 2.0
        va_half = max((vah - val) / 2.0, step)
        va_pos = (price - va_mid) / va_half
        va_pos = float(np.clip(va_pos, -1.0, 1.0))

        # ── vap: volatility-normalised premium/discount to value (mean-rev) ───
        #  Distance of close from POC, scaled by the value-area half-width so the
        #  number is comparable across symbols of very different price/vol. We
        #  invert the sign so DISCOUNT (below value) is POSITIVE = a long bias,
        #  matching the oversold-is-positive convention of zscore_signal.
        vap = -(price - poc_price) / va_half
        vap = float(np.clip(vap, -3.0, 3.0))

        out.iat[end, 0] = poc_price
        out.iat[end, 1] = vap
        out.iat[end, 2] = va_pos

    return out


def calculate_all_indicators(
    symbol_data: pd.DataFrame,
    oscillator_calculator: LiquidityOscillator,
    driver_closes: Optional[pd.DataFrame] = None,
    ticker: str = "",
) -> pd.DataFrame | None:
    """
    Calculate all indicators for a single symbol's full history.

    Returns a DataFrame indexed by date with columns for price, returns,
    oscillators, RSI, moving averages, deviations, and z-scores across
    daily and weekly timeframes, plus the daily volume-profile features
    (POC, value-area position ``vap``, and in-value position ``va_pos``),
    plus Pragati's two tapes and the Conviction-Value Grid state (``driver_closes`` feeds the value
    tape's macro hedge; ``ticker`` sets its driver timing and home index).
    Returns ``None`` on empty input.
    """
    daily_data = symbol_data.copy()
    if daily_data.empty:
        return None

    weekly_data = resample_data(daily_data, 'W-FRI')
    
    all_results_df = pd.DataFrame(index=daily_data.index)
    all_results_df['price'] = daily_data['close']
    all_results_df['% change'] = daily_data['close'].pct_change()

    timeframes = {'latest': daily_data, 'weekly': weekly_data}
    
    for tf_name, df in timeframes.items():
        if len(df) < 2:
            continue
        
        osc = oscillator_calculator.calculate(df)
        if not osc.dropna().empty:
            all_results_df[f'osc {tf_name}'] = osc
            all_results_df[f'9ema osc {tf_name}'] = osc.ewm(span=9).mean()
            all_results_df[f'21ema osc {tf_name}'] = osc.ewm(span=21).mean()

            if len(osc.dropna()) >= 20:
                osc_sma20 = osc.rolling(window=20).mean()
                osc_std20 = osc.rolling(window=20).std()
                safe_std20 = osc_std20.replace(0, pd.NA).fillna(1.0)
                all_results_df[f'zscore {tf_name}'] = (osc - osc_sma20) / safe_std20

        rsi_series = calculate_rsi(df)
        if rsi_series is not None and not rsi_series.dropna().empty:
            all_results_df[f'rsi {tf_name}'] = rsi_series

        for period in [20, 90, 200]:
            if len(df) >= period:
                all_results_df[f'ma{period} {tf_name}'] = df['close'].rolling(window=period).mean()
                if period == 20:
                    all_results_df[f'dev{period} {tf_name}'] = df['close'].rolling(window=period).std()

    # ── Volume profile (daily only): POC / value-area position ────────────────
    #  Measured, no-inference structure ported from the Inferred-Delta volume
    #  profile. Gives the system its missing "where volume actually traded"
    #  dimension; feeds the regime detector's acceptance factor.
    vp = compute_volume_profile(daily_data)
    all_results_df['poc latest'] = vp['poc']
    all_results_df['vap latest'] = vp['vap']
    all_results_df['va_pos latest'] = vp['va_pos']

    all_results_df = all_results_df.reindex(daily_data.index)

    weekly_cols = [col for col in all_results_df.columns if 'weekly' in col]
    all_results_df[weekly_cols] = all_results_df[weekly_cols].ffill()

    # ── Pragati's two tapes, and the Conviction-Value Grid state ─────────────
    #  What the Conviction-Value Grid allocates from. Computed HERE because this is the
    #  last place the full OHLCV exists: the snapshots carry close prices only,
    #  and start after a warm-up both tapes need. Attached AFTER the weekly
    #  forward-fill above on purpose: the weekly rungs are not resampled series
    #  waiting to be spread across their week — they are daily readings,
    #  reconstructed from the week as it forms.
    dh = compute_readings(daily_data, driver_closes, ticker)
    for col in CVG_COLUMNS:
        all_results_df[col] = dh[col].reindex(all_results_df.index)

    return all_results_df


def get_default_universe() -> List[str]:
    """Get the default ETF universe from the universe module.

    universe.ETF_UNIVERSE is the single source of truth (see
    AUDIT_DIRECTIVES.md B4 — this function previously had its own
    hardcoded 30-symbol list that had silently drifted from
    universe.ETF_UNIVERSE, a THIRD divergent definition alongside
    symbols.txt). If the universe module is ever unavailable, fall back to
    symbols.txt (kept in sync with universe.ETF_UNIVERSE) rather than a
    second hardcoded copy that can drift again.
    """
    try:
        from universe import ETF_UNIVERSE
        return ETF_UNIVERSE
    except ImportError:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(here, "symbols.txt"), "r") as f:
                return [line.strip() for line in f if line.strip()]
        except OSError:
            return []

# Default universe (can be overridden by caller)
SYMBOLS_UNIVERSE = get_default_universe()

# Define the column order here so it can be used by the generator
COLUMN_ORDER = [
    'date', 'symbol', 'price', 'rsi latest', 'rsi weekly',
    '% change', 'osc latest', 'osc weekly',
    '9ema osc latest', '9ema osc weekly',
    '21ema osc latest', '21ema osc weekly',
    'zscore latest', 'zscore weekly',
    'ma20 latest', 'ma90 latest', 'ma200 latest',
    'ma20 weekly', 'ma90 weekly', 'ma200 weekly',
    'dev20 latest', 'dev20 weekly',
    # Volume-profile features (daily): point-of-control + value-area position.
    'poc latest', 'vap latest', 'va_pos latest',
    # Pragati (pragati.py, samanvaya.py) and the grid (cvgrid.py): the conviction
    # tape and its rungs, the
    # value tape, its chart reading, hedge weight and drivers, and the state.
    *CVG_COLUMNS,
]

# --- NEW: Export max indicator period ---
INDICATOR_PERIODS = [20, 90, 200]
MAX_INDICATOR_PERIOD = max(INDICATOR_PERIODS)


# ── Second-pass recovery for symbols the batch download missed ────────────────
#
# `yf.download` over a LIST of tickers does not raise when only SOME of them
# fail — it returns all-NaN columns for those. The RetryWithBackoff wrapped
# around the batch call therefore never fires for the common case (two or three
# symbols out of thirty), and those columns were dropped silently, leaving the
# run on a quietly smaller universe. Because the panel is cached for an hour
# (app._load_historical_data), one unlucky fetch degraded every rerun until the
# TTL expired.
#
# The constants below bound a per-symbol second pass over exactly the symbols
# that came back empty. It is deliberately NOT attempted when most of the
# universe is missing: that is a service outage, and re-requesting 200 symbols
# one at a time turns a fast failure into a very slow one.
_RECOVERY_MAX_SYMBOLS = 40        # cap on single-symbol fetches per run
_RECOVERY_FAILURE_RATIO = 0.5     # >50% of the universe missing = outage, don't retry
_RECOVERY_ABORT_AFTER = 5         # consecutive failures that mean "stop, it's the service"
_RECOVERY_ATTEMPTS = 2            # attempts per symbol
_RECOVERY_DELAY = 1.0             # seconds between attempts on one symbol

# Trading bars a symbol may lag the panel end before it counts as missed rather
# than merely quiet. Shared with the forward-fill below so "recoverable gap" and
# "gap we carry the last price across" are the same number by construction.
_STALE_BARS = 5


def fetch_macro_drivers(start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Daily closes of the value tape's macro drivers — one batch per panel.

    Samanvaya's hedge basket (samanvaya.DRIVERS): US yields, bond-ETF proxies
    for the other 10-year yields, the dollar index, energy, metals, the INR
    crosses and the home equity indices. Shared by every name in the panel;
    each name aligns them to its own calendar and close time.

    Returns None rather than raising. The value engine runs without drivers —
    the RV leg becomes the name's own path, the Pine's "Macro hedge: Off" —
    so a failed fetch degrades the tape, it does not end the run. Each missing
    driver is reported, since a basket short of a factor is a different basket.
    """
    try:
        @yfinance_circuit.protect
        @RetryWithBackoff(max_retries=2, initial_delay=2.0, backoff_factor=2.0)
        def _download():
            return yf.download(DRIVER_TICKERS, start=start_date, end=end_date + timedelta(days=1),
                               progress=False, auto_adjust=True)

        raw = _download()
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.dropna(how="all", axis=1)
    except Exception as e:
        get_metrics().add_warning(f"Macro drivers unavailable ({type(e).__name__}: {e}) — "
                                  "value tape runs unhedged")
        console.warning(f"Macro drivers unavailable ({type(e).__name__}) — value tape runs unhedged")
        return None
    missing = [t for t in DRIVER_TICKERS if t not in close.columns]
    if missing:
        get_metrics().add_warning(f"{len(missing)} macro driver(s) returned no data: "
                                  f"{', '.join(missing)} — dropped from their factors")
        console.warning(f"Macro drivers missing: {', '.join(missing)} — dropped from their factors")
    console.detail(f"macro drivers · {close.shape[1]} of {len(DRIVER_TICKERS)} series · "
                   f"{len(close.index)} bars")
    return pd.DataFrame(close)


# A close that repeats the one before it, in a run of at least this many consecutive repeats
# (11 identical closes), is a dead quote, not a market: yfinance carries NESTLEIND.NS flat from
# the start of its history to Jan 2010 (986 repeats from Jan 2006; 786 inside the research
# panels, which start Oct 2006) and BAJAJ-AUTO.NS flat for 45 around its 2008 relisting.
# research/style_blends.py found them; the same rule (style_blends.unstale) repairs this panel.
_DEAD_QUOTE_RUN = 10


def mask_dead_quotes(close: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Unprice dead quotes: a close repeating the one before it, in a run of >= _DEAD_QUOTE_RUN
    consecutive repeats, becomes NaN. Returns the masked frame and the count per column.
    Causal only over the frame it is given — apply it to a frame that ends on the run date."""
    if close is None or close.empty:
        return close, {}
    same = close.diff().eq(0)
    run = same.apply(lambda c: c.groupby((~c).cumsum()).transform("sum"))
    dead = same & (run >= _DEAD_QUOTE_RUN)
    counts = {str(c): int(n) for c, n in dead.sum().items() if n}
    return (close.mask(dead) if counts else close), counts


_mask_dead = mask_dead_quotes

# One-day close moves that are also overnight gaps of this size, on an Indian listing, are read
# as corporate actions yfinance left unadjusted (MM-B3): demergers (BAJAJFINSV 2008, ADANIENT
# 2015-06-03, TMPV 2025-10-14) and mis-dated splits or bonuses (LT 2006-09-27/28, TRENT
# 2026-01-01). NSE price bands make a real 30% overnight gap rare; a US small cap's earnings
# gap is not, so other listings are left alone. A close-only rule flags 200 crypto days, and a
# rule that also wants a calm open-to-close misses two of the seven events.
_CORP_ACTION_JUMP = 0.30


def corporate_action_gaps(close: pd.DataFrame, open_: pd.DataFrame) -> List[Tuple[str, pd.Timestamp, float]]:
    """(symbol, date, close / previous close) for every move read as an unadjusted corporate action:
    an Indian listing (.NS / .BO) whose close moved >= 30% on a >= 30% overnight gap."""
    cols = [c for c in close.columns if str(c).upper().endswith((".NS", ".BO"))]
    if not cols or open_ is None:
        return []
    c = close[cols].apply(pd.to_numeric, errors="coerce")
    o = open_.reindex(index=c.index, columns=cols).apply(pd.to_numeric, errors="coerce")
    prev = c.ffill(limit=5).shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = (((c / prev - 1.0).abs() >= _CORP_ACTION_JUMP)
               & ((o / prev - 1.0).abs() >= _CORP_ACTION_JUMP)).to_numpy()
    return [(cols[j], c.index[i], float(c.iat[i, j] / prev.iat[i, j])) for i, j in np.argwhere(hit)]


def back_adjust(frame: pd.DataFrame, events, columns=None) -> pd.DataFrame:
    """Scale every row before each event by its ratio, so the jump leaves the return series and no
    return before or after it changes (causal: only earlier levels move, by a constant).
    `columns` maps an event's symbol to the frame columns to scale (default: the symbol)."""
    if not events:
        return frame
    out = frame.copy()
    for sym, t, k in events:
        cols = columns(sym) if columns is not None else [sym]
        cols = [c for c in cols if c in out.columns]
        if cols and np.isfinite(k) and k > 0:
            out.loc[out.index < t, cols] = out.loc[out.index < t, cols] * k
    return out


def fetch_close_history(symbols: List[str], start_date: datetime,
                        end_date: datetime, mask_dead_quotes: bool = True) -> Optional[pd.DataFrame]:
    """Daily closes of the universe over a long window — what Managed Momentum's overlay reads.

    The estimation panel (generate_historical_data) carries ~400 sessions (~19 months), enough for
    every other style. Managed Momentum also needs the market's 24-month return for its bear
    gate and its overlay's volatility over all the history there is for the expanding median
    that scales it (nco.mmom_overlay) — closes only, no indicators, so one batch download.

    Columns are named as the snapshots name them (".NS" dropped), the same adjusted closes the
    panel's `price` column carries; a symbol listed twice (an ADR and its NSE line both named
    INFY) keeps the listing that comes last in `symbols`, as the panel does. An Indian listing's
    unadjusted corporate action (a >= 30% move on a >= 30% overnight gap) is back-adjusted
    (corporate_action_gaps). Symbols the batch missed get the panel's
    own second pass (_recover_missing_symbols). A close repeating the one before it in a run of
    >= _DEAD_QUOTE_RUN consecutive repeats is set to NaN (mask_dead_quotes) — pass
    mask_dead_quotes=False to mask later, after slicing to a run date, so that repeats after it
    cannot decide what is masked before it. Returns None rather than raising: the overlay then
    stands down to the grid, and the app says so.
    """
    if not symbols:
        return None
    try:
        @yfinance_circuit.protect
        @RetryWithBackoff(max_retries=2, initial_delay=2.0, backoff_factor=2.0)
        def _download():
            return yf.download(list(symbols), start=start_date, end=end_date + timedelta(days=1),
                               progress=False)

        raw = _download()
        if isinstance(raw.columns, pd.MultiIndex) and len(symbols) > 1:
            raw, _rec = _recover_missing_symbols(raw, list(symbols), start_date, end_date)
            if _rec.get("recovered"):
                console.detail(f"close history · re-fetched {len(_rec['recovered'])} symbol(s) "
                               f"the batch missed: {', '.join(_rec['recovered'][:12])}")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(
            [symbols[0]], axis=1)
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
        _open = (raw["Open"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Open"]].set_axis(
            [symbols[0]], axis=1)) if "Open" in raw.columns.get_level_values(0) else None
    except Exception as e:
        get_metrics().add_warning(f"Close history unavailable ({type(e).__name__}: {e}) — "
                                  "Managed Momentum stands down to the grid")
        console.warning(f"Close history unavailable ({type(e).__name__}) — "
                        "Managed Momentum stands down to the grid")
        return None
    if close.empty:
        return None
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close.loc[:pd.Timestamp(end_date).normalize()]
    if _open is not None:
        _open = _open.set_axis(pd.DatetimeIndex(_open.index).tz_localize(None).normalize(), axis=0)
        _open = _open[~_open.index.duplicated(keep="last")]
        _ev = corporate_action_gaps(close, _open)
        if _ev:
            close = back_adjust(close, _ev)
            console.detail("close history · corporate-action gaps back-adjusted: " + ", ".join(
                f"{c.replace('.NS', '')} {t:%Y-%m-%d} ×{k:.3f}" for c, t, k in _ev))
    if mask_dead_quotes:
        close, _dead = _mask_dead(close)
        if _dead:
            console.detail("close history · unpriced dead quotes: "
                           + ", ".join(f"{c.replace('.NS', '')} {n}" for c, n in _dead.items()))
    # One listing per stripped name: the LAST in the caller's list order, as the snapshot rows
    # and nco.cvg_readings (keep="last") resolve it — not yfinance's alphabetical column order,
    # which for ['INFY.NS', 'INFY'] gave the grid the ADR and the overlay the NSE line (MM-B11).
    # Only true duplicates are resolved; a column that matches no requested string exactly
    # (yfinance normalising a ticker's case) is kept, as before.
    _order = {str(s).upper(): i for i, s in enumerate(symbols)}
    _rank: Dict[str, int] = {}
    _keep: Dict[str, int] = {}
    for j, c in enumerate(close.columns):
        k, r = str(c).replace(".NS", ""), _order.get(str(c).upper(), -1)
        if k not in _rank or r >= _rank[k]:
            _rank[k], _keep[k] = r, j
    close = close.iloc[:, sorted(_keep.values())]
    close.columns = [str(c).replace(".NS", "") for c in close.columns]
    console.detail(f"close history · {close.shape[1]} of {len(set(symbols))} symbols · "
                   f"{close.index[0]:%Y-%m-%d} → {close.index[-1]:%Y-%m-%d}")
    return close


def _unfetched_symbols(close: pd.DataFrame, symbols: List[str]) -> List[str]:
    """Symbols whose batch download returned nothing, or stopped early.

    Two distinct failures share one signature here: a column that is entirely
    NaN (nothing came back at all) and a column whose last valid bar sits more
    than ``_STALE_BARS`` before the end of the panel (the download stopped
    early, so every price the run reads for that symbol is stale).

    A LEADING gap is deliberately NOT counted as missed. A symbol that listed
    part-way through the window genuinely has no earlier history, and retrying
    it every run would spend the whole recovery budget re-requesting data that
    does not exist.
    """
    if close is None or close.empty:
        return list(symbols)

    missed: List[str] = []
    n_rows = len(close.index)
    for sym in symbols:
        if sym not in close.columns:
            missed.append(sym)              # yfinance omitted the column outright
            continue
        col = pd.to_numeric(close[sym], errors="coerce")
        if not col.notna().any():
            missed.append(sym)              # column present, nothing in it
            continue
        last = col.last_valid_index()
        loc = int(close.index.get_indexer(pd.Index([last]))[0]) if last is not None else -1
        if loc < 0:
            continue
        if n_rows - 1 - loc > _STALE_BARS:
            missed.append(sym)
    return missed


def _refetch_symbol(symbol: str, start_date: datetime,
                    end_date: datetime) -> Optional[pd.DataFrame]:
    """Re-fetch ONE symbol through ``Ticker.history`` — a different endpoint.

    Using a different endpoint is the point: when the batch quote endpoint
    returns an empty column for a symbol it is frequently a per-request failure
    rather than a missing instrument, and the per-ticker chart endpoint returns
    the series normally.

    Deliberately OUTSIDE the circuit breaker. A delisted or misspelled symbol
    fails on every attempt, and counting those against the breaker would trip it
    (threshold 5) and block the next run's batch download for a minute over what
    is a universe problem, not a service problem.

    Returns a tz-naive daily frame, or None when nothing came back.
    """
    for attempt in range(_RECOVERY_ATTEMPTS):
        try:
            df = yf.Ticker(symbol).history(
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=True,     # matches yf.download's default, so recovered
                actions=False,        # prices sit on the same adjustment basis
            )
        except Exception:
            df = None
        if df is not None and not df.empty:
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                # The batch panel is tz-naive; Ticker.history returns
                # Asia/Kolkata for .NS names. Reindexing one onto the other
                # without this silently yields an all-NaN column.
                idx = idx.tz_localize(None)
            out: pd.DataFrame = df.copy()
            out.index = idx.normalize()
            return out
        if attempt + 1 < _RECOVERY_ATTEMPTS:
            time.sleep(_RECOVERY_DELAY)
    return None


def _recover_missing_symbols(
    all_data: pd.DataFrame,
    symbols_to_process: List[str],
    start_date: datetime,
    end_date: datetime,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Re-fetch and merge the symbols the batch download missed.

    Runs BEFORE the failed-ticker drop, so a symbol recovered here stays in the
    universe rather than disappearing from the book. Recovered bars only ever
    FILL holes — an existing batch value is never overwritten, so recovery
    cannot quietly restate prices the rest of the panel was built on.
    """
    report: Dict[str, Any] = {"missing": [], "recovered": [], "failed": [],
                              "skipped_reason": None}
    try:
        # Selecting one level of a MultiIndex column yields a DataFrame. The
        # pandas stubs type __getitem__(str) as returning a Series, which it is
        # not here, so the shape has to be asserted rather than inferred.
        close = cast(pd.DataFrame, all_data["Close"])
    except (KeyError, TypeError):
        return all_data, report

    missing = _unfetched_symbols(close, symbols_to_process)
    report["missing"] = list(missing)
    if not missing:
        return all_data, report

    # An outage is not a per-symbol problem, and the batch retry has already
    # covered the transient case. Fail fast instead of grinding.
    if len(missing) > _RECOVERY_FAILURE_RATIO * len(symbols_to_process):
        report["skipped_reason"] = "outage"
        return all_data, report
    if len(missing) > _RECOVERY_MAX_SYMBOLS:
        report["skipped_reason"] = "budget"
        missing = missing[:_RECOVERY_MAX_SYMBOLS]

    recovered: Dict[str, pd.DataFrame] = {}
    consecutive_failures = 0
    for sym in missing:
        df = _refetch_symbol(sym, start_date, end_date)
        if df is None:
            report["failed"].append(sym)
            consecutive_failures += 1
            if consecutive_failures >= _RECOVERY_ABORT_AFTER:
                # Nothing is coming back at all — stop spending the budget.
                report["skipped_reason"] = "aborted"
                break
            continue
        consecutive_failures = 0
        recovered[sym] = df
        report["recovered"].append(sym)

    for sym, df in recovered.items():
        for field_name in ("Open", "High", "Low", "Close", "Volume"):
            if field_name not in df.columns:
                continue
            series = pd.to_numeric(df[field_name], errors="coerce").reindex(all_data.index)
            col = (field_name, sym)
            if col in all_data.columns:
                all_data[col] = all_data[col].fillna(series)   # fill holes only
            else:
                all_data[col] = series
    if recovered:
        # Adding columns leaves the MultiIndex unsorted, which makes the
        # (slice(None), tickers) selection below raise or warn.
        all_data = all_data.sort_index(axis=1)

    if report["recovered"] or report["failed"]:
        # Mechanics only. WHICH symbols were missing and what became of them is
        # reported by the step that owns this fetch (app._log_recovery), so the
        # two do not print the same list twice at different indents.
        msg = (f"re-fetch · {len(report['recovered'])} of {len(report['missing'])} "
               f"recovered on the second pass")
        if report["failed"]:
            console.warning(msg + f" · {len(report['failed'])} still unavailable")
        else:
            console.detail(msg)
    return all_data, report


def generate_historical_data(
    symbols_to_process: List[str],
    start_date: datetime,
    end_date: datetime,
) -> List[Tuple[datetime, pd.DataFrame]]:
    """
    Generate historical indicator snapshots for a list of symbols.
    
    LITERATURE-RIGOROUS VALIDATION:
    - Validates symbol universe
    - Validates date range
    - Validates data quality
    - Propagates errors explicitly
    - Uses circuit breaker for yfinance

    Args:
        symbols_to_process: Stock ticker symbols (e.g. ``["RELIANCE.NS"]``).
        start_date: Beginning of the download window (must include warmup).
        end_date: End of the snapshot window.

    Returns:
        Chronologically ordered list of ``(date, indicator_df)`` tuples.
        
    Raises:
        ValueError: If symbol universe is empty or date range is invalid
        ConnectionError: If yfinance API fails
        RuntimeError: If no valid data received
    """
    # Get metrics tracker
    metrics = get_metrics()
    
    # === VALIDATION 1: Symbol Universe ===
    if not symbols_to_process:
        metrics.add_error("ValueError", "Symbol universe is empty", "generate_historical_data")
        raise ValueError("Symbol universe is empty - please select a valid universe")
    
    if len(symbols_to_process) > 500:
        metrics.add_warning(f"Large universe ({len(symbols_to_process)} symbols) - may be slow")
        console.warning(
            f"Large universe: {len(symbols_to_process)} symbols (recommended: <300) - "
            "the fetch and every downstream estimate will be slower"
        )
    
    # === VALIDATION 2: Date Range ===
    if start_date > end_date:
        metrics.add_error(
            "ValueError",
            f"Start date ({start_date}) is after end date ({end_date})",
            "generate_historical_data"
        )
        raise ValueError(f"Start date ({start_date}) cannot be after end date ({end_date})")
    
    # Note: No limit on date range - allow user to fetch any range they need
    # Large date ranges will take longer but are valid
    
    # Update metrics
    metrics.symbols_count = len(symbols_to_process)
    
    # === DOWNLOAD WITH CIRCUIT BREAKER + RETRY ===
    try:
        # Retry INSIDE the circuit breaker, breaker OUTSIDE: a single yfinance
        # transient (the common case — brief rate-limit blip, momentary
        # network hiccup) gets a couple of quick backoff retries before
        # counting as a breaker failure. Without this, RetryWithBackoff was
        # imported but never applied anywhere, so one transient failure
        # ended the whole run immediately (see AUDIT_DIRECTIVES.md B6).
        @yfinance_circuit.protect
        @RetryWithBackoff(max_retries=2, initial_delay=2.0, backoff_factor=2.0)
        def download_data():
            return yf.download(
                symbols_to_process,
                start=start_date,
                end=end_date + timedelta(days=1),
                progress=False,
            )

        console.detail(
            f"yfinance batch · {len(symbols_to_process)} symbols · "
            f"{start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}"
        )
        all_data = download_data()
        # The conviction ladder reads DOWN (v9.1): every intraday frame yfinance carries,
        # fetched once for the whole universe, read by cvgrid.compute_readings per name.
        try:
            import intraday as _idm
            _t0 = time.time()
            _cov = _idm.prefetch(list(symbols_to_process))
            if _cov:
                _n = len(set(symbols_to_process))
                console.detail("intraday ladder · " + " · ".join(f"{f} {_cov.get(f, 0)}" for f in _idm.DAILY_FRAMES)
                               + (f" of {_n} symbols · {_dt:.1f}s" if (_dt := time.time() - _t0) >= 0.1 else f" of {_n} symbols · cached"))
                if _n - max(_cov.values()):
                    console.warning(f"{_n - max(_cov.values())} symbol(s) have no intraday history — "
                                    "their conviction reads D · W (↺)")
            else:
                console.detail("intraday ladder disabled — conviction reads D · W (↺)")
        except Exception as _e:        # the tape then reads Ladder up for every name
            console.warning(f"intraday prefetch failed ({type(_e).__name__}: {_e}) — conviction reads D · W (↺)")
        console.detail(
            f"received {len(all_data.index)} bars × "
            f"{len(getattr(all_data.get('Close', all_data), 'columns', []))} price columns"
        )
        
    except Exception as e:
        # Circuit breaker or download failed
        metrics.add_error(type(e).__name__, str(e), "yfinance.download")
        
        # Check if it's a circuit breaker error
        if "Circuit" in str(e) and "OPEN" in str(e):
            raise ConnectionError(
                f"yfinance service unavailable (circuit breaker OPEN): {str(e)}"
            ) from e
        else:
            raise ConnectionError(f"yfinance API failed: {str(e)}") from e
    
    # === VALIDATION 3: Data Received ===
    if all_data.empty or all_data['Close'].dropna(how='all').empty:
        metrics.add_error("RuntimeError", "No valid market data received from yfinance", "data_validation")
        raise ValueError("No valid market data received from yfinance - check symbols and date range")
    
    # === RECOVERY: re-fetch what the batch download missed ===
    # Runs BEFORE the drop below, so a symbol that comes back on the second pass
    # keeps its place in the universe instead of vanishing from the book.
    if len(symbols_to_process) > 1:
        all_data, recovery = _recover_missing_symbols(
            all_data, symbols_to_process, start_date, end_date
        )
        metrics.data_recovery = recovery
        if recovery["recovered"]:
            metrics.add_warning(
                f"Re-fetched {len(recovery['recovered'])} symbol(s) missed by the "
                f"{start_date:%Y-%m-%d}→{end_date:%Y-%m-%d} batch: "
                f"{', '.join(recovery['recovered'])}"
            )

    # === VALIDATION 4: Remove Failed Tickers ===
    if len(symbols_to_process) > 1:
        valid_tickers = all_data['Close'].dropna(how='all', axis=1).columns
        invalid_tickers = [s for s in symbols_to_process if s not in valid_tickers]
        
        if invalid_tickers:
            invalid_ratio = len(invalid_tickers) / len(symbols_to_process)
            # Every drop is now reported, not just a majority failure: a book
            # built on 28 of 30 names is a different book, and the two symbols
            # survived a dedicated re-fetch before being given up on.
            metrics.add_warning(
                f"{len(invalid_tickers)}/{len(symbols_to_process)} tickers have no data "
                f"after re-fetch: {', '.join(invalid_tickers[:12])}"
                + (" …" if len(invalid_tickers) > 12 else "")
            )
            if invalid_ratio > 0.5:
                console.warning(
                    f"{len(invalid_tickers)}/{len(symbols_to_process)} tickers have no data "
                    "- check symbol validity"
                )
            else:
                console.warning(
                    f"Dropping {len(invalid_tickers)} symbol(s) with no data after re-fetch: "
                    f"{', '.join(invalid_tickers[:12])}"
                    + (" …" if len(invalid_tickers) > 12 else "")
                )
            
            if len(invalid_tickers) == len(symbols_to_process):
                metrics.add_error(
                    "RuntimeError", 
                    "No valid tickers in data - all symbols failed", 
                    "ticker_validation"
                )
                raise ValueError("No valid tickers in data - all symbols failed. Check your universe selection")
            
            all_data = all_data.loc[:, (slice(None), valid_tickers)]
            # Filter the EXISTING order rather than adopting yfinance's column
            # order, which is alphabetical. Snapshot rows are emitted in this
            # order and Equal Weight now selects in it (see nco.compute_nco_
            # portfolio), so a partial failure must not silently reorder the
            # universe.
            _valid = set(valid_tickers)
            symbols_to_process = [s for s in symbols_to_process if s in _valid]
    
    # Update metrics with actual valid symbols
    metrics.symbols_count = len(symbols_to_process)
    
    all_data.columns.names = ['Indicator', 'Symbol']
    oscillator_calculator = LiquidityOscillator(length=20, impact_window=3)

    # The value tape's macro basket — fetched ONCE for the whole panel, over the
    # same window as the names, and shared by every one of them.
    driver_closes = fetch_macro_drivers(start_date, end_date)
    
    # 2. --- Pre-calculate all indicators for all symbols ---
    ticker_indicator_cache = {}
    # A run before a market's close receives TODAY's bar still forming (partial volume, a
    # moving close). Read as a finished session it moved the Nifty book 2.3-4.5% and the
    # state of 2-18% of names depending on the hour (CVG-B2). Each name's bar for today is
    # therefore dropped until its own market has closed (samanvaya._close_utc; 24h markets
    # never close, so their today is always forming); the calendar carry below fills that
    # date from the last completed session, so readings and price are as of that close.
    _now = datetime.now(timezone.utc)
    _today, _hour = pd.Timestamp(_now.date()), _now.hour + _now.minute / 60.0
    _forming: List[str] = []
    _holiday_rows = 0
    # Exchange holidays: dates on which at least half the names that report volume printed
    # a flat (high == low), zero-volume row — yfinance emits such rows for NSE holidays. A
    # single fund's no-trade day is not one: it stays (dropping those could carry a thin ETF
    # out of the panel after five).
    _holiday_dates = pd.DatetimeIndex([])
    try:
        if len(symbols_to_process) > 1:
            _v, _h, _l = (all_data[k].apply(pd.to_numeric, errors="coerce") for k in ("Volume", "High", "Low"))
            _reports = (_v > 0).any()
            if int(_reports.sum()) >= 5:
                _flat = ((_v <= 0) & (_h == _l)).loc[:, _reports]
                _present = _v.loc[:, _reports].notna()
                _share = _flat.sum(axis=1) / _present.sum(axis=1).clip(lower=1)
                _holiday_dates = pd.DatetimeIndex(_share.index[_share >= 0.5])
    except KeyError:
        pass
    _corp_events: List[Tuple[str, pd.Timestamp, float]] = []
    for i, ticker in enumerate(symbols_to_process):
        try:
            if len(symbols_to_process) > 1:
                symbol_df = all_data.xs(ticker, level='Symbol', axis=1).copy()
            else:
                symbol_df = all_data.copy()
                
            symbol_df.columns = [col.lower() for col in symbol_df.columns]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in symbol_df.columns:
                    symbol_df[col] = pd.to_numeric(symbol_df[col], errors='coerce')
            
            symbol_df = symbol_df.dropna(subset=['close', 'volume'])
            # Exchange holiday prints (_holiday_dates above): not a session, so not a bar for
            # the tape engines (CVG-B8; data hygiene, measured -0.04%/yr on Nifty).
            if len(_holiday_dates) and {'high', 'low'}.issubset(symbol_df.columns):
                _flat = (symbol_df.index.isin(_holiday_dates) & (symbol_df['volume'] <= 0)
                         & (symbol_df['high'] == symbol_df['low']))
                if _flat.any():
                    _holiday_rows += int(_flat.sum())
                    symbol_df = symbol_df[~_flat]
            # An unadjusted demerger or mis-dated split (corporate_action_gaps) would reach the
            # tapes as a -40% session and HRP / ERC's covariance as a -40% return: back-adjust
            # the bars before it, as the close history does (MM-B3).
            if {'open', 'high', 'low'}.issubset(symbol_df.columns):
                _ev = corporate_action_gaps(symbol_df[['close']].set_axis([ticker], axis=1),
                                            symbol_df[['open']].set_axis([ticker], axis=1))
                if _ev:
                    symbol_df = back_adjust(symbol_df, _ev,
                                            columns=lambda _s: ['open', 'high', 'low', 'close'])
                    _corp_events.extend(_ev)
            if (len(symbol_df) and symbol_df.index[-1].normalize() == _today
                    and _hour < _close_utc(ticker)):
                symbol_df = symbol_df.iloc[:-1]
                _forming.append(ticker)
            symbol_df.name = ticker
            
            if not symbol_df.empty:
                indicators_df = calculate_all_indicators(symbol_df, oscillator_calculator,
                                                         driver_closes, ticker)
                # calculate_all_indicators returns None for a symbol it cannot
                # compute. Caching that None meant the snapshot loop below
                # dereferenced it (`full_indicator_df.index`) and took the whole
                # run down over one bad symbol — the failure mode the per-symbol
                # skip in this loop's except clause exists to prevent.
                if indicators_df is not None:
                    ticker_indicator_cache[ticker] = indicators_df

        except (pd.errors.DataError, KeyError, IndexError, ValueError) as e:
            # ValueError added: a malformed bar (e.g. NaN high/low reaching an
            # int(np.floor(...)) call before the guard in compute_volume_profile
            # was fixed) previously aborted the WHOLE data-fetch phase for
            # every symbol, not just the offending one. One bad ticker must
            # not fail the run — log and skip it instead.
            console.warning(
                f"Skipping {ticker}: indicator computation failed ({type(e).__name__}: {e})")
            continue

    if _forming:
        console.warning(f"intraday run · today's bar is still forming for {len(_forming)} symbol(s) — "
                        "their readings and prices are as of the last completed close")
    if _corp_events:
        console.detail("corporate-action gaps back-adjusted: " + ", ".join(
            f"{c.replace('.NS', '')} {t:%Y-%m-%d} ×{k:.3f}" for c, t, k in _corp_events))
    if _holiday_rows:
        console.detail(f"holiday prints · dropped {_holiday_rows} flat zero-volume row(s) on "
                       f"{len(_holiday_dates)} exchange holiday(s) before the tapes")
    _skipped_indicators = [s for s in symbols_to_process if s not in ticker_indicator_cache]
    console.detail(
        f"indicators computed for {len(ticker_indicator_cache)} of "
        f"{len(symbols_to_process)} symbols"
        + (f" · {len(_skipped_indicators)} skipped" if _skipped_indicators else "")
    )
    if _skipped_indicators:
        console.warning("No indicators for: " + ", ".join(_skipped_indicators[:12])
                        + (" …" if len(_skipped_indicators) > 12 else ""))

    # 3. --- Generate Daily Snapshots in Memory ---
    snapshot_list: List[Tuple[datetime, pd.DataFrame]] = []
    # Use the index of the downloaded data as the authoritative date range
    date_range = all_data.index.normalize().unique()

    # Warm-up is measured in TRADING bars, not calendar days: ma200 needs 200
    # trading rows (~290 calendar days for NSE/US calendars with weekends +
    # holidays), so a calendar-day cutoff of MAX_INDICATOR_PERIOD (200) days
    # left the first ~90 calendar days (~30% of a typical panel) with NaN
    # ma200/ma90-weekly etc. still being emitted as snapshots. Skip the first
    # MAX_INDICATOR_PERIOD *bars* of date_range instead — the caller already
    # over-fetches enough calendar days (see _load_historical_data's x1.5+30
    # buffer) to have that many bars available before the requested window.
    _warm_dates = set(date_range[:MAX_INDICATOR_PERIOD])

    # Align every symbol onto the SHARED trading calendar with a bounded
    # forward-fill.
    #
    # Each symbol's indicator frame is indexed by its OWN print dates. A thinly
    # traded instrument that simply didn't tick on a given day was therefore
    # absent from that day's snapshot entirely — not "stale", but gone: excluded
    # from the eligible set and from the book. On the ETF
    # universe this silently removed 19 of 30 names from the live run measured
    # on 2026-07-28, because yfinance had not yet published that bar for them.
    # A one-day data gap is not a reason to liquidate a position.
    #
    # Reindexing onto the union calendar with ffill(limit=_STALE_BARS) carries a
    # symbol's last known values across a short gap while still dropping
    # anything that has genuinely stopped trading. The limit is what makes this
    # safe: an unbounded ffill would keep a delisted instrument in the book
    # forever at a frozen price. Leading NaNs are untouched (ffill only
    # propagates forward), so a symbol still cannot appear before its history
    # begins, and the warmup skip above is unaffected.
    _filled_symbols, _filled_bars = 0, 0
    for _t in list(ticker_indicator_cache):
        _df = ticker_indicator_cache[_t]
        if _df is None or _df.empty:
            continue
        _aligned = _df.reindex(date_range)
        _holes = int(_aligned["price"].isna().sum()) if "price" in _aligned.columns else 0
        _aligned = _aligned.ffill(limit=_STALE_BARS)
        _repaired = _holes - (int(_aligned["price"].isna().sum())
                              if "price" in _aligned.columns else 0)
        if _repaired > 0:
            _filled_symbols += 1
            _filled_bars += _repaired
        ticker_indicator_cache[_t] = _aligned
    if _filled_symbols:
        # This is the repair that keeps a symbol which simply did not tick from
        # dropping out of the book for a day, so its size is worth stating: a
        # large number here means the panel is thin, not that the fill is wrong.
        console.detail(
            f"calendar alignment · carried {_filled_bars} missing bar(s) across "
            f"{_filled_symbols} symbol(s) (limit {_STALE_BARS} bars)"
        )

    for snapshot_date in date_range:
        # --- Only start generating snapshots *after* the indicator warmup
        # We also only care about dates *within* the requested range (end_date)
        if snapshot_date in _warm_dates or snapshot_date > end_date:
            continue

        daily_results: List[Dict[str, Any]] = []
        for ticker in symbols_to_process:
            if ticker not in ticker_indicator_cache:
                continue
            
            full_indicator_df = ticker_indicator_cache[ticker]
            
            if snapshot_date not in full_indicator_df.index:
                continue
                
            try:
                indicator_row = full_indicator_df.loc[snapshot_date]
                if indicator_row.isnull().all() or pd.isna(indicator_row.get('price')):
                    continue # Skip if all data is NaN or price is NaN

                indicators = indicator_row.to_dict()
                indicators['symbol'] = ticker.replace('.NS', '')
                indicators['date'] = snapshot_date.strftime('%d %b')
                indicators['% change'] = indicators['% change'] * 100
                
                daily_results.append(indicators)
            except KeyError:
                continue
        
        if daily_results:
            final_df = pd.DataFrame(daily_results)
            for col in COLUMN_ORDER:
                if col not in final_df.columns:
                    final_df[col] = pd.NA
            
            final_df = final_df[COLUMN_ORDER]
            snapshot_list.append((snapshot_date, final_df))

    if snapshot_list:
        console.detail(
            f"snapshots · {len(snapshot_list)} days "
            f"({snapshot_list[0][0]:%Y-%m-%d} → {snapshot_list[-1][0]:%Y-%m-%d}) · "
            f"{len(_warm_dates)} warmup bars skipped · "
            f"{len(snapshot_list[-1][1])} symbols in the latest"
        )
    else:
        console.warning(
            f"No snapshots produced from {len(date_range)} bars — every date fell in the "
            f"{MAX_INDICATOR_PERIOD}-bar warmup or past the end date"
        )

    return snapshot_list


def main():
    """Standalone Streamlit UI for generating indicator snapshots."""
    import streamlit as st
    import zipfile
    import shutil

    st.set_page_config(
        page_title="Indicator Snapshot Generator (Optimized)",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("📊 Daily Indicator Snapshot Generator (Optimized)")

    with st.sidebar:
        st.header("1. Select Date Range")
        today = datetime.now()
        # --- UPDATED: Default start date to be far enough back for indicators
        default_start = today - timedelta(days=MAX_INDICATOR_PERIOD + 90)
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)

        st.header("2. Ticker Universe")
        if SYMBOLS_UNIVERSE:
            st.info(f"Using default ETF universe ({len(SYMBOLS_UNIVERSE)} tickers).")
            with st.expander("View Tickers"):
                st.dataframe(SYMBOLS_UNIVERSE, width='stretch')
        else:
            st.error("No tickers available. Cannot proceed.")
        
        st.header("3. Generate")
        process_button = st.button("Generate Snapshots", type="primary", width='stretch')

    if process_button:
        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
        elif not SYMBOLS_UNIVERSE:
            st.error("Error: No tickers available in the default universe.")
        else:
            symbols_to_process = SYMBOLS_UNIVERSE
            
            fetch_start_date = start_date - timedelta(days=int(MAX_INDICATOR_PERIOD * 1.5) + 30)

            with st.spinner(f"Generating historical data from {fetch_start_date} to {end_date}..."):
                all_generated_data = generate_historical_data(
                    symbols_to_process, 
                    fetch_start_date, # Pass the earlier date for indicator warmup
                    end_date
                )
            
            if not all_generated_data:
                st.error("Failed to generate any data.")
                return

            # --- Filter the generated data to *only* the user's requested date range
            all_generated_data = [
                (date, df) for date, df in all_generated_data 
                if date.date() >= start_date and date.date() <= end_date
            ]
            
            if not all_generated_data:
                st.warning("Data was fetched, but no valid trading days found in the selected Start/End range.")
                return

            base_dir = "data"
            reports_dir = os.path.join(base_dir, "historical")
            zip_dir = os.path.join(base_dir, "zip")

            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(reports_dir)
            os.makedirs(zip_dir)

            st.info("Saving daily snapshots to 'data/historical' folder...")
            progress_bar = st.progress(0)
            last_day_df = pd.DataFrame()

            if all_generated_data:
                for i, (snapshot_date, final_df) in enumerate(all_generated_data):
                    if not final_df.empty:
                        last_day_df = final_df
                        filename = os.path.join(reports_dir, f"{snapshot_date.strftime('%Y-%m-%d')}.csv")
                        final_df.to_csv(filename, index=False, float_format='%.2f')
                    
                    progress_bar.progress((i + 1) / len(all_generated_data))
            
                zip_file_name_only = f"indicator_reports_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.zip"
                zip_full_path = os.path.join(zip_dir, zip_file_name_only)
                
                with zipfile.ZipFile(zip_full_path, 'w') as zipf:
                    for root, _, files in os.walk(reports_dir):
                        for file in files:
                            zipf.write(os.path.join(root, file), os.path.join(os.path.basename(root), file))

                st.success("✅ Snapshots and Zip file generated successfully in the 'data' folder!")
                
                st.subheader(f"Data for {end_date.strftime('%Y-%m-%d')} (Last Day)")
                if not last_day_df.empty:
                    st.dataframe(last_day_df[COLUMN_ORDER].round(2))
                
                with open(zip_full_path, "rb") as fp:
                    st.download_button(
                        label="⬇️ Download All Reports (.zip)",
                        data=fp,
                        file_name=zip_file_name_only,
                        mime="application/zip"
                    )
            else:
                st.warning("No data was generated for the selected date range.")

__all__ = [
    'LiquidityOscillator',
    'resample_data',
    'calculate_rsi',
    'calculate_all_indicators',
    'get_default_universe',
    'generate_historical_data',
    'SYMBOLS_UNIVERSE',
    'MAX_INDICATOR_PERIOD',
]

if __name__ == "__main__":
    main()
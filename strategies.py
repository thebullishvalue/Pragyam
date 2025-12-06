import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import io
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

# --- Base Classes and Utilities ---
# Note: Copied from pragati.py to make this module self-contained.

def fix_csv_export(df: pd.DataFrame) -> bytes:
    """Utility (though not used by strategies directly) kept for potential future module use."""
    output = io.BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()

class BaseStrategy(ABC):
    @abstractmethod
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        pass

    def _clean_data(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """A standardized data cleaning utility for strategies."""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for this strategy: {missing_cols}")

        df_copy = df.copy()
        
        rsi_columns = ['rsi latest', 'rsi weekly']
        for col in rsi_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(50.0)

        ma_columns = ['ma90 latest', 'ma200 latest', 'ma90 weekly', 'ma200 weekly', 'ma20 latest', 'ma20 weekly']
        fallback_ma = df_copy['price'].median()
        for col in ma_columns:
            if col in df_copy.columns:
                invalid = (df_copy[col].isna()) | (df_copy[col] <= 0) | (df_copy[col] / df_copy['price'] > 10) | (df_copy[col] / df_copy['price'] < 0.1)
                df_copy[col] = np.where(invalid, fallback_ma, df_copy[col])

        df_copy = df_copy.replace([np.inf, -np.inf], 0).fillna(0)
        return df_copy

    def _allocate_portfolio(self, df: pd.DataFrame, sip_amount: float) -> pd.DataFrame:
        """Standardized portfolio allocation and cash distribution logic."""
        if 'weightage' not in df.columns or df['weightage'].sum() <= 0:
            return pd.DataFrame(columns=['symbol', 'price', 'weightage_pct', 'units', 'value'])

        # Cap and redistribute weights (10% max, 1% min)
        for _ in range(10): # Iterate to allow weights to settle
            df['weightage'] = df['weightage'].clip(lower=0.01, upper=0.10)
            total_w = df['weightage'].sum()
            if total_w > 0:
                df['weightage'] = df['weightage'] / total_w
            if abs(df['weightage'].sum() - 1.0) < 1e-6:
                break
        
        df['weightage_pct'] = df['weightage'] * 100
        df = df.sort_values('weightage', ascending=False).reset_index(drop=True)

        # Allocate units and handle remaining cash
        df['units'] = np.floor((sip_amount * df['weightage']) / df['price'])
        allocated_capital = (df['units'] * df['price']).sum()
        remaining_cash = sip_amount - allocated_capital

        # Re-allocate remaining cash to top-weighted stocks
        for idx in df.index:
            if remaining_cash >= df.at[idx, 'price']:
                df.at[idx, 'units'] += 1
                remaining_cash -= df.at[idx, 'price']
            else:
                break # Stop if cash can't even buy the next top stock

        df['value'] = df['units'] * df['price']
        return df[['symbol', 'price', 'weightage_pct', 'units', 'value']].reset_index(drop=True)

# =====================================
# NEW: AlphaPrime Strategy (Regime-Adaptive Divergence Hunter)
# =====================================

class AlphaPrime(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        AlphaPrime: A Regime-Adaptive Divergence Hunter.
        
        Core Logic:
        1. Explicitly targets the 'Tier 2' (Bullish Divergence) signal of the Unified Booster.
        2. Detects Price vs Oscillator Divergence (Price making lows, Oscillator making highs).
        3. Adapts to Volatility: Demands deeper Z-scores in high volatility regimes.
        """
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'zscore weekly', 
            'ma20 latest', 'ma200 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Calculate Regime Factors
        # Volatility Ratio: Current volatility vs Price (Normalized)
        df['vol_ratio'] = df['dev20 latest'] / df['price']
        
        # Regime Multiplier: If volatility is high (>3%), we become defensive (demand higher quality setup)
        df['regime_strictness'] = np.where(df['vol_ratio'] > 0.03, 1.5, 1.0)

        # 2. Divergence Detection (The Alpha Source)
        # Logic: Price is below MA20 (pullback) BUT Oscillator Momentum (9ema) is above Signal (21ema)
        # This is a classic "Hidden Divergence" or early reversal setup.
        
        df['price_structure'] = np.where(df['price'] < df['ma20 latest'], 1, 0)
        df['momentum_structure'] = np.where(df['9ema osc latest'] > df['21ema osc latest'], 1, 0)
        
        # Divergence Strength: How oversold is it while having this structure?
        # We use Z-Score to quantify "Oversold". Lower is better for entry, but we need the momentum turn.
        df['divergence_quality'] = np.where(
            (df['price_structure'] == 1) & (df['momentum_structure'] == 1),
            np.abs(np.clip(df['zscore latest'], -4, 0)), # Score is higher if Z-score is deeper (e.g. -3 -> 3)
            0
        )

        # 3. Unified Booster Alignment (Pre-empting the Boost)
        # The Unified Booster rewards: Osc < -5 AND Rising. 
        df['unified_alignment'] = np.where(
            (df['osc latest'] < -5) & (df['osc latest'] > df['osc latest'].shift(1).fillna(0)),
            1.5, # 50% Bonus for aligning with Booster logic
            1.0
        )

        # 4. Deep Value Trap Filter (Avoiding "Catching a Knife")
        # If Weekly RSI is < 30 (Crash), we kill the score UNLESS Z-score is extreme (<-3)
        df['safety_lock'] = np.where(
            (df['rsi weekly'] < 30) & (df['zscore weekly'] > -3.0),
            0.1, # Penalty
            1.0
        )

        # 5. Composite Scoring
        # Score = (Divergence Quality + Normalized RSI Reversal) * Alignment * Safety / Strictness
        
        # RSI Reversal Potential: (50 - RSI) normalized. Higher score for lower RSI.
        df['rsi_potential'] = np.clip((50 - df['rsi latest']) / 50, 0, 1)

        df['alpha_score'] = (
            (df['divergence_quality'] * 0.6 + df['rsi_potential'] * 0.4) 
            * df['unified_alignment'] 
            * df['safety_lock']
            / df['regime_strictness']
        )
        
        # Filter: Only take positive alpha scores
        df = df[df['alpha_score'] > 0.1].copy()

        # 6. Weighting
        total_score = df['alpha_score'].sum()
        df['weightage'] = df['alpha_score'] / total_score if total_score > 0 else 0

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: QuantumVelocity Strategy (Statistical Momentum Acceleration)
# =====================================

class QuantumVelocity(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        QuantumVelocity: Statistical Momentum Acceleration.
        
        Core Logic:
        1. Targets 'Tier 1' (Confirmed Bullish) & 'Tier 3' (Oversold) Booster signals.
        2. 'Coiled Spring' Theory: Potential Energy (Z-Score) -> Kinetic Energy (Velocity).
        3. Identifies assets where the *derivative* of momentum is positive and accelerating.
        """
        required_columns = [
            'symbol', 'price', 'osc latest', 'osc weekly', 
            'zscore latest', 'zscore weekly', 'dev20 latest', 'ma90 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Potential Energy (Z-Score Extension)
        # The further price is from the mean (negative z-score), the higher the potential snap-back energy.
        # We clip at -4 (extreme) and 0 (mean). We want negative Z-scores.
        df['potential_energy'] = np.abs(np.clip(df['zscore latest'], -4, 0))

        # 2. Kinetic Energy (Momentum Velocity)
        # How fast is the Oscillator moving? 
        # Since we don't have 'osc previous' directly in columns usually, we infer velocity 
        # from the spread between 9EMA and 21EMA (available in other strategies, calculated here if needed)
        # If not available, we assume 'osc latest' vs 'osc weekly' proxies trend speed.
        
        # Ideally, we use the pre-calculated EMAs if available in df, else fallback.
        if '9ema osc latest' in df.columns and '21ema osc latest' in df.columns:
            df['velocity'] = df['9ema osc latest'] - df['21ema osc latest']
        else:
            # Fallback: Proxy velocity using latest vs weekly (rough approximation of speed)
            df['velocity'] = df['osc latest'] - df['osc weekly']

        # 3. The "Quantum Leap" Condition
        # We want High Potential Energy (Deep Oversold) + Positive Velocity (Turning Up)
        # OR High Positive Velocity (Strong Trend) + Low Volatility (Sustainable)
        
        # Condition A: Mean Reversion Snap (The "V" Bottom)
        # Z-Score < -2 (High Potential) AND Velocity > 0 (Started Moving)
        df['score_snap'] = np.where(
            (df['zscore latest'] < -2.0) & (df['velocity'] > 0),
            df['potential_energy'] * 1.5, # Multiplier for the snap
            0
        )

        # Condition B: Momentum Breakout (The "Rocket")
        # Price > MA90 (Trend Up) AND Velocity > 5 (Strong Speed) AND Volatility Low
        vol_normalized = df['dev20 latest'] / df['price']
        df['score_rocket'] = np.where(
            (df['price'] > df['ma90 latest']) & (df['velocity'] > 5) & (vol_normalized < 0.02),
            (df['velocity'] / 10) * 1.2,
            0
        )

        # 4. Unified Booster Alignment
        # Booster Tier 1 requires "Confirmed Bullish". 
        # High positive velocity usually triggers the "Green Circle" logic in the booster.
        df['booster_proxy'] = np.where(df['velocity'] > 0, 1.2, 0.8)

        # 5. Final Quantum Score
        df['quantum_score'] = (df['score_snap'] + df['score_rocket']) * df['booster_proxy']

        # Filter out zero scores
        df = df[df['quantum_score'] > 0.01].copy()

        # 6. Weighting
        total_score = df['quantum_score'].sum()
        df['weightage'] = df['quantum_score'] / total_score if total_score > 0 else 0

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# PR_v1 Strategy Implementation
# =====================================

class PRStrategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        PR_v1 Strategy: Original Pragati Logic with Full Multiplier Fidelity
        """
        # --- Data Validation & NaN Handling (aligned with PR_v1) ---
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)


        # --- Multiplier Calculations (Vectorized, PR_v1 Exact Logic) ---

        # RSI Multiplier
        weighted_rsi = df['rsi weekly'] * 0.55 + df['rsi latest'] * 0.45
        df['rsi_mult'] = np.select(
            [weighted_rsi < 30, (weighted_rsi >= 30) & (weighted_rsi < 50),
             (weighted_rsi >= 50) & (weighted_rsi < 70), weighted_rsi >= 70],
            [3.5 - (weighted_rsi / 30) * 1.5,
             2 - (weighted_rsi - 30) / 20,
             1 - (weighted_rsi - 50) / 20,
             0.3 + (100 - weighted_rsi) / 30],
            default=1.0
        )

        # OSC Multiplier (12-tier, corrected order)
        osc_w, osc_d = df['osc weekly'], df['osc latest']
        df['osc_mult'] = np.select(
            [(osc_w < -80) & (osc_d < -95), (osc_w < -80) & (osc_d >= -95),
             (osc_w < -70) & (osc_d < -90), (osc_w < -70) & (osc_d >= -90),
             (osc_w < -60) & (osc_d < -85),
             (osc_w < -50) & (osc_d < -80),
             (osc_w < -40) & (osc_d < -70),
             (osc_w < -30) & (osc_d < -60),
             (osc_w < -20) & (osc_d < -50),
             (osc_w < -10) & (osc_d < -40),
             (osc_w < 0) & (osc_d < -30),
             osc_d < -95,
             (osc_d > 80) & (osc_w > 70)],
            [3.5, 3.2, 2.8, 2.5, 2.3, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 2.0, 0.2],
            default=1.0
        )

        # 9EMA OSC Multiplier
        ema9_w, ema9_d = df['9ema osc weekly'], df['9ema osc latest']
        df['ema_osc_mult'] = np.select(
            [(ema9_w < -80) & (ema9_d < -90), (ema9_w < -80) & (ema9_d >= -90),
             (ema9_w < -70) & (ema9_d < -80), (ema9_w < -70) & (ema9_d >= -80),
             (ema9_w < -60) & (ema9_d < -70),
             (ema9_w < -50) & (ema9_d < -60),
             (ema9_w < -40) & (ema9_d < -50),
             (ema9_w < -30) & (ema9_d < -40),
             ema9_d < -90],
            [3.5, 3.2, 2.8, 2.5, 2.3, 2.0, 1.8, 1.6, 2.0],
            default=1.0
        )

        # 21EMA OSC Multiplier
        ema21_w, ema21_d = df['21ema osc weekly'], df['21ema osc latest']
        df['21ema_osc_mult'] = np.select(
            [(ema21_w < -80) & (ema21_d < -90), (ema21_w < -80) & (ema21_d >= -90),
             (ema21_w < -70) & (ema21_d < -80), (ema21_w < -70) & (ema21_d >= -80),
             (ema21_w < -60) & (ema21_d < -70),
             (ema21_w < -50) & (ema21_d < -60),
             (ema21_w < -40) & (ema21_d < -50),
             (ema21_w < -30) & (ema21_d < -40),
             ema21_d < -90],
            [3.5, 3.2, 2.8, 2.5, 2.3, 2.0, 1.8, 1.6, 2.0],
            default=1.0
        )

        # Z-Score Multiplier
        z_w, z_d = df['zscore weekly'], df['zscore latest']
        df['zscore_mult'] = np.select(
            [(z_w < -2.5) & (z_d < -2.5), (z_w < -2.5) & (z_d >= -2.5),
             (z_w < -2.0) & (z_d < -2.0),
             (z_w < -1.5) & (z_d < -1.5),
             (z_w < -1.2) & (z_d < -1.2),
             (z_w < -1.0) & (z_d < -1.0),
             z_d < -2.5],
            [3.5, 3.2, 2.8, 2.5, 2.2, 2.0, 2.0],
            default=1.0
        )

        # Spread Multiplier
        eps = 1e-6
        safe_div = lambda num, den: np.clip(num * 100 / np.where(den > eps, den, eps), -100, 100)
        s90d = safe_div(df['ma90 latest'] - df['price'], df['ma90 latest'])
        s200d = safe_div(df['ma200 latest'] - df['price'], df['ma200 latest'])
        s90w = safe_div(df['ma90 weekly'] - df['price'], df['ma90 weekly'])
        s200w = safe_div(df['ma200 weekly'] - df['price'], df['ma200 weekly'])
        ws90 = s90d * 0.6 + s90w * 0.4
        ws200 = s200d * 0.6 + s200w * 0.4
        df['spread_mult'] = np.select(
            [(ws90 > 1.5) & (ws200 > 1.5) & (df['rsi latest'] < 40),
             (ws90 < -1.5) & (ws200 < -1.5) & (df['rsi latest'] > 70)],
            [3.5, 0.5],
            default=1.0
        )

        # Bollinger Multiplier (using MA20/dev20)
        bd, bw = df['ma20 latest'], df['ma20 weekly']
        dd, dw = 2 * df['dev20 latest'], 2 * df['dev20 weekly']
        lb_d, lb_w = bd - dd, bw - dw
        ub_d, ub_w = bd + dd, bw + dw
        w_lb = lb_d * 0.6 + lb_w * 0.4
        w_ub = ub_d * 0.6 + ub_w * 0.4
        w_dev = dd * 0.6 + dw * 0.4
        std_below = (w_lb - df['price']) / np.maximum(w_dev, eps)
        std_above = (df['price'] - w_ub) / np.maximum(w_dev, eps)
        df['bollinger_mult'] = np.select(
            [(df['price'] < w_lb) & (df['rsi latest'] < 40) & (std_below > 3.0),
             (df['price'] < w_lb) & (df['rsi latest'] < 40) & (std_below > 2.0),
             (df['price'] < w_lb) & (df['rsi latest'] < 40) & (std_below > 1.0),
             (df['price'] < w_lb) & (df['rsi latest'] < 40),
             (df['price'] > w_ub) & (df['rsi latest'] > 70) & (std_above > 3.0),
             (df['price'] > w_ub) & (df['rsi latest'] > 70) & (std_above > 2.0),
             (df['price'] > w_ub) & (df['rsi latest'] > 70) & (std_above > 1.0),
             (df['price'] > w_ub) & (df['rsi latest'] > 70)],
            [2.8, 2.5, 2.0, 1.5, 0.5, 0.6, 0.7, 0.8],
            default=1.0
        )

        # Trend Strength & Weekly Boost
        df['trend_strength'] = np.select(
            [(df['osc latest'] < -50), (df['osc weekly'] < -50),
             (df['9ema osc latest'] > 0) & (df['21ema osc latest'] > 0) & (df['osc latest'] > 0) & (df['rsi latest'] > 60)],
            [1.3, 1.5, 0.8],
            default=1.0
        )
        df['weekly_oversold_boost'] = np.where(df['osc weekly'] < -20, 1.2, 1.0)

        # --- Final Weighting (Fixed Weights) ---
        weights = {'rsi': 0.15, 'osc': 0.20, 'ema_osc': 0.15, '21ema_osc': 0.10,
                   'zscore': 0.15, 'spread': 0.15, 'bollinger': 0.10}
        df['base_mult'] = (
            df['rsi_mult'] * weights['rsi'] +
            df['osc_mult'] * weights['osc'] +
            df['ema_osc_mult'] * weights['ema_osc'] +
            df['21ema_osc_mult'] * weights['21ema_osc'] +
            df['zscore_mult'] * weights['zscore'] +
            df['spread_mult'] * weights['spread'] +
            df['bollinger_mult'] * weights['bollinger']
        )
        df['final_mult'] = df['base_mult'] * df['trend_strength'] * df['weekly_oversold_boost']

        # Normalize to weights
        total_mult = df['final_mult'].sum()
        if total_mult <= 0 or not np.isfinite(total_mult):
            df['weightage'] = 1.0 / len(df)
        else:
            df['weightage'] = df['final_mult'] / total_mult

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# CL_v1 Strategy Implementation
# =====================================

class CL1Strategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        class QuantitativeETFAnalyzer:
            def __init__(self):
                self.factor_weights = {}
                self.regime_indicators = {}
                self.quality_threshold = 0.6 

            def validate_and_prepare_data(self, df):
                required_columns = ['symbol', 'price', 'rsi latest', 'rsi weekly',
                                'osc latest', 'osc weekly', '9ema osc latest', '9ema osc weekly',
                                '21ema osc latest', '21ema osc weekly', 'zscore latest', 'zscore weekly',
                                'date', 'ma90 latest', 'ma200 latest', 'ma90 weekly', 'ma200 weekly',
                                'dev20 latest', 'dev20 weekly']

                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                df = self._intelligent_nan_handling(df)
                df['data_quality_score'] = self._calculate_data_quality(df)
                df = df[df['data_quality_score'] >= self.quality_threshold]
                return df

            def _intelligent_nan_handling(self, df):
                rsi_columns = ['rsi latest', 'rsi weekly']
                for col in rsi_columns:
                    df[col] = df[col].fillna(50)
                osc_columns = [col for col in df.columns if 'osc' in col.lower()]
                for col in osc_columns:
                    df[col] = df[col].fillna(df[col].median())
                zscore_columns = [col for col in df.columns if 'zscore' in col.lower()]
                for col in zscore_columns:
                    df[col] = df[col].fillna(0)
                ma_columns = [col for col in df.columns if col.startswith('ma')]
                for col in ma_columns:
                    df[col] = df[col].fillna(df['price'])
                for col in df.columns:
                    if col not in rsi_columns + osc_columns + zscore_columns + ma_columns:
                        df[col] = df[col].ffill().bfill().fillna(0)
                return df

            def _calculate_data_quality(self, df):
                quality_factors = []
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                completeness = 1 - (df[numeric_cols] == 0).sum(axis=1) / len(numeric_cols)
                quality_factors.append(completeness)
                key_metrics = ['rsi latest', 'rsi weekly', 'osc latest', 'osc weekly']
                for metric in key_metrics:
                    if metric in df.columns:
                        z_scores = np.abs(stats.zscore(df[metric].fillna(0)))
                        consistency = np.clip(1 - z_scores / 5, 0, 1) 
                        quality_factors.append(consistency)
                return np.mean(quality_factors, axis=0)

            def detect_market_regime(self, df):
                avg_rsi = df['rsi latest'].mean()
                avg_osc = df['osc latest'].mean()
                volatility_regime = df['dev20 latest'].mean() / df['price'].mean()
                momentum_consistency = np.corrcoef(df['osc latest'], df['osc weekly'])[0, 1]

                if avg_rsi < 35 and avg_osc < -40 and volatility_regime > 0.02:
                    regime = "CRISIS"
                    confidence = 0.9
                elif avg_rsi < 45 and avg_osc < -20 and momentum_consistency > 0.7:
                    regime = "BEAR_TREND"
                    confidence = 0.8
                elif avg_rsi > 65 and avg_osc > 30 and volatility_regime < 0.015:
                    regime = "BULL_EUPHORIA"
                    confidence = 0.85
                elif avg_rsi > 55 and avg_osc > 10 and momentum_consistency > 0.6:
                    regime = "BULL_TREND"
                    confidence = 0.75
                else:
                    regime = "NEUTRAL_RANGE"
                    confidence = 0.6

                self.regime_indicators = {
                    'regime': regime,
                    'confidence': confidence,
                    'avg_rsi': avg_rsi,
                    'avg_osc': avg_osc,
                    'volatility': volatility_regime,
                    'momentum_consistency': momentum_consistency
                }
                return regime, confidence

            def calculate_statistical_anomaly_score(self, df):
                anomaly_components = []
                rsi_divergence = np.abs(df['rsi latest'] - df['rsi weekly']) / (df['rsi latest'] + df['rsi weekly'] + 1e-6)
                rsi_anomaly = np.where((df['rsi latest'] < 30) & (df['rsi weekly'] < 35) & (rsi_divergence < 0.2),
                                    3.0 - (df['rsi latest'] + df['rsi weekly']) / 20, 0)
                anomaly_components.append(rsi_anomaly)
                osc_cascade = np.where((df['osc latest'] < -70) & (df['osc weekly'] < -60) &
                                    (df['9ema osc latest'] < df['21ema osc latest']),
                                    2.5 + np.abs(df['osc latest'] + df['osc weekly']) / 100, 0)
                anomaly_components.append(osc_cascade)
                zscore_significance = np.where((df['zscore latest'] < -2.0) & (df['zscore weekly'] < -1.5),
                                            np.minimum(np.abs(df['zscore latest']) + np.abs(df['zscore weekly']), 5.0), 0)
                anomaly_components.append(zscore_significance)
                df['anomaly_score'] = np.sum(anomaly_components, axis=0)
                return df

            def calculate_momentum_score(self, df):
                momentum_score = (
                    (df['rsi latest'] < 40).astype(int) * 1.5 +
                    (df['rsi weekly'] < 45).astype(int) * 1.0 +
                    (df['osc latest'] < -50).astype(int) * 1.2 +
                    (df['osc weekly'] < -40).astype(int) * 0.8
                )
                df['momentum_score'] = momentum_score
                return df

            def calculate_risk_adjusted_score(self, df):
                df['risk_score'] = np.where(
                    (df['dev20 latest'] / df['price'] < 0.015) & (np.abs(df['zscore latest']) < 2.0),
                    2.0,
                    np.where((df['dev20 latest'] / df['price'] < 0.03) & (np.abs(df['zscore latest']) < 3.0), 1.0, 0.5)
                )
                return df

            def calculate_composite_score(self, df):
                regime, _ = self.detect_market_regime(df)
                if regime == "CRISIS":
                    self.factor_weights = {'anomaly': 0.4, 'momentum': 0.2, 'risk_adjusted': 0.2, 'consistency': 0.1, 'quality': 0.1}
                elif regime == "BEAR_TREND":
                    self.factor_weights = {'anomaly': 0.3, 'momentum': 0.3, 'risk_adjusted': 0.2, 'consistency': 0.1, 'quality': 0.1}
                else:
                    self.factor_weights = {'anomaly': 0.25, 'momentum': 0.25, 'risk_adjusted': 0.2, 'consistency': 0.15, 'quality': 0.15}

                df['consistency_score'] = np.where(
                    np.abs(df['rsi latest'] - df['rsi weekly']) < 10,
                    2.0,
                    np.where(np.abs(df['rsi latest'] - df['rsi weekly']) < 20, 1.0, 0.5)
                )

                scaler = StandardScaler()
                score_columns = ['anomaly_score', 'momentum_score', 'risk_score', 'consistency_score', 'data_quality_score']
                normalized_scores = scaler.fit_transform(df[score_columns])

                df['composite_score'] = np.sum(normalized_scores * list(self.factor_weights.values()), axis=1)
                return df

            def allocate_portfolio(self, df, sip_amount):
                df = df.sort_values('composite_score', ascending=False)
                total_score = df['composite_score'].sum()
                if total_score > 0:
                    df['weightage'] = df['composite_score'] / total_score
                else:
                    df['weightage'] = 1 / len(df) if len(df) > 0 else 0
                return df

        try:
            analyzer = QuantitativeETFAnalyzer()
            df_prepared = analyzer.validate_and_prepare_data(df.copy())
            if df_prepared.empty:
                return pd.DataFrame()
            regime, confidence = analyzer.detect_market_regime(df_prepared)
            df_prepared = analyzer.calculate_statistical_anomaly_score(df_prepared)
            df_prepared = analyzer.calculate_momentum_score(df_prepared)
            df_prepared = analyzer.calculate_risk_adjusted_score(df_prepared)
            df_prepared = analyzer.calculate_composite_score(df_prepared)
            portfolio_df = analyzer.allocate_portfolio(df_prepared, sip_amount)
            return self._allocate_portfolio(portfolio_df, sip_amount)
        except Exception as e:
            logging.error(f"Error in CL1Strategy portfolio generation: {str(e)}")
            raise

# =====================================
# CL_v2 Strategy Implementation
# =====================================

class CL2Strategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        class QuantitativeETFAnalyzer:
            def __init__(self):
                self.factor_weights = {}
                self.regime_indicators = {}
                self.quality_threshold = 0.6

            def validate_and_prepare_data(self, df):
                required_columns = ['symbol', 'price', 'rsi latest', 'rsi weekly',
                                   'osc latest', 'osc weekly', '9ema osc latest', '9ema osc weekly',
                                   '21ema osc latest', '21ema osc weekly', 'zscore latest', 'zscore weekly',
                                   'date', 'ma90 latest', 'ma200 latest', 'ma90 weekly', 'ma200 weekly',
                                   'dev20 latest', 'dev20 weekly']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                df = self._intelligent_nan_handling(df)
                df['data_quality_score'] = self._calculate_data_quality(df)
                return df

            def _intelligent_nan_handling(self, df):
                rsi_columns = ['rsi latest', 'rsi weekly']
                for col in rsi_columns:
                    df[col] = df[col].fillna(50)
                osc_columns = [col for col in df.columns if 'osc' in col.lower()]
                for col in osc_columns:
                    df[col] = df[col].fillna(df[col].median())
                zscore_columns = [col for col in df.columns if 'zscore' in col.lower()]
                for col in zscore_columns:
                    df[col] = df[col].fillna(0)
                ma_columns = [col for col in df.columns if col.startswith('ma')]
                for col in ma_columns:
                    df[col] = df[col].fillna(df['price'])
                for col in df.columns:
                    if col not in rsi_columns + osc_columns + zscore_columns + ma_columns:
                        df[col] = df[col].ffill().bfill().fillna(0)
                return df

            def _calculate_data_quality(self, df):
                quality_factors = []
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                completeness = 1 - (df[numeric_cols] == 0).sum(axis=1) / len(numeric_cols)
                quality_factors.append(completeness)
                key_metrics = ['rsi latest', 'rsi weekly', 'osc latest', 'osc weekly']
                for metric in key_metrics:
                    if metric in df.columns:
                        z_scores = np.abs(stats.zscore(df[metric].fillna(0)))
                        consistency = np.clip(1 - z_scores / 5, 0, 1)
                        quality_factors.append(consistency)
                return np.mean(quality_factors, axis=0)

            def detect_market_regime(self, df):
                avg_rsi = df['rsi latest'].mean()
                avg_osc = df['osc latest'].mean()
                volatility_regime = df['dev20 latest'].mean() / df['price'].mean()
                momentum_consistency = np.corrcoef(df['osc latest'], df['osc weekly'])[0, 1]
                if avg_rsi < 35 and avg_osc < -40 and volatility_regime > 0.02:
                    regime, confidence = "CRISIS", 0.9
                elif avg_rsi < 45 and avg_osc < -20 and momentum_consistency > 0.7:
                    regime, confidence = "BEAR_TREND", 0.8
                elif avg_rsi > 65 and avg_osc > 30 and volatility_regime < 0.015:
                    regime, confidence = "BULL_EUPHORIA", 0.85
                elif avg_rsi > 55 and avg_osc > 10 and momentum_consistency > 0.6:
                    regime, confidence = "BULL_TREND", 0.75
                else:
                    regime, confidence = "NEUTRAL_RANGE", 0.6
                self.regime_indicators = {
                    'regime': regime,
                    'confidence': confidence,
                    'avg_rsi': avg_rsi,
                    'avg_osc': avg_osc,
                    'volatility': volatility_regime,
                    'momentum_consistency': momentum_consistency
                }
                return regime, confidence

            def calculate_enhanced_technical_multipliers(self, df):
                df['rsi_mult'] = df.apply(self._calculate_rsi_mult_enhanced, axis=1)
                df['osc_mult'] = df.apply(self._calculate_osc_mult_enhanced, axis=1)
                df['ema9_osc_mult'] = df.apply(self._calculate_9ema_osc_mult_enhanced, axis=1)
                df['ema21_osc_mult'] = df.apply(self._calculate_21ema_osc_mult_enhanced, axis=1)
                df['zscore_mult'] = df.apply(self._calculate_zscore_mult_enhanced, axis=1)
                df['trend_strength'] = df.apply(self._calculate_trend_strength_enhanced, axis=1)
                df['spread_mult'] = df.apply(self._calculate_spread_mult_enhanced, axis=1)
                df['bollinger_mult'] = df.apply(self._calculate_bollinger_mult_enhanced, axis=1)
                return df

            def _calculate_rsi_mult_enhanced(self, row):
                weighted_rsi = (row['rsi weekly'] * 0.55 + row['rsi latest'] * 0.45)
                if weighted_rsi < 30:
                    base_mult = 3.5 - (weighted_rsi / 30) * 1.5
                elif weighted_rsi < 50:
                    base_mult = 2 - (weighted_rsi - 30) / 20
                elif weighted_rsi < 70:
                    base_mult = 1 - (weighted_rsi - 50) / 20
                else:
                    base_mult = 0.3 + (100 - weighted_rsi) / 30
                timeframe_consistency = 1.1 if abs(row['rsi latest'] - row['rsi weekly']) < 10 else 1.0
                return base_mult * timeframe_consistency

            def _calculate_osc_mult_enhanced(self, row):
                osc_weekly, osc_latest = row['osc weekly'], row['osc latest']
                if osc_weekly < -80 and osc_latest < -95: return 3.5
                elif osc_weekly < -80: return 3.2
                elif osc_weekly < -70 and osc_latest < -90: return 2.8
                elif osc_weekly < -70: return 2.5
                elif osc_weekly < -60 and osc_latest < -85: return 2.3
                elif osc_weekly < -50 and osc_latest < -80: return 2.0
                elif osc_weekly < -40 and osc_latest < -70: return 1.8
                elif osc_weekly < -30 and osc_latest < -60: return 1.6
                elif osc_weekly < -20 and osc_latest < -50: return 1.5
                elif osc_weekly < -10 and osc_latest < -40: return 1.4
                elif osc_weekly < 0 and osc_latest < -30: return 1.3
                elif osc_latest < -95: return 2.0
                elif osc_latest > 80 and osc_weekly > 70: return 0.2
                else: return 0.1

            def _calculate_9ema_osc_mult_enhanced(self, row):
                ema_weekly, ema_latest = row['9ema osc weekly'], row['9ema osc latest']
                if ema_weekly < -80 and ema_latest < -90: return 3.5
                elif ema_weekly < -80: return 3.2
                elif ema_weekly < -70 and ema_latest < -80: return 2.8
                elif ema_weekly < -70: return 2.5
                elif ema_weekly < -60 and ema_latest < -70: return 2.3
                elif ema_weekly < -50 and ema_latest < -60: return 2.0
                elif ema_weekly < -40 and ema_latest < -50: return 1.8
                elif ema_weekly < -30 and ema_latest < -40: return 1.6
                elif ema_latest < -90: return 2.0
                else: return 0.1

            def _calculate_21ema_osc_mult_enhanced(self, row):
                ema_21_weekly, ema_21_latest = row['21ema osc weekly'], row['21ema osc latest']
                if ema_21_weekly < -80 and ema_21_latest < -90: return 3.5
                elif ema_21_weekly < -80: return 3.2
                elif ema_21_weekly < -70 and ema_21_latest < -80: return 2.8
                elif ema_21_weekly < -70: return 2.5
                elif ema_21_weekly < -60 and ema_21_latest < -70: return 2.3
                elif ema_21_weekly < -50 and ema_21_latest < -60: return 2.0
                elif ema_21_weekly < -40 and ema_21_latest < -50: return 1.8
                elif ema_21_weekly < -30 and ema_21_latest < -40: return 1.6
                elif ema_21_latest < -90: return 2.0
                else: return 0.1

            def _calculate_zscore_mult_enhanced(self, row):
                zscore_weekly, zscore_latest = row['zscore weekly'], row['zscore latest']
                if zscore_weekly < -2.5 and zscore_latest < -3: return 3.5
                elif zscore_weekly < -2.5: return 3.2
                elif zscore_weekly < -2 and zscore_latest < -2.5: return 2.8
                elif zscore_weekly < -1.5 and zscore_latest < -2: return 2.5
                elif zscore_weekly < -1.2 and zscore_latest < -1.8: return 2.2
                elif zscore_weekly < -1 and zscore_latest < -1.5: return 2.0
                elif zscore_latest < -3: return 2.0
                else: return 0.1

            def _calculate_trend_strength_enhanced(self, row):
                if row['9ema osc latest'] > row['21ema osc latest'] and row['osc latest'] < -50:
                    return 1.3
                elif row['9ema osc weekly'] > row['21ema osc weekly'] and row['osc weekly'] < -50:
                    return 1.5
                elif row['9ema osc latest'] > 0 and row['21ema osc latest'] > 0 and row['osc latest'] > 0:
                    return 0.7
                else:
                    return 1.0

            def _calculate_spread_mult_enhanced(self, row, spreadup=1.5, spreadown=-1.5, lower_thres=40, higher_thres=70):
                epsilon = 1e-6
                def safe_div(n, d): return n * 100 / (d if d != 0 else epsilon)
                spread90_latest = safe_div(row['ma90 latest'] - row['price'], row['ma90 latest'])
                spread200_latest = safe_div(row['ma200 latest'] - row['price'], row['ma200 latest'])
                spread90_weekly = safe_div(row['ma90 weekly'] - row['price'], row['ma90 weekly'])
                spread200_weekly = safe_div(row['ma200 weekly'] - row['price'], row['ma200 weekly'])
                weighted_spread90 = spread90_latest * 0.60 + spread90_weekly * 0.40
                weighted_spread200 = spread200_latest * 0.60 + spread200_weekly * 0.40
                if weighted_spread90 > spreadup and weighted_spread200 > spreadup and row['rsi latest'] < lower_thres:
                    return 3.5
                elif weighted_spread90 < spreadown and weighted_spread200 < spreadown and row['rsi latest'] > higher_thres:
                    return 0.5
                else:
                    return 1.0

            def _calculate_bollinger_mult_enhanced(self, row, mult=2.0, lower_thres=40, higher_thres=70):
                basis_latest = row['ma90 latest']
                dev_latest = mult * row['dev20 latest']
                upper_s_latest = basis_latest + dev_latest
                lower1_latest = basis_latest - dev_latest
                basis_weekly = row['ma90 weekly']
                dev_weekly = mult * row['dev20 weekly']
                upper_s_weekly = basis_weekly + dev_weekly
                lower1_weekly = basis_weekly - dev_weekly
                weighted_upper_s = upper_s_latest * 0.60 + upper_s_weekly * 0.40
                weighted_lower1 = lower1_latest * 0.60 + lower1_weekly * 0.40
                if row['price'] < weighted_lower1 and row['rsi latest'] < lower_thres:
                    return 3.0
                elif row['price'] > weighted_upper_s and row['rsi latest'] > higher_thres:
                    return 0.5
                else:
                    return 1.0

            def calculate_momentum_convergence(self, df):
                ema_9_21_latest = df['9ema osc latest'] - df['21ema osc latest']
                ema_9_21_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
                momentum_strength = np.abs(ema_9_21_latest) + np.abs(ema_9_21_weekly)
                convergence_quality = np.where(
                    (np.sign(ema_9_21_latest) == np.sign(ema_9_21_weekly)) & (momentum_strength > 10),
                    np.minimum(momentum_strength / 50, 2.0), 0
                )
                trend_exhaustion = np.where(
                    ((df['osc latest'] < -90) & (ema_9_21_latest > 0)) |
                    ((df['osc weekly'] < -80) & (ema_9_21_weekly > 0)),
                    2.0 + np.abs(df['osc latest']) / 100, 0
                )
                momentum_persistence = np.where(
                    (convergence_quality > 0) & (trend_exhaustion > 0),
                    np.minimum(convergence_quality + trend_exhaustion, 3.0),
                    convergence_quality + trend_exhaustion
                )
                df['momentum_score'] = momentum_persistence
                return df

            def calculate_risk_adjusted_attractiveness(self, df):
                vol_latest = df['dev20 latest'] / (df['price'] + 1e-6)
                vol_weekly = df['dev20 weekly'] / (df['price'] + 1e-6)
                avg_volatility = (vol_latest + vol_weekly) / 2
                risk_adj_factor = np.where(
                    avg_volatility < 0.01, 0.8,
                    np.where(avg_volatility < 0.03, 1.2,
                            np.where(avg_volatility < 0.05, 1.0,
                                    np.maximum(0.5, 1 - (avg_volatility - 0.05) * 10)))
                )
                ma_spread_90 = (df['ma90 latest'] - df['price']) / (df['ma90 latest'] + 1e-6)
                ma_spread_200 = (df['ma200 latest'] - df['price']) / (df['ma200 latest'] + 1e-6)
                mean_reversion_prob = np.where(
                    (ma_spread_90 > 0.1) & (ma_spread_200 > 0.15) & (df['rsi latest'] < 40),
                    np.minimum((ma_spread_90 + ma_spread_200) * 5, 2.5), 0
                )
                bb_basis = (df['ma90 latest'] + df['ma200 latest']) / 2
                bb_lower = bb_basis - 2 * df['dev20 latest']
                bb_position = (df['price'] - bb_lower) / (bb_basis - bb_lower + 1e-6)
                bb_score = np.where(
                    bb_position < 0.2, 2.0,
                    np.where(bb_position < 0.4, 1.5,
                            np.where(bb_position < 0.6, 1.0, 0.5))
                )
                df['risk_adjusted_score'] = (mean_reversion_prob + bb_score) * risk_adj_factor
                return df

            def calculate_adaptive_factor_weights(self, regime, confidence):
                base_weights = {'anomaly': 0.25, 'momentum': 0.20, 'risk_adjusted': 0.25, 'quality': 0.15, 'consistency': 0.15}
                if regime == "CRISIS":
                    self.factor_weights = {'anomaly': 0.35, 'momentum': 0.15, 'risk_adjusted': 0.30, 'quality': 0.15, 'consistency': 0.05}
                elif regime == "BEAR_TREND":
                    self.factor_weights = {'anomaly': 0.25, 'momentum': 0.30, 'risk_adjusted': 0.25, 'quality': 0.10, 'consistency': 0.10}
                elif regime in ["BULL_TREND", "BULL_EUPHORIA"]:
                    self.factor_weights = {'anomaly': 0.15, 'momentum': 0.20, 'risk_adjusted': 0.20, 'quality': 0.25, 'consistency': 0.20}
                else:
                    self.factor_weights = base_weights
                confidence_factor = 0.7 + 0.3 * confidence
                for k in self.factor_weights:
                    self.factor_weights[k] *= confidence_factor

            def calculate_consistency_score(self, df):
                latest_weekly_corr = []
                metrics_pairs = [('rsi latest', 'rsi weekly'), ('osc latest', 'osc weekly'),
                                ('9ema osc latest', '9ema osc weekly'), ('21ema osc latest', '21ema osc weekly')]
                for latest_col, weekly_col in metrics_pairs:
                    if len(df) > 1:
                        corr = np.corrcoef(df[latest_col], df[weekly_col])[0, 1]
                        latest_weekly_corr.append(max(0, corr))
                avg_correlation = np.mean(latest_weekly_corr) if latest_weekly_corr else 0
                oversold_signals = (
                    (df['rsi latest'] < 35).astype(int) +
                    (df['osc latest'] < -60).astype(int) +
                    (df['zscore latest'] < -1.5).astype(int) +
                    (df['rsi weekly'] < 40).astype(int) +
                    (df['osc weekly'] < -50).astype(int)
                )
                signal_coherence = oversold_signals / 5
                temporal_stability = 1 - np.minimum(np.abs(df['rsi latest'] - df['rsi weekly']) / 100, 1)
                df['consistency_score'] = avg_correlation * 0.4 + signal_coherence * 0.4 + temporal_stability * 0.2
                return df

            def calculate_original_base_score(self, df):
                original_weights = {'rsi': 0.15, 'osc': 0.20, 'ema_osc': 0.15, '21ema_osc': 0.10, 'zscore': 0.15, 'spread': 0.15, 'bollinger': 0.10}
                df['original_base_mult'] = (
                    df['rsi_mult'] * original_weights['rsi'] +
                    df['osc_mult'] * original_weights['osc'] +
                    df['ema9_osc_mult'] * original_weights['ema_osc'] +
                    df['ema21_osc_mult'] * original_weights['21ema_osc'] +
                    df['zscore_mult'] * original_weights['zscore'] +
                    df['spread_mult'] * original_weights['spread'] +
                    df['bollinger_mult'] * original_weights['bollinger']
                )
                df['original_final_mult'] = df['original_base_mult'] * df['trend_strength']
                df['weekly_oversold_boost'] = df['osc weekly'].apply(lambda x: 1.2 if x < -20 else 0.8)
                df['original_final_mult'] = df['original_final_mult'] * df['weekly_oversold_boost']
                return df

            def generate_hybrid_composite_score(self, df):
                if 'momentum_score' not in df.columns: df['momentum_score'] = 1.0
                if 'risk_adjusted_score' not in df.columns: df['risk_adjusted_score'] = 1.0
                if 'consistency_score' not in df.columns: df['consistency_score'] = 1.0
                max_original = df['original_final_mult'].max()
                df['normalized_original'] = (df['original_final_mult'] / max_original) * 5.0 if max_original > 0 else 0
                df['composite_score'] = (
                    df['normalized_original'] * 0.70 +
                    (df['momentum_score'] + df['risk_adjusted_score'] + df['consistency_score']) / 3 * 1.5 * 0.30
                )
                return df

            def intelligent_portfolio_construction(self, df, concentration_limit=0.10):
                selected_etfs = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
                selected_etfs = self._calculate_dynamic_weights(selected_etfs, concentration_limit)
                return selected_etfs

            def _calculate_dynamic_weights(self, df, max_weight=0.10):
                n_etfs = len(df)
                equal_weight = 1.0 / n_etfs
                min_weight = 0.01 
                max_weight_limit = 0.10 
                scores = df['composite_score'].values
                if n_etfs == 1:
                    df['tier_multiplier'] = [1.0]
                else:
                    percentile_80 = np.percentile(scores, 80)
                    percentile_60 = np.percentile(scores, 60)
                    percentile_40 = np.percentile(scores, 40)
                    percentile_20 = np.percentile(scores, 20)
                    tier_multipliers = []
                    for score in scores:
                        if score >= percentile_80:
                            multiplier = 2.5 + (score - percentile_80) / max(scores.max() - percentile_80, 1e-6) * 0.5
                        elif score >= percentile_60:
                            multiplier = 1.8 + (score - percentile_60) / max(percentile_80 - percentile_60, 1e-6) * 0.7
                        elif score >= percentile_40:
                            multiplier = 1.2 + (score - percentile_40) / max(percentile_60 - percentile_40, 1e-6) * 0.6
                        elif score >= percentile_20:
                            multiplier = 0.8 + (score - percentile_20) / max(percentile_40 - percentile_20, 1e-6) * 0.4
                        else:
                            multiplier = 0.5 + (score - scores.min()) / max(percentile_20 - scores.min(), 1e-6) * 0.3
                        tier_multipliers.append(multiplier)
                    df['tier_multiplier'] = tier_multipliers
                raw_weights = np.array(df['tier_multiplier']) * equal_weight
                df['raw_weight'] = raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else raw_weights
                df = self._apply_concentration_limits(df, max_weight_limit, min_weight)
                df['optimized_weight'] = df['final_weight']
                df['weightage_pct'] = df['optimized_weight'] * 100
                return df

            def _apply_concentration_limits(self, df, max_weight, min_weight):
                max_iterations = 20
                iteration = 0
                df['final_weight'] = df['raw_weight'].copy()
                while iteration < max_iterations:
                    over_weight_mask = df['final_weight'] > max_weight
                    under_weight_mask = df['final_weight'] < min_weight
                    if not (over_weight_mask.any() or under_weight_mask.any()):
                        break
                    if under_weight_mask.any():
                        shortfall = (min_weight - df.loc[under_weight_mask, 'final_weight']).sum()
                        df.loc[under_weight_mask, 'final_weight'] = min_weight
                        eligible_for_reduction = df['final_weight'] > min_weight
                        if eligible_for_reduction.sum() > 0:
                            reduction_capacity = (df.loc[eligible_for_reduction, 'final_weight'] - min_weight).sum()
                            if reduction_capacity > shortfall:
                                excess_weights = df.loc[eligible_for_reduction, 'final_weight'] - min_weight
                                reduction_factors = excess_weights / excess_weights.sum() * shortfall
                                df.loc[eligible_for_reduction, 'final_weight'] -= reduction_factors
                    if over_weight_mask.any():
                        excess = (df.loc[over_weight_mask, 'final_weight'] - max_weight).sum()
                        df.loc[over_weight_mask, 'final_weight'] = max_weight
                        eligible_for_increase = (df['final_weight'] < max_weight) & (df['final_weight'] >= min_weight)
                        if eligible_for_increase.sum() > 0:
                            available_capacity = max_weight - df.loc[eligible_for_increase, 'final_weight']
                            score_weights = df.loc[eligible_for_increase, 'composite_score'] / df.loc[eligible_for_increase, 'composite_score'].sum()
                            capacity_weights = available_capacity / available_capacity.sum()
                            allocation_factors = (score_weights * 0.5 + capacity_weights * 0.5)
                            allocation_factors = allocation_factors / allocation_factors.sum()
                            additional_weights = allocation_factors * excess
                            new_weights = df.loc[eligible_for_increase, 'final_weight'] + additional_weights
                            new_weights = np.minimum(new_weights, max_weight)
                            df.loc[eligible_for_increase, 'final_weight'] = new_weights
                    iteration += 1
                total = df['final_weight'].sum()
                if total > 0:
                    df['final_weight'] = df['final_weight'] / total
                return df

        try:
            analyzer = QuantitativeETFAnalyzer()
            df_prepared = analyzer.validate_and_prepare_data(df.copy())
            if df_prepared.empty:
                return pd.DataFrame()

            regime, confidence = analyzer.detect_market_regime(df_prepared)
            analyzer.calculate_adaptive_factor_weights(regime, confidence)
            df_prepared = analyzer.calculate_enhanced_technical_multipliers(df_prepared)
            df_prepared = analyzer.calculate_original_base_score(df_prepared)
            df_prepared = analyzer.calculate_momentum_convergence(df_prepared)
            df_prepared = analyzer.calculate_risk_adjusted_attractiveness(df_prepared)
            df_prepared = analyzer.calculate_consistency_score(df_prepared)
            df_prepared = analyzer.generate_hybrid_composite_score(df_prepared)
            portfolio_df = analyzer.intelligent_portfolio_construction(df_prepared, concentration_limit=0.10)
            
            total_score = portfolio_df['composite_score'].sum()
            if total_score > 0:
                 portfolio_df['weightage'] = portfolio_df['composite_score'] / total_score
            else:
                 portfolio_df['weightage'] = 1 / len(portfolio_df) if len(portfolio_df) > 0 else 0

            return self._allocate_portfolio(portfolio_df, sip_amount)

        except Exception as e:
            logging.error(f"Error in CL2Strategy portfolio generation: {str(e)}")
            raise

# =====================================
# CL_v3 Strategy Implementation
# =====================================

class CL3Strategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        class UltimateETFAnalyzer:
            def __init__(self):
                self.market_regime = None
                self.regime_confidence = 0.5
                self.conviction_threshold = 0.65
                self.position_limits = {'min': 0.01, 'max': 0.10}

            def prepare_and_validate_data(self, df):
                required_columns = [
                    'symbol', 'price', 'date',
                    'rsi latest', 'rsi weekly',
                    'osc latest', 'osc weekly',
                    '9ema osc latest', '9ema osc weekly',
                    '21ema osc latest', '21ema osc weekly',
                    'zscore latest', 'zscore weekly',
                    'ma90 latest', 'ma200 latest',
                    'ma90 weekly', 'ma200 weekly',
                    'dev20 latest', 'dev20 weekly'
                ]
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {missing}")
                df = self._handle_missing_values(df)
                df['data_quality'] = self._calculate_data_quality(df)
                df = self._handle_outliers(df)
                return df

            def _handle_missing_values(self, df):
                for col in ['rsi latest', 'rsi weekly']:
                    df[col] = df[col].fillna(50)
                osc_cols = [c for c in df.columns if 'osc' in c.lower()]
                for col in osc_cols:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
                zscore_cols = [c for c in df.columns if 'zscore' in c.lower()]
                for col in zscore_cols:
                    df[col] = df[col].fillna(0)
                ma_cols = [c for c in df.columns if c.startswith('ma')]
                for col in ma_cols:
                    df[col] = df[col].fillna(df['price'])
                dev_cols = [c for c in df.columns if 'dev' in c.lower()]
                for col in dev_cols:
                    df[col] = df[col].fillna(df['price'] * 0.01)
                return df

            def _calculate_data_quality(self, df):
                quality_scores = []
                for idx, row in df.iterrows():
                    score = 1.0
                    if row['rsi latest'] == 50 and row['rsi weekly'] == 50:
                        score -= 0.2
                    if row['osc latest'] == 0 or row['osc weekly'] == 0:
                        score -= 0.1
                    if row['price'] <= 0 or row['price'] > 10000:
                        score -= 0.3
                    quality_scores.append(max(0.3, score))
                return quality_scores

            def _handle_outliers(self, df):
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if col not in ['price', 'symbol', 'data_quality']:
                        mean = df[col].mean()
                        std = df[col].std()
                        if std > 0:
                            df[col] = df[col].clip(mean - 3*std, mean + 3*std)
                return df

            def detect_market_regime(self, df):
                metrics = {
                    'avg_rsi': df['rsi latest'].mean(),
                    'avg_osc': df['osc latest'].mean(),
                    'volatility': df['dev20 latest'].mean() / df['price'].mean(),
                    'breadth': len(df[df['rsi latest'] < 50]) / len(df),
                    'momentum_consistency': np.corrcoef(df['osc latest'], df['osc weekly'])[0, 1]
                }
                if metrics['avg_rsi'] < 35 and metrics['avg_osc'] < -40:
                    self.market_regime, self.regime_confidence = "OVERSOLD_EXTREME", 0.9
                elif metrics['avg_rsi'] < 45 and metrics['breadth'] > 0.6:
                    self.market_regime, self.regime_confidence = "BEARISH", 0.8
                elif metrics['avg_rsi'] > 65 and metrics['avg_osc'] > 30:
                    self.market_regime, self.regime_confidence = "OVERBOUGHT", 0.85
                elif metrics['avg_rsi'] > 55 and metrics['breadth'] < 0.4:
                    self.market_regime, self.regime_confidence = "BULLISH", 0.75
                else:
                    self.market_regime, self.regime_confidence = "NEUTRAL", 0.6
                return self.market_regime, self.regime_confidence, metrics

            def calculate_technical_suite(self, df):
                df['rsi_mult'] = df.apply(self._calc_rsi_mult, axis=1)
                df['osc_mult'] = df.apply(self._calc_osc_mult, axis=1)
                df['ema9_mult'] = df.apply(self._calc_ema9_mult, axis=1)
                df['ema21_mult'] = df.apply(self._calc_ema21_mult, axis=1)
                df['zscore_mult'] = df.apply(self._calc_zscore_mult, axis=1)
                df['trend_strength'] = df.apply(self._calc_trend_strength, axis=1)
                df['spread_mult'] = df.apply(self._calc_spread_mult, axis=1)
                df['bollinger_mult'] = df.apply(self._calc_bollinger_mult, axis=1)
                df['weekly_boost'] = df['osc weekly'].apply(lambda x: 1.2 if x < -20 else 0.8)
                return df

            def _calc_rsi_mult(self, row):
                weighted_rsi = row['rsi weekly'] * 0.55 + row['rsi latest'] * 0.45
                if weighted_rsi < 30:
                    return 3.5 - (weighted_rsi / 30) * 1.5
                elif weighted_rsi < 50:
                    return 2 - (weighted_rsi - 30) / 20
                elif weighted_rsi < 70:
                    return 1 - (weighted_rsi - 50) / 20
                else:
                    return 0.3 + (100 - weighted_rsi) / 30

            def _calc_osc_mult(self, row):
                w, d = row['osc weekly'], row['osc latest']
                if w < -80 and d < -95: return 3.5
                elif w < -80: return 3.2
                elif w < -70 and d < -90: return 2.8
                elif w < -70: return 2.5
                elif w < -60 and d < -85: return 2.3
                elif w < -50 and d < -80: return 2.0
                elif w < -40 and d < -70: return 1.8
                elif w < -30 and d < -60: return 1.6
                elif w < -20 and d < -50: return 1.5
                elif w < -10 and d < -40: return 1.4
                elif w < 0 and d < -30: return 1.3
                elif d < -95: return 2.0
                elif d > 80 and w > 70: return 0.2
                else: return 0.1

            def _calc_ema9_mult(self, row):
                w, d = row['9ema osc weekly'], row['9ema osc latest']
                if w < -80 and d < -90: return 3.5
                elif w < -80: return 3.2
                elif w < -70 and d < -80: return 2.8
                elif w < -70: return 2.5
                elif w < -60 and d < -70: return 2.3
                elif w < -50 and d < -60: return 2.0
                elif w < -40 and d < -50: return 1.8
                elif w < -30 and d < -40: return 1.6
                elif d < -90: return 2.0
                else: return 0.1

            def _calc_ema21_mult(self, row):
                w, d = row['21ema osc weekly'], row['21ema osc latest']
                if w < -80 and d < -90: return 3.5
                elif w < -80: return 3.2
                elif w < -70 and d < -80: return 2.8
                elif w < -70: return 2.5
                elif w < -60 and d < -70: return 2.3
                elif w < -50 and d < -60: return 2.0
                elif w < -40 and d < -50: return 1.8
                elif w < -30 and d < -40: return 1.6
                elif d < -90: return 2.0
                else: return 0.1

            def _calc_zscore_mult(self, row):
                w, d = row['zscore weekly'], row['zscore latest']
                if w < -2.5 and d < -3: return 3.5
                elif w < -2.5: return 3.2
                elif w < -2 and d < -2.5: return 2.8
                elif w < -1.5 and d < -2: return 2.5
                elif w < -1.2 and d < -1.8: return 2.2
                elif w < -1 and d < -1.5: return 2.0
                elif d < -3: return 2.0
                else: return 0.1

            def _calc_trend_strength(self, row):
                if row['9ema osc latest'] > row['21ema osc latest'] and row['osc latest'] < -50:
                    return 1.3
                elif row['9ema osc weekly'] > row['21ema osc weekly'] and row['osc weekly'] < -50:
                    return 1.5
                elif row['9ema osc latest'] > 0 and row['21ema osc latest'] > 0 and row['osc latest'] > 0:
                    return 0.7
                else:
                    return 1.0

            def _calc_spread_mult(self, row):
                def safe_pct(val, base):
                    return val * 100 / (base if base != 0 else 1e-6)
                s90d = safe_pct(row['ma90 latest'] - row['price'], row['ma90 latest'])
                s200d = safe_pct(row['ma200 latest'] - row['price'], row['ma200 latest'])
                s90w = safe_pct(row['ma90 weekly'] - row['price'], row['ma90 weekly'])
                s200w = safe_pct(row['ma200 weekly'] - row['price'], row['ma200 weekly'])
                ws90 = s90d * 0.6 + s90w * 0.4
                ws200 = s200d * 0.6 + s200w * 0.4
                if ws90 > 1.5 and ws200 > 1.5 and row['rsi latest'] < 40:
                    return 3.5
                elif ws90 < -1.5 and ws200 < -1.5 and row['rsi latest'] > 70:
                    return 0.5
                else:
                    return 1.0

            def _calc_bollinger_mult(self, row):
                bd = row['ma90 latest']
                dd = 2 * row['dev20 latest']
                bw = row['ma90 weekly']
                dw = 2 * row['dev20 weekly']
                lower_d = bd - dd
                lower_w = bw - dw
                upper_d = bd + dd
                upper_w = bw + dw
                wl = lower_d * 0.6 + lower_w * 0.4
                wu = upper_d * 0.6 + upper_w * 0.4
                if row['price'] < wl and row['rsi latest'] < 40:
                    return 3.0
                elif row['price'] > wu and row['rsi latest'] > 70:
                    return 0.5
                else:
                    return 1.0

            def calculate_conviction_scores(self, df):
                df['signal_alignment'] = df.apply(self._calc_signal_alignment, axis=1)
                df['divergence_score'] = df.apply(self._detect_divergence, axis=1)
                df['mean_reversion'] = df.apply(self._calc_mean_reversion, axis=1)
                df['vol_quality'] = df.apply(self._calc_vol_quality, axis=1)
                df['conviction'] = (
                    df['signal_alignment'] * 0.35 +
                    df['divergence_score'] * 0.25 +
                    df['mean_reversion'] * 0.25 +
                    df['vol_quality'] * 0.15
                )
                return df

            def _calc_signal_alignment(self, row):
                signals = 0
                if row['rsi latest'] < 30 and row['rsi weekly'] < 35:
                    signals += 2
                elif row['rsi latest'] < 35 or row['rsi weekly'] < 40:
                    signals += 1
                if row['osc latest'] < -80 and row['osc weekly'] < -60:
                    signals += 2
                elif row['osc latest'] < -60 or row['osc weekly'] < -40:
                    signals += 1
                if row['zscore latest'] < -2 and row['zscore weekly'] < -1.5:
                    signals += 2
                elif row['zscore latest'] < -1.5 or row['zscore weekly'] < -1:
                    signals += 1
                return min(2.0, signals / 3)

            def _detect_divergence(self, row):
                price_low = row['price'] < row['ma90 latest'] * 0.95
                osc_improving = row['9ema osc latest'] > row['21ema osc latest']
                if price_low and osc_improving and row['osc latest'] < -50:
                    return 2.0
                elif price_low and osc_improving:
                    return 1.5
                elif osc_improving and row['osc latest'] < -30:
                    return 1.2
                else:
                    return 1.0

            def _calc_mean_reversion(self, row):
                if row['ma200 latest'] > 0:
                    dist = (row['ma200 latest'] - row['price']) / row['ma200 latest']
                else:
                    dist = 0
                if dist > 0.2 and row['zscore latest'] < -2:
                    return 2.5
                elif dist > 0.15 and row['zscore latest'] < -1.5:
                    return 2.0
                elif dist > 0.1:
                    return 1.5
                else:
                    return 1.0

            def _calc_vol_quality(self, row):
                if row['price'] > 0:
                    avg_vol = (row['dev20 latest'] + row['dev20 weekly']) / (2 * row['price'])
                else:
                    avg_vol = 0.05
                if 0.01 < avg_vol < 0.03:
                    return 1.5
                elif avg_vol < 0.01:
                    return 0.8
                elif avg_vol < 0.05:
                    return 1.0
                else:
                    return 0.7

            def calculate_composite_scores(self, df):
                weights = {'rsi': 0.15, 'osc': 0.20, 'ema9': 0.15, 'ema21': 0.10, 'zscore': 0.15, 'spread': 0.15, 'bollinger': 0.10}
                df['base_score'] = (
                    df['rsi_mult'] * weights['rsi'] +
                    df['osc_mult'] * weights['osc'] +
                    df['ema9_mult'] * weights['ema9'] +
                    df['ema21_mult'] * weights['ema21'] +
                    df['zscore_mult'] * weights['zscore'] +
                    df['spread_mult'] * weights['spread'] +
                    df['bollinger_mult'] * weights['bollinger']
                )
                df['base_score'] = df['base_score'] * df['trend_strength'] * df['weekly_boost']
                max_base = df['base_score'].max()
                df['base_score_norm'] = (df['base_score'] / max_base * 5) if max_base > 0 else df['base_score']
                df['composite_score'] = (
                    df['base_score_norm'] * 0.60 +
                    df['conviction'] * 1.5 * 0.25 +
                    df['data_quality'] * 2 * 0.15
                )
                if self.market_regime == "OVERSOLD_EXTREME":
                    df['composite_score'] *= 1.2
                elif self.market_regime == "BEARISH":
                    df['composite_score'] *= 1.1
                elif self.market_regime == "OVERBOUGHT":
                    df['composite_score'] *= 0.8
                return df

            def construct_portfolio(self, df, capital):
                portfolio = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
                total_score = portfolio['composite_score'].sum()
                if total_score > 0:
                    portfolio['weightage'] = portfolio['composite_score'] / total_score
                else:
                    portfolio['weightage'] = 1 / len(portfolio) if len(portfolio) > 0 else 0
                return portfolio

        try:
            analyzer = UltimateETFAnalyzer()
            df_prepared = analyzer.prepare_and_validate_data(df.copy())
            if df_prepared.empty:
                return pd.DataFrame()
            regime, confidence, regime_metrics = analyzer.detect_market_regime(df_prepared)
            df_prepared = analyzer.calculate_technical_suite(df_prepared)
            df_prepared = analyzer.calculate_conviction_scores(df_prepared)
            df_prepared = analyzer.calculate_composite_scores(df_prepared)
            portfolio_df = analyzer.construct_portfolio(df_prepared, sip_amount)
            return self._allocate_portfolio(portfolio_df, sip_amount)

        except Exception as e:
            logging.error(f"Error in CL3Strategy portfolio generation: {str(e)}")
            raise

# =====================================
# MOM_v1 Strategy: Multi-Factor Momentum Regime
# =====================================

class MOM1Strategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        # Data validation
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        df = self._detect_market_regime(df)
        df = self._calculate_momentum_scores(df)
        df = self._calculate_acceleration_factor(df)
        df = self._calculate_volatility_adjustment(df)
        df = self._calculate_mean_reversion_overlay(df)
        df = self._calculate_composite_momentum(df)
        
        total_score = df['final_momentum_score'].sum()
        if total_score > 0:
            df['weightage'] = df['final_momentum_score'] / total_score
        else:
            df['weightage'] = 1 / len(df) if len(df) > 0 else 0
        
        return self._allocate_portfolio(df, sip_amount)
    
    def _detect_market_regime(self, df):
        avg_rsi = df['rsi latest'].mean()
        avg_osc = df['osc latest'].mean()
        avg_vol = (df['dev20 latest'] / df['price']).mean()
        if avg_rsi < 40 and avg_osc < -30:
            df['regime'] = 'OVERSOLD'
            df['regime_factor'] = 0.7 
        elif avg_rsi > 60 and avg_osc > 20:
            df['regime'] = 'MOMENTUM'
            df['regime_factor'] = 1.3 
        elif avg_vol > 0.03:
            df['regime'] = 'VOLATILE'
            df['regime_factor'] = 0.8
        else:
            df['regime'] = 'NEUTRAL'
            df['regime_factor'] = 1.0
        return df
    
    def _calculate_momentum_scores(self, df):
        df['price_mom_90'] = (df['price'] / df['ma90 latest'] - 1) * 100
        df['price_mom_200'] = (df['price'] / df['ma200 latest'] - 1) * 100
        df['price_mom_score'] = (df['price_mom_90'] * 0.6 + df['price_mom_200'] * 0.4)
        df['osc_momentum'] = (df['osc latest'] * 0.4 + df['osc weekly'] * 0.6)
        df['ema_slope'] = ((df['9ema osc latest'] - df['21ema osc latest']) * 0.5 +
            (df['9ema osc weekly'] - df['21ema osc weekly']) * 0.5)
        df['rsi_momentum'] = ((df['rsi latest'] - 50) * 0.4 + (df['rsi weekly'] - 50) * 0.6)
        df['zscore_momentum'] = -(df['zscore latest'] * 0.45 + df['zscore weekly'] * 0.55)
        return df
    
    def _calculate_acceleration_factor(self, df):
        df['timeframe_alignment'] = np.where(
            (np.sign(df['osc latest']) == np.sign(df['osc weekly'])) &
            (np.sign(df['9ema osc latest']) == np.sign(df['9ema osc weekly'])),
            1.5, 
            np.where(
                np.sign(df['osc latest']) == np.sign(df['osc weekly']),
                1.2, 
                1.0 
            )
        )
        df['ema_crossover_strength'] = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) &
            (df['9ema osc weekly'] > df['21ema osc weekly']),
            1.4,
            np.where(
                (df['9ema osc latest'] > df['21ema osc latest']),
                1.2,
                0.8
            )
        )
        df['acceleration'] = df['timeframe_alignment'] * df['ema_crossover_strength']
        return df
    
    def _calculate_volatility_adjustment(self, df):
        df['volatility'] = (df['dev20 latest'] / df['price'] * 0.6 + df['dev20 weekly'] / df['price'] * 0.4)
        epsilon = 1e-6
        df['vol_adj_factor'] = np.where(
            df['volatility'] < 0.015,
            1.3,
            np.where(
                df['volatility'] < 0.025,
                1.0, 
                np.maximum(0.7, 1 - (df['volatility'] - 0.025) * 10) 
            )
        )
        return df
    
    def _calculate_mean_reversion_overlay(self, df):
        df['mean_reversion_boost'] = np.where(
            (df['zscore latest'] < -2.0) &
            (df['zscore weekly'] < -1.8) &
            (df['rsi latest'] < 35) &
            (df['osc latest'] < -60) &
            (df['9ema osc latest'] > df['21ema osc latest']), 
            1.5, 
            np.where(
                (df['zscore latest'] < -1.5) &
                (df['rsi latest'] < 40) &
                (df['9ema osc latest'] > df['21ema osc latest']),
                1.2, 
                1.0
            )
        )
        return df
    
    def _calculate_composite_momentum(self, df):
        regime_weights = df['regime_factor'].iloc[0]
        def normalize_score(series):
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - min_val) / (max_val - min_val)
        df['norm_price_mom'] = normalize_score(df['price_mom_score'])
        df['norm_osc_mom'] = normalize_score(df['osc_momentum'])
        df['norm_ema_slope'] = normalize_score(df['ema_slope'])
        df['norm_rsi_mom'] = normalize_score(df['rsi_momentum'])
        df['norm_zscore_mom'] = normalize_score(df['zscore_momentum'])
        df['raw_momentum_score'] = (
            df['norm_price_mom'] * 0.25 +
            df['norm_osc_mom'] * 0.25 +
            df['norm_ema_slope'] * 0.20 +
            df['norm_rsi_mom'] * 0.15 +
            df['norm_zscore_mom'] * 0.15
        )
        df['final_momentum_score'] = (
            df['raw_momentum_score'] *
            df['acceleration'] *
            df['vol_adj_factor'] *
            df['mean_reversion_boost'] *
            regime_weights
        )
        return df

# =====================================
# MOM_v2 Strategy: Statistical Arbitrage Momentum
# =====================================

class MOM2Strategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        df = self._calculate_statistical_factors(df)
        df = self._calculate_momentum_persistence(df)
        df = self._calculate_relative_momentum(df)
        df = self._detect_momentum_clusters(df)
        df = self._calculate_edge_score(df)
        df = self._construct_optimal_portfolio(df)
        
        total_score = df['final_composite_score'].sum()
        if total_score > 0:
            df['weightage'] = df['final_composite_score'] / total_score
        else:
            df['weightage'] = 1 / len(df) if len(df) > 0 else 0
            
        return self._allocate_portfolio(df, sip_amount)
    
    def _calculate_statistical_factors(self, df):
        df['rsi_zscore'] = (df['rsi latest'] - df['rsi latest'].mean()) / (df['rsi latest'].std() + 1e-6)
        df['osc_zscore'] = (df['osc latest'] - df['osc latest'].mean()) / (df['osc latest'].std() + 1e-6)
        df['price_ma_ratio'] = df['price'] / df['ma90 latest']
        df['price_mom_zscore'] = (df['price_ma_ratio'] - df['price_ma_ratio'].mean()) / (df['price_ma_ratio'].std() + 1e-6)
        df['vol_adj_ret'] = (df['price'] / df['ma20 latest'] - 1) / (df['dev20 latest'] / df['price'] + 1e-6)
        df['vol_adj_ret_zscore'] = (df['vol_adj_ret'] - df['vol_adj_ret'].mean()) / (df['vol_adj_ret'].std() + 1e-6)
        df['ts_mom_score'] = -(df['zscore latest'] * 0.6 + df['zscore weekly'] * 0.4) 
        return df
    
    def _calculate_momentum_persistence(self, df):
        df['daily_weekly_consistency'] = np.where(
            (np.sign(df['osc latest']) == np.sign(df['osc weekly'])) &
            (np.abs(df['osc latest'] - df['osc weekly']) < 30),
            2.0, 
            np.where(
                np.sign(df['osc latest']) == np.sign(df['osc weekly']),
                1.5,
                0.8 
            )
        )
        df['ema_trend_strength'] = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) &
            (df['9ema osc weekly'] > df['21ema osc weekly']) &
            (df['osc latest'] < -40), 
            2.5, 
            np.where(
                (df['9ema osc latest'] > df['21ema osc latest']) &
                (df['osc latest'] < -20),
                1.8, 
                np.where(
                    (df['9ema osc latest'] < df['21ema osc latest']) &
                    (df['osc latest'] > 20),
                    0.5, 
                    1.0
                )
            )
        )
        df['rsi_persistence'] = np.where(
            (df['rsi latest'] < 35) & (df['rsi weekly'] < 40),
            1.5, 
            np.where(
                (df['rsi latest'] < 40) | (df['rsi weekly'] < 45),
                1.2, 
                np.where(
                    (df['rsi latest'] > 60) & (df['rsi weekly'] > 55),
                    0.7, 
                    1.0
                )
            )
        )
        return df
    
    def _calculate_relative_momentum(self, df):
        df['rsi_percentile'] = df['rsi latest'].rank(pct=True)
        df['osc_percentile'] = df['osc latest'].rank(pct=True)
        df['price_mom_percentile'] = df['price_ma_ratio'].rank(pct=True)
        df['relative_momentum'] = (
            (1 - df['rsi_percentile']) * 0.30 +
            (1 - df['osc_percentile']) * 0.35 + 
            df['price_mom_percentile'] * 0.35 
        )
        df['relative_momentum'] = df['relative_momentum'] * 2
        return df
    
    def _detect_momentum_clusters(self, df):
        df['momentum_composite'] = (
            df['osc latest'] * 0.3 +
            df['osc weekly'] * 0.3 +
            df['rsi latest'] * 0.2 +
            df['9ema osc latest'] * 0.2
        )
        try:
            bin_labels = pd.qcut(df['momentum_composite'], q=5, labels=False, duplicates='drop')
        except ValueError:
            bin_labels = pd.Series(2, index=df.index) 

        weight_map = {0: 2.0, 1: 1.3, 2: 1.0, 3: 0.8, 4: 0.6}
        df['cluster_weight'] = bin_labels.map(weight_map)
        df['cluster_weight'] = df['cluster_weight'].fillna(1.0)
        return df
    
    def _calculate_edge_score(self, df):
        df['momentum_reversion_edge'] = np.where(
            (df['zscore latest'] < -1.8) &
            (df['9ema osc latest'] > df['21ema osc latest']) &
            (df['osc latest'] < -50),
            2.5,
            np.where(
                (df['zscore latest'] < -1.2) &
                (df['osc latest'] < -40),
                1.8, 
                1.0
            )
        )
        df['vol_regime_edge'] = np.where(
            (df['dev20 latest'] / df['price'] < 0.02) & 
            (df['osc latest'] < -30), 
            1.5, 
            np.where(
                (df['dev20 latest'] / df['price'] > 0.04) & 
                (df['osc latest'] < -60), 
                1.3, 
                1.0
            )
        )
        bb_lower = df['ma20 latest'] - 2 * df['dev20 latest']
        bb_upper = df['ma20 latest'] + 2 * df['dev20 latest']
        df['bb_position'] = (df['price'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
        df['bb_edge'] = np.where(
            (df['bb_position'] < 0.2) & (df['rsi latest'] < 40),
            1.8,
            np.where(
                (df['bb_position'] < 0.4) & (df['rsi latest'] < 45),
                1.3,
                np.where(
                    df['bb_position'] > 0.8,
                    0.7, 
                    1.0
                )
            )
        )
        df['total_edge_score'] = (
            df['momentum_reversion_edge'] * 0.35 +
            df['vol_regime_edge'] * 0.30 +
            df['bb_edge'] * 0.35
        )
        return df
    
    def _construct_optimal_portfolio(self, df):
        df['final_composite_score'] = (
            df['rsi_zscore'] * -0.10 + 
            df['osc_zscore'] * -0.10 +
            df['price_mom_zscore'] * 0.10 +
            df['vol_adj_ret_zscore'] * 0.10 +
            df['ts_mom_score'] * 0.15 +
            df['daily_weekly_consistency'] * 0.10 +
            df['ema_trend_strength'] * 0.10 +
            df['relative_momentum'] * 0.10 +
            df['cluster_weight'] * 0.05 +
            df['total_edge_score'] * 0.10
        )
        return df

# =====================================
# MomentumMasters Strategy
# =====================================
class MomentumMasters(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'ma90 latest', 'ma200 latest'
        ]
        df = self._clean_data(df, required_columns)
        conditions = [
            (df['price'] > df['ma90 latest']) & (df['ma90 latest'] > df['ma200 latest']), 
            (df['price'] > df['ma90 latest']) & (df['price'] > df['ma200 latest']),   
            (df['price'] > df['ma200 latest']),                                       
        ]
        multipliers = [1.5, 1.0, 0.5]
        df['trend_conviction'] = np.select(conditions, multipliers, default=0.1) 
        rsi_score = np.clip((df['rsi latest'] - 50) / 30, 0, 2.0)
        osc_score = (
            (df['osc latest'] > 20).astype(int) * 0.5 +
            (df['osc weekly'] > 0).astype(int) * 0.5 +
            (df['9ema osc latest'] > df['21ema osc latest']).astype(int) * 1.0
        )
        velocity_score = np.clip(df['zscore latest'], 0, 3.0)
        df['composite_score'] = (
            (rsi_score * 0.4 +
             osc_score * 0.3 +
             velocity_score * 0.3)
            * df['trend_conviction'] 
        )
        df['composite_score'] = df['composite_score'] + 1e-6
        eligible_stocks = df.sort_values('composite_score', ascending=False).copy()
        total_score = eligible_stocks['composite_score'].sum()
        eligible_stocks['weightage'] = eligible_stocks['composite_score'] / total_score if total_score > 0 else (1 / len(eligible_stocks) if len(eligible_stocks) > 0 else 0)
        return self._allocate_portfolio(eligible_stocks, sip_amount)

# =====================================
# VolatilitySurfer Strategy
# =====================================
class VolatilitySurfer(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'zscore latest', 
            'ma90 weekly', 'ma20 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)
        conditions = [
            (df['price'] > df['ma90 weekly']),   
            (df['price'] > df['ma20 latest']),   
        ]
        multipliers = [1.0, 0.5]
        df['trend_conviction'] = np.select(conditions, multipliers, default=0.1)
        upper_band = df['ma20 latest'] + 2 * df['dev20 latest']
        proximity = (df['price'] - upper_band) / (df['ma20 latest'] + 1e-6) 
        breakout_score = np.clip(1 + (proximity * 10), 0, 3.0)
        band_width = (2 * 2 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)
        squeeze_multiplier = np.select(
            [band_width < 0.05, band_width < 0.10],
            [1.5, 1.2],
            default=1.0
        )
        osc_confirmation = np.clip(df['osc latest'] / 50, 0, 2.0)
        zscore_confirmation = np.clip(df['zscore latest'], 0, 3.0)
        confirmation_score = (osc_confirmation * 0.5 + zscore_confirmation * 0.5)
        df['composite_score'] = (
            breakout_score * squeeze_multiplier * confirmation_score * df['trend_conviction']
        )
        df['composite_score'] = df['composite_score'] + 1e-6
        eligible_stocks = df.sort_values('composite_score', ascending=False).copy()
        total_score = eligible_stocks['composite_score'].sum()
        eligible_stocks['weightage'] = eligible_stocks['composite_score'] / total_score if total_score > 0 else (1 / len(eligible_stocks) if len(eligible_stocks) > 0 else 0)
        return self._allocate_portfolio(eligible_stocks, sip_amount)

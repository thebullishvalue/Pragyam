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
# PR_v1 Strategy Implementation
# =====================================

class PRStrategy(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        PR_v1 Strategy: Original Pragati Logic with Full Multiplier Fidelity
        - Fixed weightings (no regime adaptation)
        - 10% max position, 1% min position
        - Intelligent cash allocation based on weekly oversold signals
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
        # Extracted from CL_v1.py: QuantitativeETFAnalyzer class
        class QuantitativeETFAnalyzer:
            """
            Advanced Quantitative ETF Analysis Engine

            Implements sophisticated multi-factor models for ETF selection based on:
            1. Statistical Anomaly Detection
            2. Multi-Timeframe Momentum Convergence
            3. Volatility-Adjusted Risk Assessment
            4. Cross-Asset Correlation Analysis
            5. Regime-Aware Factor Rotation
            """

            def __init__(self):
                self.factor_weights = {}
                self.regime_indicators = {}
                self.quality_threshold = 0.6  # Minimum quality score for selection

            def validate_and_prepare_data(self, df):
                """Enhanced data validation with quality scoring"""
                required_columns = ['symbol', 'price', 'rsi latest', 'rsi weekly',
                                'osc latest', 'osc weekly', '9ema osc latest', '9ema osc weekly',
                                '21ema osc latest', '21ema osc weekly', 'zscore latest', 'zscore weekly',
                                'date', 'ma90 latest', 'ma200 latest', 'ma90 weekly', 'ma200 weekly',
                                'dev20 latest', 'dev20 weekly']

                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

                # Handle NaN values intelligently
                df = self._intelligent_nan_handling(df)

                # Calculate data quality score for each ETF
                df['data_quality_score'] = self._calculate_data_quality(df)

                # Filter out low-quality data points
                df = df[df['data_quality_score'] >= self.quality_threshold]

                return df

            def _intelligent_nan_handling(self, df):
                """Intelligent NaN handling preserving statistical properties"""
                # RSI columns: use neutral 50 for missing values
                rsi_columns = ['rsi latest', 'rsi weekly']
                for col in rsi_columns:
                    df[col] = df[col].fillna(50)

                # Oscillator columns: use sector/market median if available, otherwise 0
                osc_columns = [col for col in df.columns if 'osc' in col.lower()]
                for col in osc_columns:
                    df[col] = df[col].fillna(df[col].median())

                # Z-score: use 0 (neutral) for missing values
                zscore_columns = [col for col in df.columns if 'zscore' in col.lower()]
                for col in zscore_columns:
                    df[col] = df[col].fillna(0)

                # Moving averages: use price if available
                ma_columns = [col for col in df.columns if col.startswith('ma')]
                for col in ma_columns:
                    df[col] = df[col].fillna(df['price'])

                # Other columns: forward fill then backward fill
                for col in df.columns:
                    if col not in rsi_columns + osc_columns + zscore_columns + ma_columns:
                        df[col] = df[col].ffill().bfill().fillna(0)

                return df

            def _calculate_data_quality(self, df):
                """Calculate comprehensive data quality score"""
                quality_factors = []

                # Completeness factor (penalize excessive zeros/NaNs)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                completeness = 1 - (df[numeric_cols] == 0).sum(axis=1) / len(numeric_cols)
                quality_factors.append(completeness)

                # Consistency factor (check for outliers in key metrics)
                key_metrics = ['rsi latest', 'rsi weekly', 'osc latest', 'osc weekly']
                for metric in key_metrics:
                    if metric in df.columns:
                        z_scores = np.abs(stats.zscore(df[metric].fillna(0)))
                        consistency = np.clip(1 - z_scores / 5, 0, 1)  # Penalize extreme outliers
                        quality_factors.append(consistency)

                # Average all quality factors
                return np.mean(quality_factors, axis=0)

            def detect_market_regime(self, df):
                """Advanced regime detection using multiple indicators"""
                # Calculate aggregate market indicators
                avg_rsi = df['rsi latest'].mean()
                avg_osc = df['osc latest'].mean()
                volatility_regime = df['dev20 latest'].mean() / df['price'].mean()

                # Cross-timeframe momentum consistency
                momentum_consistency = np.corrcoef(df['osc latest'], df['osc weekly'])[0, 1]

                # Regime classification with confidence score
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
                """Identify statistical anomalies across multiple dimensions"""
                anomaly_components = []

                # 1. Multi-timeframe RSI divergence
                rsi_divergence = np.abs(df['rsi latest'] - df['rsi weekly']) / (df['rsi latest'] + df['rsi weekly'] + 1e-6)
                rsi_anomaly = np.where((df['rsi latest'] < 30) & (df['rsi weekly'] < 35) & (rsi_divergence < 0.2),
                                    3.0 - (df['rsi latest'] + df['rsi weekly']) / 20, 0)
                anomaly_components.append(rsi_anomaly)

                # 2. Oscillator cascade effect (multiple timeframes aligned)
                osc_cascade = np.where((df['osc latest'] < -70) & (df['osc weekly'] < -60) &
                                    (df['9ema osc latest'] < df['21ema osc latest']),
                                    2.5 + np.abs(df['osc latest'] + df['osc weekly']) / 100, 0)
                anomaly_components.append(osc_cascade)

                # 3. Z-score statistical significance
                zscore_significance = np.where((df['zscore latest'] < -2.0) & (df['zscore weekly'] < -1.5),
                                            np.minimum(np.abs(df['zscore latest']) + np.abs(df['zscore weekly']), 5.0), 0)
                anomaly_components.append(zscore_significance)

                # Combine anomaly scores
                df['anomaly_score'] = np.sum(anomaly_components, axis=0)
                return df

            def calculate_momentum_score(self, df):
                """Calculate momentum convergence score"""
                momentum_score = (
                    (df['rsi latest'] < 40).astype(int) * 1.5 +
                    (df['rsi weekly'] < 45).astype(int) * 1.0 +
                    (df['osc latest'] < -50).astype(int) * 1.2 +
                    (df['osc weekly'] < -40).astype(int) * 0.8
                )
                df['momentum_score'] = momentum_score
                return df

            def calculate_risk_adjusted_score(self, df):
                """Calculate risk-adjusted score using volatility and z-scores"""
                df['risk_score'] = np.where(
                    (df['dev20 latest'] / df['price'] < 0.015) & (np.abs(df['zscore latest']) < 2.0),
                    2.0,
                    np.where((df['dev20 latest'] / df['price'] < 0.03) & (np.abs(df['zscore latest']) < 3.0), 1.0, 0.5)
                )
                return df

            def calculate_composite_score(self, df):
                """Combine all factors into a composite score"""
                # Dynamic factor weights based on regime
                regime, _ = self.detect_market_regime(df)
                if regime == "CRISIS":
                    self.factor_weights = {'anomaly': 0.4, 'momentum': 0.2, 'risk_adjusted': 0.2, 'consistency': 0.1, 'quality': 0.1}
                elif regime == "BEAR_TREND":
                    self.factor_weights = {'anomaly': 0.3, 'momentum': 0.3, 'risk_adjusted': 0.2, 'consistency': 0.1, 'quality': 0.1}
                else:
                    self.factor_weights = {'anomaly': 0.25, 'momentum': 0.25, 'risk_adjusted': 0.2, 'consistency': 0.15, 'quality': 0.15}

                # Calculate consistency score
                df['consistency_score'] = np.where(
                    np.abs(df['rsi latest'] - df['rsi weekly']) < 10,
                    2.0,
                    np.where(np.abs(df['rsi latest'] - df['rsi weekly']) < 20, 1.0, 0.5)
                )

                # Normalize scores
                scaler = StandardScaler()
                score_columns = ['anomaly_score', 'momentum_score', 'risk_score', 'consistency_score', 'data_quality_score']
                normalized_scores = scaler.fit_transform(df[score_columns])

                # Apply weights
                df['composite_score'] = np.sum(normalized_scores * list(self.factor_weights.values()), axis=1)

                return df

            def allocate_portfolio(self, df, sip_amount):
                """Allocate portfolio based on composite scores"""
                df = df.sort_values('composite_score', ascending=False)
                total_score = df['composite_score'].sum()
                if total_score > 0:
                    df['weightage'] = df['composite_score'] / total_score
                else:
                    df['weightage'] = 1 / len(df) if len(df) > 0 else 0
                
                # The parent class's _allocate_portfolio handles capping and unit calculation
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
        """
        CL_v2 Strategy: Hybrid Pragati + Advanced Enhancements
        Analyzes all entries and allocates with intelligent tiered weighting.
        """
        class QuantitativeETFAnalyzer:
            """
            Advanced Quantitative ETF Analysis Engine
            Implements sophisticated multi-factor models for ETF selection based on:
            1. Statistical Anomaly Detection
            2. Multi-Timeframe Momentum Convergence
            3. Volatility-Adjusted Risk Assessment
            4. Cross-Asset Correlation Analysis
            5. Regime-Aware Factor Rotation
            """
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
                min_weight = 0.01 # Fixed minimum weight
                max_weight_limit = 0.10 # Fixed maximum weight
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

            def _allocate_remaining_cash(self, portfolio_df, remaining_cash):
                allocation_priority = portfolio_df.sort_values('composite_score', ascending=False)
                max_iterations = 100
                iteration = 0
                while remaining_cash > 0 and iteration < max_iterations:
                    allocated = False
                    for idx in allocation_priority.index:
                        etf_price = portfolio_df.at[idx, 'price']
                        total_value = (portfolio_df['units'] * portfolio_df['price']).sum()
                        if total_value == 0:
                            current_weight = 0
                        else:
                            current_weight = (portfolio_df.at[idx, 'units'] * etf_price) / total_value
                        if remaining_cash >= etf_price and current_weight < 0.12:
                            portfolio_df.at[idx, 'units'] += 1
                            remaining_cash -= etf_price
                            allocated = True
                            break
                    iteration += 1
                    if not allocated:
                        break
                return portfolio_df

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
            
            # Use total score for weighting
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
        """
        CL_v3 Strategy: Pragati Ultimate – Renaissance-Grade Quantitative Engine
        Always selects top 30 ETFs (or all if <30) with conviction-based dynamic allocation.
        """
        class UltimateETFAnalyzer:
            """
            Ultimate ETF Analysis Engine - Renaissance Technologies Grade
            Architecture:
            - Layer 1: Original Pragati Technical Indicators (60% weight)
            - Layer 2: Statistical Validation & Conviction Scoring (25% weight)
            - Layer 3: Market Regime & Risk Management (15% weight)
            """
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
    """
    MOM_v1: Adaptive Multi-Factor Momentum with Regime Detection
    
    Core Philosophy:
    - Identifies momentum persistence across multiple timeframes
    - Uses cross-sectional momentum ranking with volatility adjustment
    - Implements momentum decay functions for optimal entry timing
    - Applies regime-specific factor tilts
    
    Key Features:
    1. Dual-timeframe momentum scoring (weekly dominance)
    2. Volatility-adjusted momentum strength
    3. Momentum acceleration detection
    4. Mean reversion overlay for extreme conditions
    5. Cross-asset momentum correlation filters
    """
    
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
        
        # Convert final scores to weights for the allocation function
        total_score = df['final_momentum_score'].sum()
        if total_score > 0:
            df['weightage'] = df['final_momentum_score'] / total_score
        else:
            df['weightage'] = 1 / len(df) if len(df) > 0 else 0
        
        return self._allocate_portfolio(df, sip_amount)
    
    def _detect_market_regime(self, df):
        """Detect market regime for adaptive weighting"""
        avg_rsi = df['rsi latest'].mean()
        avg_osc = df['osc latest'].mean()
        avg_vol = (df['dev20 latest'] / df['price']).mean()
        
        # Regime classification
        if avg_rsi < 40 and avg_osc < -30:
            df['regime'] = 'OVERSOLD'
            df['regime_factor'] = 0.7  # Reduce momentum bias in oversold
        elif avg_rsi > 60 and avg_osc > 20:
            df['regime'] = 'MOMENTUM'
            df['regime_factor'] = 1.3  # Amplify momentum in trending
        elif avg_vol > 0.03:
            df['regime'] = 'VOLATILE'
            df['regime_factor'] = 0.8  # Dampen in high volatility
        else:
            df['regime'] = 'NEUTRAL'
            df['regime_factor'] = 1.0
        
        return df
    
    def _calculate_momentum_scores(self, df):
        """Calculate multi-factor momentum scores"""
        
        # 1. Price Momentum (relative to moving averages)
        df['price_mom_90'] = (df['price'] / df['ma90 latest'] - 1) * 100
        df['price_mom_200'] = (df['price'] / df['ma200 latest'] - 1) * 100
        df['price_mom_score'] = (df['price_mom_90'] * 0.6 + df['price_mom_200'] * 0.4)
        
        # 2. Oscillator Momentum (trend strength)
        # Positive when above zero, negative when below
        df['osc_momentum'] = (
            df['osc latest'] * 0.4 + 
            df['osc weekly'] * 0.6  # Weekly gets more weight
        )
        
        # 3. EMA Slope Momentum (acceleration)
        df['ema_slope'] = (
            (df['9ema osc latest'] - df['21ema osc latest']) * 0.5 +
            (df['9ema osc weekly'] - df['21ema osc weekly']) * 0.5
        )
        
        # 4. RSI Momentum (relative strength)
        # Transform RSI into momentum signal (50 is neutral)
        df['rsi_momentum'] = (
            (df['rsi latest'] - 50) * 0.4 +
            (df['rsi weekly'] - 50) * 0.6
        )
        
        # 5. Z-Score Momentum (statistical extremes)
        # Negative z-score = oversold = potential momentum reversal
        df['zscore_momentum'] = -(df['zscore latest'] * 0.45 + df['zscore weekly'] * 0.55)
        
        return df
    
    def _calculate_acceleration_factor(self, df):
        """Detect momentum acceleration (second derivative)"""
        
        # Timeframe alignment score (both daily and weekly agreeing)
        df['timeframe_alignment'] = np.where(
            (np.sign(df['osc latest']) == np.sign(df['osc weekly'])) &
            (np.sign(df['9ema osc latest']) == np.sign(df['9ema osc weekly'])),
            1.5,  # Strong alignment
            np.where(
                np.sign(df['osc latest']) == np.sign(df['osc weekly']),
                1.2,  # Moderate alignment
                1.0   # No alignment
            )
        )
        
        # EMA crossover momentum (bullish when 9 > 21)
        df['ema_crossover_strength'] = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) &
            (df['9ema osc weekly'] > df['21ema osc weekly']),
            1.4,  # Strong bullish
            np.where(
                (df['9ema osc latest'] > df['21ema osc latest']),
                1.2,  # Moderate bullish
                0.8   # Bearish
            )
        )
        
        # Acceleration score
        df['acceleration'] = df['timeframe_alignment'] * df['ema_crossover_strength']
        
        return df
    
    def _calculate_volatility_adjustment(self, df):
        """Adjust momentum for volatility (reward low-vol momentum)"""
        
        # Calculate normalized volatility
        df['volatility'] = (
            df['dev20 latest'] / df['price'] * 0.6 +
            df['dev20 weekly'] / df['price'] * 0.4
        )
        
        # Volatility-adjusted momentum (prefer low-vol high-momentum)
        # Sharpe-style ratio: momentum per unit of volatility
        epsilon = 1e-6
        df['vol_adj_factor'] = np.where(
            df['volatility'] < 0.015,
            1.3,  # Low volatility boost
            np.where(
                df['volatility'] < 0.025,
                1.0,  # Normal volatility
                np.maximum(0.7, 1 - (df['volatility'] - 0.025) * 10)  # High vol penalty
            )
        )
        
        return df
    
    def _calculate_mean_reversion_overlay(self, df):
        """Add mean reversion signal for extreme oversold in strong momentum"""
        
        # Identify extreme oversold conditions with potential momentum reversal
        df['mean_reversion_boost'] = np.where(
            (df['zscore latest'] < -2.0) &
            (df['zscore weekly'] < -1.8) &
            (df['rsi latest'] < 35) &
            (df['osc latest'] < -60) &
            (df['9ema osc latest'] > df['21ema osc latest']),  # But showing turn
            1.5,  # Strong reversal setup
            np.where(
                (df['zscore latest'] < -1.5) &
                (df['rsi latest'] < 40) &
                (df['9ema osc latest'] > df['21ema osc latest']),
                1.2,  # Moderate reversal setup
                1.0
            )
        )
        
        return df
    
    def _calculate_composite_momentum(self, df):
        """Combine all momentum factors with regime-aware weighting"""
        
        regime_weights = df['regime_factor'].iloc[0]
        
        # Normalize individual scores to 0-1 range for combining
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
        
        # Weighted composite (adjust weights based on regime)
        df['raw_momentum_score'] = (
            df['norm_price_mom'] * 0.25 +
            df['norm_osc_mom'] * 0.25 +
            df['norm_ema_slope'] * 0.20 +
            df['norm_rsi_mom'] * 0.15 +
            df['norm_zscore_mom'] * 0.15
        )
        
        # Apply multipliers
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
    """
    MOM_v2: Statistical Arbitrage with Momentum Clustering
    
    Core Philosophy:
    - Uses statistical co-movement and divergence detection
    - Implements momentum factor decomposition
    - Applies pair-wise momentum correlation
    - Uses ensemble learning principles for signal aggregation
    
    Key Features:
    1. Z-score normalized momentum across all stocks
    2. Cluster-based relative momentum
    3. Multi-horizon momentum consistency scoring
    4. Statistical edge detection (momentum vs. mean reversion)
    5. Dynamic factor exposure management
    """
    
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
        """Calculate cross-sectional statistical factors"""
        
        # Factor 1: Cross-sectional RSI z-score
        df['rsi_zscore'] = (df['rsi latest'] - df['rsi latest'].mean()) / (df['rsi latest'].std() + 1e-6)
        
        # Factor 2: Cross-sectional Oscillator z-score
        df['osc_zscore'] = (df['osc latest'] - df['osc latest'].mean()) / (df['osc latest'].std() + 1e-6)
        
        # Factor 3: Price momentum z-score (relative to MA)
        df['price_ma_ratio'] = df['price'] / df['ma90 latest']
        df['price_mom_zscore'] = (df['price_ma_ratio'] - df['price_ma_ratio'].mean()) / (df['price_ma_ratio'].std() + 1e-6)
        
        # Factor 4: Volatility-adjusted returns z-score
        df['vol_adj_ret'] = (df['price'] / df['ma20 latest'] - 1) / (df['dev20 latest'] / df['price'] + 1e-6)
        df['vol_adj_ret_zscore'] = (df['vol_adj_ret'] - df['vol_adj_ret'].mean()) / (df['vol_adj_ret'].std() + 1e-6)
        
        # Factor 5: Time-series z-score momentum
        # Using the existing z-scores as momentum indicators
        df['ts_mom_score'] = -(df['zscore latest'] * 0.6 + df['zscore weekly'] * 0.4)  # Negative because low z-score = momentum opportunity
        
        return df
    
    def _calculate_momentum_persistence(self, df):
        """Calculate momentum consistency across timeframes"""
        
        # Consistency score: how aligned are different timeframes?
        df['daily_weekly_consistency'] = np.where(
            (np.sign(df['osc latest']) == np.sign(df['osc weekly'])) &
            (np.abs(df['osc latest'] - df['osc weekly']) < 30),
            2.0,  # High consistency
            np.where(
                np.sign(df['osc latest']) == np.sign(df['osc weekly']),
                1.5,  # Moderate consistency
                0.8   # Low consistency (potential reversal)
            )
        )
        
        # EMA trend persistence
        df['ema_trend_strength'] = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) &
            (df['9ema osc weekly'] > df['21ema osc weekly']) &
            (df['osc latest'] < -40),  # But oversold
            2.5,  # Strong persistent downtrend with momentum building
            np.where(
                (df['9ema osc latest'] > df['21ema osc latest']) &
                (df['osc latest'] < -20),
                1.8,  # Moderate momentum build
                np.where(
                    (df['9ema osc latest'] < df['21ema osc latest']) &
                    (df['osc latest'] > 20),
                    0.5,  # Weak/bearish
                    1.0
                )
            )
        )
        
        # Multi-horizon RSI persistence
        df['rsi_persistence'] = np.where(
            (df['rsi latest'] < 35) & (df['rsi weekly'] < 40),
            1.5,  # Persistent oversold
            np.where(
                (df['rsi latest'] < 40) | (df['rsi weekly'] < 45),
                1.2,  # Moderately oversold
                np.where(
                    (df['rsi latest'] > 60) & (df['rsi weekly'] > 55),
                    0.7,  # Overbought
                    1.0
                )
            )
        )
        
        return df
    
    def _calculate_relative_momentum(self, df):
        """Calculate momentum relative to universe"""
        
        # Percentile ranking of key momentum metrics
        df['rsi_percentile'] = df['rsi latest'].rank(pct=True)
        df['osc_percentile'] = df['osc latest'].rank(pct=True)
        df['price_mom_percentile'] = df['price_ma_ratio'].rank(pct=True)
        
        # Relative momentum score (prefer bottom quintile for mean reversion)
        df['relative_momentum'] = (
            (1 - df['rsi_percentile']) * 0.30 +  # Lower RSI = better
            (1 - df['osc_percentile']) * 0.35 +  # Lower OSC = better
            df['price_mom_percentile'] * 0.35    # Higher price momentum = better (for continuation)
        )
        
        # Normalize to 0-2 range
        df['relative_momentum'] = df['relative_momentum'] * 2
        
        return df
    
    def _detect_momentum_clusters(self, df):
        """Identify statistical momentum clusters"""
        
        # Create composite momentum signal for clustering
        df['momentum_composite'] = (
            df['osc latest'] * 0.3 +
            df['osc weekly'] * 0.3 +
            df['rsi latest'] * 0.2 +
            df['9ema osc latest'] * 0.2
        )
        
        # --- FIX START: Use numeric labels to avoid Categorical type error ---
        # Use labels=False to get integer bin identifiers (e.g., 0, 1, 2, 3, 4)
        # This creates a numeric series, preventing the TypeError during multiplication.
        try:
            bin_labels = pd.qcut(df['momentum_composite'], q=5, labels=False, duplicates='drop')
        except ValueError: # Fallback if qcut fails
            bin_labels = pd.Series(2, index=df.index) # Assign neutral middle bin

        # Map the numeric bin identifiers to the desired float weights
        weight_map = {
            0: 2.0,  # Bin 0 (Most oversold - Q1)
            1: 1.3,  # Bin 1 (Q2)
            2: 1.0,  # Bin 2 (Q3)
            3: 0.8,  # Bin 3 (Q4)
            4: 0.6   # Bin 4 (Most overbought - Q5)
        }
        df['cluster_weight'] = bin_labels.map(weight_map)
        
        # Handle any NaN values that might result from the mapping
        df['cluster_weight'] = df['cluster_weight'].fillna(1.0)
        # --- FIX END ---
        
        return df
    
    def _calculate_edge_score(self, df):
        """Calculate statistical edge score (probability of momentum continuation)"""
        
        # Edge Factor 1: Momentum + Mean Reversion combo
        df['momentum_reversion_edge'] = np.where(
            (df['zscore latest'] < -1.8) &  # Oversold
            (df['9ema osc latest'] > df['21ema osc latest']) &  # But momentum turning
            (df['osc latest'] < -50),
            2.5,  # High edge
            np.where(
                (df['zscore latest'] < -1.2) &
                (df['osc latest'] < -40),
                1.8,  # Moderate edge
                1.0
            )
        )
        
        # Edge Factor 2: Volatility regime edge
        df['vol_regime_edge'] = np.where(
            (df['dev20 latest'] / df['price'] < 0.02) &  # Low volatility
            (df['osc latest'] < -30),  # But oversold
            1.5,  # Low vol + oversold = potential breakout
            np.where(
                (df['dev20 latest'] / df['price'] > 0.04) &  # High volatility
                (df['osc latest'] < -60),  # Deep oversold
                1.3,  # High vol panic = potential reversal
                1.0
            )
        )
        
        # Edge Factor 3: Bollinger position edge
        bb_lower = df['ma20 latest'] - 2 * df['dev20 latest']
        bb_upper = df['ma20 latest'] + 2 * df['dev20 latest']
        df['bb_position'] = (df['price'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
        
        df['bb_edge'] = np.where(
            (df['bb_position'] < 0.2) & (df['rsi latest'] < 40),
            1.8,  # Strong edge at lower band
            np.where(
                (df['bb_position'] < 0.4) & (df['rsi latest'] < 45),
                1.3,  # Moderate edge
                np.where(
                    df['bb_position'] > 0.8,
                    0.7,  # Weak edge at upper band
                    1.0
                )
            )
        )
        
        # Combine all edge factors
        df['total_edge_score'] = (
            df['momentum_reversion_edge'] * 0.35 +
            df['vol_regime_edge'] * 0.30 +
            df['bb_edge'] * 0.35
        )
        
        return df
    
    def _construct_optimal_portfolio(self, df):
        """Construct portfolio using ensemble scoring"""
        
        # Combine all signals into final composite
        df['final_composite_score'] = (
            df['rsi_zscore'] * -0.10 +  # Negative because low RSI is good
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
# NEW: MomentumMasters Strategy
# =====================================
class MomentumMasters(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        Sophisticated High-Momentum Strategy (Trend Following).
        - Identifies stocks in a strong, established uptrend.
        - Scores based on a blend of RSI strength, positive oscillator momentum, and statistical velocity.
        - Filters out stocks not in a confirmed uptrend.
        """
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'ma90 latest', 'ma200 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Trend Conviction Multiplier (Soft Filter)
        # Instead of a hard filter, we'll assign a multiplier based on trend strength.
        conditions = [
            (df['price'] > df['ma90 latest']) & (df['ma90 latest'] > df['ma200 latest']), # Golden Cross (Ideal)
            (df['price'] > df['ma90 latest']) & (df['price'] > df['ma200 latest']),   # Strong Uptrend
            (df['price'] > df['ma200 latest']),                                       # General Uptrend
        ]
        multipliers = [1.5, 1.0, 0.5]
        df['trend_conviction'] = np.select(conditions, multipliers, default=0.1) # Penalize if below MA200

        # 2. Multi-Factor Momentum Scoring
        # RSI Strength Score: Rewards RSI above 50, indicating positive momentum.
        rsi_score = np.clip((df['rsi latest'] - 50) / 30, 0, 2.0)
        
        # Oscillator Score: Rewards positive values and accelerating momentum (9EMA > 21EMA).
        osc_score = (
            (df['osc latest'] > 20).astype(int) * 0.5 +
            (df['osc weekly'] > 0).astype(int) * 0.5 +
            (df['9ema osc latest'] > df['21ema osc latest']).astype(int) * 1.0
        )
        
        # Velocity Score: Rewards high positive Z-scores, indicating statistically significant upward moves.
        velocity_score = np.clip(df['zscore latest'], 0, 3.0)

        # 3. Composite Score Calculation with weighted factors
        df['composite_score'] = (
            (rsi_score * 0.4 +
             osc_score * 0.3 +
             velocity_score * 0.3)
            * df['trend_conviction'] # Apply the soft trend filter
        )
        
        # Ensure all stocks get a non-zero score to be fully inclusive
        df['composite_score'] = df['composite_score'] + 1e-6
        
        # Rank all stocks based on the composite score
        eligible_stocks = df.sort_values('composite_score', ascending=False).copy()

        # 5. Normalize scores to create portfolio weights
        total_score = eligible_stocks['composite_score'].sum()
        # Fallback to equal weight if total score is somehow zero
        eligible_stocks['weightage'] = eligible_stocks['composite_score'] / total_score if total_score > 0 else (1 / len(eligible_stocks) if len(eligible_stocks) > 0 else 0)
        
        return self._allocate_portfolio(eligible_stocks, sip_amount)

# =====================================
# NEW: VolatilitySurfer Strategy
# =====================================
class VolatilitySurfer(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        Sophisticated High-Momentum Strategy (Breakout & Expansion).
        - Identifies stocks breaking out of low-volatility ranges.
        - Scores based on Bollinger Band breakouts, expansion in volatility, and confirmation from oscillators.
        - Aims to capture the beginning of powerful new trends.
        """
        required_columns = [
            'symbol', 'price', 'osc latest', 'zscore latest', 
            'ma90 weekly', 'ma20 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Trend Conviction Multiplier (Soft Filter)
        conditions = [
            (df['price'] > df['ma90 weekly']),   # Primary condition: above weekly MA
            (df['price'] > df['ma20 latest']),   # Secondary: above daily MA
        ]
        multipliers = [1.0, 0.5]
        df['trend_conviction'] = np.select(conditions, multipliers, default=0.1)

        # 2. Calculate Bollinger Bands to identify breakouts.
        upper_band = df['ma20 latest'] + 2 * df['dev20 latest']
        
        # Breakout Proximity Score: Rewards stocks breaking out AND those approaching a breakout.
        proximity = (df['price'] - upper_band) / (df['ma20 latest'] + 1e-6) # Normalized distance to band
        # Scores are highest for breakouts, decay for stocks further away, but still considers close ones.
        breakout_score = np.clip(1 + (proximity * 10), 0, 3.0)


        # 3. Volatility Squeeze Multiplier: Rewards breakouts from low volatility.
        # Bollinger Band Width (BBW) indicates how tight the range was.
        band_width = (2 * 2 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)
        # Give a bonus to stocks breaking out from a tight squeeze (low band width).
        squeeze_multiplier = np.select(
            [band_width < 0.05, band_width < 0.10],
            [1.5, 1.2],
            default=1.0
        )

        # 4. Confirmation Score: Ensure the breakout has underlying strength.
        # We want strong, positive oscillator and z-score values.
        osc_confirmation = np.clip(df['osc latest'] / 50, 0, 2.0)
        zscore_confirmation = np.clip(df['zscore latest'], 0, 3.0)
        confirmation_score = (osc_confirmation * 0.5 + zscore_confirmation * 0.5)

        # 5. Composite Score Calculation
        df['composite_score'] = (
            breakout_score * squeeze_multiplier * confirmation_score * df['trend_conviction']
        )
        
        # Ensure all stocks get a non-zero score to be fully inclusive
        df['composite_score'] = df['composite_score'] + 1e-6
        
        # Rank all stocks based on the composite score
        eligible_stocks = df.sort_values('composite_score', ascending=False).copy()

        # 7. Normalize scores to create portfolio weights
        total_score = eligible_stocks['composite_score'].sum()
        # Fallback to equal weight if total score is somehow zero
        eligible_stocks['weightage'] = eligible_stocks['composite_score'] / total_score if total_score > 0 else (1 / len(eligible_stocks) if len(eligible_stocks) > 0 else 0)
        
        return self._allocate_portfolio(eligible_stocks, sip_amount)

# =====================================
# NEW: AdaptiveVolBreakout Strategy
# =====================================
# Enhances VolatilitySurfer by incorporating multi-timeframe alignment,
# momentum confirmation via EMA crossovers, and adaptive volatility thresholds
# to capture stronger, more sustainable breakouts while reducing false signals.
class AdaptiveVolBreakout(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        Adaptive Volatility Breakout: Multi-Timeframe Enhanced Breakout Strategy.
        - Builds on VolatilitySurfer but adds weekly confirmation, EMA acceleration,
          and dynamic vol thresholds to filter for higher-conviction breakouts.
        - Aims to outperform by reducing whipsaws in choppy markets.
        """
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma20 latest', 'ma20 weekly',
            'dev20 latest', 'dev20 weekly', 'ma90 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Multi-Timeframe Trend Conviction (Enhanced)
        conditions = [
            (df['price'] > df['ma90 weekly']) & (df['price'] > df['ma20 weekly']),  # Strong weekly uptrend
            (df['price'] > df['ma90 weekly']) | (df['price'] > df['ma20 weekly']),   # Moderate weekly alignment
        ]
        multipliers = [1.3, 0.8]
        df['trend_conviction'] = np.select(conditions, multipliers, default=0.3)

        # 2. Adaptive Bollinger Breakout Score
        # Daily Upper Band
        upper_daily = df['ma20 latest'] + 2 * df['dev20 latest']
        dist_daily = (df['price'] - upper_daily) / (df['ma20 latest'] + 1e-6)
        breakout_daily = np.clip(1 + (dist_daily * 8), 0.2, 2.5)  # Softer penalty for below

        # Weekly Upper Band for confirmation
        upper_weekly = df['ma20 weekly'] + 2 * df['dev20 weekly']
        dist_weekly = (df['price'] - upper_weekly) / (df['ma20 weekly'] + 1e-6)
        breakout_weekly = np.clip(1 + (dist_weekly * 6), 0.3, 2.0)

        # Combined Breakout Score (requires some weekly alignment)
        df['breakout_score'] = breakout_daily * (breakout_weekly ** 0.5)  # Geometric mean for balance

        # 3. Dynamic Volatility Squeeze Multiplier
        # Adaptive threshold based on market avg vol
        avg_vol = (df['dev20 latest'] / df['price']).mean()
        daily_bbw = (4 * df['dev20 latest']) / (df['ma20 latest'] + 1e-6)
        weekly_bbw = (4 * df['dev20 weekly']) / (df['ma20 weekly'] + 1e-6)
        combined_bbw = (daily_bbw * 0.6 + weekly_bbw * 0.4)

        vol_threshold = max(0.04, avg_vol * 1.2)  # Adaptive
        squeeze_multiplier = np.select(
            [combined_bbw < vol_threshold * 0.8, combined_bbw < vol_threshold],
            [1.6, 1.3],  # Higher reward for tighter squeezes
            default=0.9
        )

        # 4. Momentum Confirmation Score (EMA Acceleration + Oscillator)
        ema_acceleration = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) &
            (df['9ema osc weekly'] > df['21ema osc weekly']),
            1.6,
            np.where(
                (df['9ema osc latest'] > df['21ema osc latest']) |
                (df['9ema osc weekly'] > df['21ema osc weekly']),
                1.2,
                0.7
            )
        )
        osc_confirm = np.clip((df['osc latest'] + df['osc weekly']) / 100, 0, 2.0)
        df['momentum_confirm'] = ema_acceleration * osc_confirm

        # 5. Overbought Filter (RSI Cap)
        rsi_filter = np.where(df['rsi latest'] < 75, 1.0, np.clip(100 - df['rsi latest'], 0.4, 1.0))

        # 6. Z-Score Statistical Boost
        z_boost = np.clip(df['zscore latest'] + 1, 0.5, 2.5)  # Shift to positive

        # 7. Composite Score
        df['composite_score'] = (
            df['breakout_score'] * squeeze_multiplier * df['momentum_confirm'] *
            df['trend_conviction'] * rsi_filter * z_boost
        )

        # Ensure inclusivity with minimum score
        df['composite_score'] = np.maximum(df['composite_score'], 0.01)

        # Sort and weight
        eligible_stocks = df.sort_values('composite_score', ascending=False).copy()
        total_score = eligible_stocks['composite_score'].sum()
        eligible_stocks['weightage'] = (
            eligible_stocks['composite_score'] / total_score
            if total_score > 0 else 1 / len(eligible_stocks)
        )

        return self._allocate_portfolio(eligible_stocks, sip_amount)

# =====================================
# NEW: VolReversalHarvester Strategy
# =====================================
# Complements VolatilitySurfer by focusing on mean-reversion in volatility expansions,
# harvesting dips in high-vol environments with oversold confirmations.
# Outperforms in ranging/choppy markets where breakouts fail.
class VolReversalHarvester(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        Volatility Reversal Harvester: Mean-Reversion in High-Vol Regimes.
        - Targets oversold stocks during vol expansions (inverse of breakouts).
        - Uses multi-timeframe oversold alignment, z-score extremes, and vol decay signals.
        - Switches dynamically based on individual stock vol regime to capture reversals.
        """
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            'zscore latest', 'zscore weekly', 'ma20 latest', 'ma20 weekly',
            'dev20 latest', 'dev20 weekly', 'ma200 latest', 'ma200 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Individual Stock Vol Regime Detection
        # High vol if dev20/price > median + std
        vol_norm_daily = df['dev20 latest'] / (df['price'] + 1e-6)
        vol_norm_weekly = df['dev20 weekly'] / (df['price'] + 1e-6)
        avg_vol_stock = (vol_norm_daily * 0.6 + vol_norm_weekly * 0.4)
        market_vol_median = avg_vol_stock.median()
        market_vol_std = avg_vol_stock.std()
        df['high_vol_regime'] = (avg_vol_stock > market_vol_median + market_vol_std).astype(int)

        # 2. Mean-Reversion Conviction (Oversold Multiplier)
        conditions = [
            (df['rsi latest'] < 30) & (df['rsi weekly'] < 35),  # Deep oversold alignment
            (df['rsi latest'] < 40) | (df['rsi weekly'] < 45),   # Moderate oversold
        ]
        multipliers = [1.8, 1.2]
        df['oversold_conviction'] = np.select(conditions, multipliers, default=0.4)

        # 3. Lower Bollinger Reversion Score
        # Focus on lower band proximity for buys
        lower_daily = df['ma20 latest'] - 2 * df['dev20 latest']
        dist_lower_daily = (lower_daily - df['price']) / (df['ma20 latest'] + 1e-6)
        reversion_daily = np.clip(1 + (dist_lower_daily * 8), 0.2, 2.5)

        lower_weekly = df['ma20 weekly'] - 2 * df['dev20 weekly']
        dist_lower_weekly = (lower_weekly - df['price']) / (df['ma20 weekly'] + 1e-6)
        reversion_weekly = np.clip(1 + (dist_lower_weekly * 6), 0.3, 2.0)

        df['reversion_score'] = reversion_daily * (reversion_weekly ** 0.5)

        # 4. Oscillator Oversold Confirmation
        # Reward extreme negatives with signs of stabilization
        osc_stabilization = np.where(
            (df['osc latest'] < -60) & (df['9ema osc latest'] > df['osc latest'] * 0.9),  # Bouncing
            1.7,
            np.where(df['osc latest'] < -40, 1.3, 0.8)
        )
        df['osc_confirm'] = osc_stabilization * np.clip((df['osc weekly'] / -100), 0, 1.5)

        # 5. Z-Score Extremity Boost (Deep Oversold)
        z_extreme = np.clip(-df['zscore latest'], 0, 3.0) * np.clip(-df['zscore weekly'], 0, 2.0)
        df['z_boost'] = np.sqrt(z_extreme) + 0.5  # Concave to reward extremes moderately

        # 6. Vol Expansion Decay Multiplier
        # In high vol, reward if vol is peaking (but we approximate with dev20 ratio)
        vol_decay = np.where(
            df['high_vol_regime'] == 1,
            np.clip(1 + (vol_norm_daily - vol_norm_weekly), 0.8, 1.5),  # Daily vol > weekly = expansion
            1.0
        )

        # 7. Trend Safety (Avoid Deep Bear Markets)
        # Penalize if far below MA200
        ma200_dist = (df['price'] / df['ma200 latest'] - 1)
        trend_safety = np.clip(ma200_dist + 0.5, 0.3, 1.5)  # -50% = 0.3, flat=0.5, +50%=1.5

        # 8. Composite Score (Activate in High Vol)
        df['composite_score'] = (
            df['oversold_conviction'] * df['reversion_score'] * df['osc_confirm'] *
            df['z_boost'] * vol_decay * trend_safety * (1 + df['high_vol_regime'] * 0.5)
        )

        # Ensure inclusivity
        df['composite_score'] = np.maximum(df['composite_score'], 0.01)

        # Sort and weight
        eligible_stocks = df.sort_values('composite_score', ascending=False).copy()
        total_score = eligible_stocks['composite_score'].sum()
        eligible_stocks['weightage'] = (
            eligible_stocks['composite_score'] / total_score
            if total_score > 0 else 1 / len(eligible_stocks)
        )

        return self._allocate_portfolio(eligible_stocks, sip_amount)
        
# =====================================
# NEW: AlphaSurge Strategy
# =====================================
# Thesis: Alpha Surge Capture - Identifies stocks with explosive alpha generation potential
# by detecting synchronized multi-timeframe momentum surges (EMA crossovers + oscillator velocity)
# combined with statistical undervaluation (negative z-score pullbacks in uptrends).
# Maximizes returns by pyramid allocation: 70% to top 10%, 20% to next 20%, 10% to rest,
# ensuring heavy concentration in highest-conviction surge candidates.
class AlphaSurge(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        AlphaSurge: Explosive Momentum Surge Detector.
        - Core Thesis: Surge in alpha occurs when short-term momentum (9EMA > 21EMA) accelerates
          through a pullback (negative z-score) in an established uptrend (price > MA90/200).
        - Allocation: Pyramid weighting - heavy on top decile for max return capture.
        """
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Uptrend Filter (Base Conviction)
        uptrend_score = np.where(
            (df['price'] > df['ma90 latest']) & (df['ma90 latest'] > df['ma200 latest']) &
            (df['price'] > df['ma90 weekly']),
            1.5,  # Strong uptrend
            np.where(
                (df['price'] > df['ma200 latest']) | (df['price'] > df['ma90 weekly']),
                1.0,  # Moderate uptrend
                0.2   # Downtrend penalty
            )
        )
        df['uptrend_conviction'] = uptrend_score

        # 2. Momentum Surge Score (EMA Crossover Velocity)
        # Velocity: Rate of change in oscillator via EMA diff
        daily_velocity = (df['9ema osc latest'] - df['21ema osc latest']) / (df['21ema osc latest'] + 1e-6)
        weekly_velocity = (df['9ema osc weekly'] - df['21ema osc weekly']) / (df['21ema osc weekly'] + 1e-6)
        surge_velocity = np.tanh(daily_velocity * 0.6 + weekly_velocity * 0.4) * 2 + 1  # Normalize to 0-3

        # Oscillator Surge Confirmation (positive and accelerating)
        osc_surge = np.where(
            (df['osc latest'] > 10) & (df['osc weekly'] > 5) & (daily_velocity > 0),
            2.0,
            np.where(
                (df['osc latest'] > 0) | (df['osc weekly'] > 0),
                1.2,
                0.5
            )
        )
        df['surge_score'] = surge_velocity * osc_surge

        # 3. Pullback Opportunity (Undervaluation via Z-Score)
        # Negative z-score indicates pullback in uptrend - high return potential
        pullback_potential = np.clip(-df['zscore latest'] * 0.7 - df['zscore weekly'] * 0.3, 0, 3.0)
        df['pullback_score'] = pullback_potential + 0.5  # Minimum baseline

        # 4. RSI Surge Readiness (Not overbought, building strength)
        rsi_readiness = np.where(
            (df['rsi latest'] > 50) & (df['rsi latest'] < 70),
            1.5,
            np.where(df['rsi latest'] > 40, 1.1, 0.6)
        )
        df['rsi_factor'] = rsi_readiness * np.where(df['rsi weekly'] > 45, 1.2, 0.8)

        # 5. Composite Alpha Score
        df['alpha_score'] = (
            df['surge_score'] * df['pullback_score'] * df['rsi_factor'] * df['uptrend_conviction']
        )

        # Ensure positive scores
        df['alpha_score'] = np.maximum(df['alpha_score'], 0.01)

        # 6. Pyramid Allocation Logic
        df_sorted = df.sort_values('alpha_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        # Top 10%: 70% allocation (equal within tier)
        top10_end = max(1, int(n * 0.1))
        df_sorted.loc[:top10_end-1, 'weightage'] = 0.7 / top10_end if top10_end > 0 else 0

        # Next 20%: 20% allocation
        next20_end = min(n, top10_end + int(n * 0.2))
        if next20_end > top10_end:
            count = next20_end - top10_end
            df_sorted.loc[top10_end:next20_end-1, 'weightage'] = 0.2 / count

        # Remaining 70%: 10% allocation (equal)
        remaining_start = next20_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.1 / count

        # Normalize to sum=1
        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

class ReturnPyramid(BaseStrategy):
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        """
        ReturnPyramid: Extreme Concentration on High-Return Outliers.
        - Core Thesis: Highest returns come from stocks at statistical extremes (high |z-score|)
          with persistent momentum (aligned oscillators) in favorable trends, projecting
          explosive mean-reversion + continuation hybrids.
        - Allocation: Ultra-pyramid - 80% on top 5 for max return amplification.
        """
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'zscore weekly',
            'ma90 latest', 'ma200 latest', 'ma90 weekly', 'ma200 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Projected Return Potential (Z-Score Magnitude)
        # High |z-score| = high reversion potential; sign matters for direction
        # Focus on positive extremes for upside, but include negative with momentum flip
        z_magnitude = np.abs(df['zscore latest']) * 0.6 + np.abs(df['zscore weekly']) * 0.4
        z_direction = np.where(
            df['zscore latest'] > 1.5, 2.0,  # Strong positive
            np.where(
                (df['zscore latest'] < -1.5) & (df['9ema osc latest'] > 0),  # Negative but turning
                1.5,
                np.where(df['zscore latest'] > 0, 1.2, 0.8)
            )
        )
        df['return_potential'] = z_magnitude * z_direction

        # 2. Momentum Persistence Score
        # Aligned signs across timeframes and EMAs
        persistence = np.where(
            (np.sign(df['osc latest']) == np.sign(df['osc weekly'])) &
            (np.sign(df['9ema osc latest']) == np.sign(df['21ema osc latest'])),
            2.2,
            np.where(
                np.sign(df['osc latest']) == np.sign(df['osc weekly']),
                1.5,
                1.0
            )
        )
        df['persistence_score'] = persistence * (1 + np.abs(df['osc latest'] - df['osc weekly']) / 200)  # Reward extremes

        # 3. Trend Leverage (Amplifies Returns)
        # Price relative to MAs, weighted long-term
        trend_leverage = (
            (df['price'] / df['ma200 latest'] - 1) * 0.5 +
            (df['price'] / df['ma90 weekly'] - 1) * 0.5
        )
        df['trend_leverage'] = np.tanh(trend_leverage) * 1.5 + 0.5  # Normalize to 0-2

        # 4. RSI Amplification (Avoid Exhaustion)
        rsi_amp = np.where(
            (df['rsi latest'] > 40) & (df['rsi latest'] < 65), 1.4,  # Sweet spot
            np.where(df['rsi latest'] > 65, 0.9, 1.1)  # Penalize overbought, reward building
        )
        df['rsi_amp'] = rsi_amp * np.where(df['rsi weekly'] > 50, 1.1, 0.9)

        # 5. Composite Return Projection
        df['projected_return'] = (
            df['return_potential'] * df['persistence_score'] *
            df['trend_leverage'] * df['rsi_amp']
        )

        # Ensure positive
        df['projected_return'] = np.maximum(df['projected_return'], 0.01)

        # 6. Ultra-Pyramid Allocation
        df_sorted = df.sort_values('projected_return', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        # Top 5 stocks: 80% allocation (equal within)
        top5_end = min(5, n)
        if top5_end > 0:
            df_sorted.loc[:top5_end-1, 'weightage'] = 0.8 / top5_end

        # Next 10 stocks: 15%
        next10_end = min(n, top5_end + 10)
        if next10_end > top5_end:
            count = next10_end - top5_end
            df_sorted.loc[top5_end:next10_end-1, 'weightage'] = 0.15 / count

        # Remaining: 5% equal
        remaining_start = next10_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.05 / count
        elif top5_end < n and next10_end == top5_end:
            # If no next10 assigned (e.g., n < top5 +1), adjust remaining to 15% + 5% = 20%? Wait, no: if next10_end == top5_end, means n <= top5_end, so no remaining
            pass
        else:
            # If less than full tiers, but some remaining after top5
            if top5_end < n:
                remaining_count = n - top5_end
                df_sorted.loc[top5_end:, 'weightage'] = 0.20 / remaining_count  # 15% + 5% combined

        # Normalize
        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w
        else:
            df_sorted['weightage'] = 1.0 / n

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: MomentumCascade Strategy
# =====================================
class MomentumCascade(BaseStrategy):
    """
    MomentumCascade: Cascading Momentum Amplifier.
    - Thesis: Alpha emerges from cascading momentum where short-term signals (daily EMA/OSC) lead and amplify longer-term trends (weekly), creating explosive upside.
    - Focus: Heavy weighting to stocks showing daily surge pulling weekly into alignment.
    - Allocation: 60% to top 8, 25% to next 12, 15% to rest for concentrated alpha capture.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Daily Surge Score (Short-term lead)
        daily_surge = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) & (df['osc latest'] > df['osc weekly']),
            2.5,  # Strong daily lead
            np.where(df['9ema osc latest'] > df['21ema osc latest'], 1.8, 1.0)
        )
        df['daily_surge'] = daily_surge * np.clip((df['rsi latest'] - 50) / 20, 0, 2)

        # 2. Weekly Alignment Pull (Long-term catch-up)
        weekly_pull = np.where(
            (df['osc weekly'] > -20) & (df['9ema osc weekly'] > df['21ema osc weekly'] * 0.95),
            2.0,  # Weekly starting to align
            np.where(df['osc weekly'] > -40, 1.4, 0.7)
        )
        df['weekly_pull'] = weekly_pull * (1 - np.clip(df['zscore weekly'], -2, 0))  # Reward undervalued weekly

        # 3. Cascade Multiplier (Daily pulling weekly)
        cascade_mult = np.where(
            (df['daily_surge'] > 1.5) & (df['weekly_pull'] > 1.2),
            2.2,  # Full cascade
            1.0
        )

        # 4. Trend Anchor (Sustained uptrend)
        trend_anchor = np.where(
            (df['price'] > df['ma90 latest']) & (df['ma90 latest'] > df['ma200 weekly']),
            1.6,
            np.where(df['price'] > df['ma200 latest'], 1.1, 0.6)
        )

        # 5. Composite Cascade Score
        df['cascade_score'] = df['daily_surge'] * df['weekly_pull'] * cascade_mult * trend_anchor

        # Minimum score
        df['cascade_score'] = np.maximum(df['cascade_score'], 0.01)

        # 6. Concentrated Allocation
        df_sorted = df.sort_values('cascade_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        # Top 8: 60%
        top8_end = min(8, n)
        df_sorted.loc[:top8_end-1, 'weightage'] = 0.6 / top8_end

        # Next 12: 25%
        next12_end = min(n, top8_end + 12)
        if next12_end > top8_end:
            count = next12_end - top8_end
            df_sorted.loc[top8_end:next12_end-1, 'weightage'] = 0.25 / count

        # Rest: 15%
        remaining_start = next12_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.15 / count
        else:
            # Adjust if fewer stocks
            if top8_end < n:
                remaining_count = n - top8_end
                df_sorted.loc[top8_end:, 'weightage'] = 0.4 / remaining_count  # 25% + 15%

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: AlphaVortex Strategy
# =====================================
class AlphaVortex(BaseStrategy):
    """
    AlphaVortex: Converging Indicator Vortex for Alpha.
    - Thesis: Alpha vortex forms when momentum indicators converge (RSI rising, OSC turning positive, Z-score normalizing) in a low-vol uptrend, sucking in capital for rapid gains.
    - Focus: Detect convergence strength for high-conviction entries.
    - Allocation: 70% to top 7, 20% to next 13, 10% to rest.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'zscore weekly',
            'ma90 latest', 'ma200 latest', 'dev20 latest', 'dev20 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. RSI Convergence (Rising from neutral)
        rsi_conv = np.clip((df['rsi latest'] - df['rsi weekly']) / 10 + 1, 0.5, 2.5) if len(df) > 1 else 1.0
        df['rsi_conv'] = np.where(df['rsi latest'] > 50, rsi_conv, rsi_conv * 0.7)

        # 2. OSC Turning Score (From negative to positive trajectory)
        osc_turn = np.where(
            (df['osc latest'] > 0) & (df['osc weekly'] > -30),
            2.3,
            np.where(df['9ema osc latest'] > df['21ema osc latest'], 1.6, 0.9)
        )
        df['osc_turn'] = osc_turn * np.clip(df['osc latest'] / 50, 0, 1.5)

        # 3. Z-Score Normalization (Pulling toward mean from extremes)
        z_norm = np.clip(1 - np.abs(df['zscore latest']), 0.3, 2.0) * np.clip(1 - np.abs(df['zscore weekly']), 0.3, 1.5)
        df['z_norm'] = z_norm if np.isscalar(z_norm) else z_norm.values  # Handle scalar

        # 4. Low-Vol Uptrend Anchor
        vol_adjust = np.where(
            ((df['dev20 latest'] / df['price']) + (df['dev20 weekly'] / df['price'])) / 2 < 0.025,
            1.4,
            0.8
        )
        trend_anchor = np.where(df['price'] > df['ma90 latest'], vol_adjust * 1.3, vol_adjust)

        # 5. Vortex Convergence Score
        df['vortex_score'] = df['rsi_conv'] * df['osc_turn'] * df['z_norm'] * trend_anchor

        df['vortex_score'] = np.maximum(df['vortex_score'], 0.01)

        # 6. Concentrated Allocation
        df_sorted = df.sort_values('vortex_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        top7_end = min(7, n)
        df_sorted.loc[:top7_end-1, 'weightage'] = 0.7 / top7_end

        next13_end = min(n, top7_end + 13)
        if next13_end > top7_end:
            count = next13_end - top7_end
            df_sorted.loc[top7_end:next13_end-1, 'weightage'] = 0.2 / count

        remaining_start = next13_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.1 / count
        else:
            if top7_end < n:
                remaining_count = n - top7_end
                df_sorted.loc[top7_end:, 'weightage'] = 0.3 / remaining_count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: SurgeSentinel Strategy
# =====================================
class SurgeSentinel(BaseStrategy):
    """
    SurgeSentinel: Post-Consolidation Momentum Sentinel.
    - Thesis: Alpha surges post-consolidation (low vol contraction followed by momentum breakout), guarded by sentinel confirmations (EMA alignment, RSI build).
    - Focus: Time the release from compression for max alpha.
    - Allocation: 65% to top 6, 25% to next 14, 10% to rest.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest',
            'ma20 latest', 'ma20 weekly', 'dev20 latest', 'dev20 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Consolidation Detection (Low Vol Contraction)
        bb_width_daily = (4 * df['dev20 latest']) / df['ma20 latest']
        bb_width_weekly = (4 * df['dev20 weekly']) / df['ma20 weekly']
        contraction = np.where(
            (bb_width_daily < bb_width_weekly * 0.8) & (bb_width_daily < 0.06),
            2.4,  # Tight daily after weekly
            np.where(bb_width_daily < 0.08, 1.5, 0.6)
        )
        df['contraction'] = contraction

        # 2. Momentum Release Score (Breakout from consolidation)
        release = np.where(
            (df['price'] > df['ma20 latest']) & (df['osc latest'] > 10),
            2.1,
            np.where(df['9ema osc latest'] > df['21ema osc latest'], 1.4, 0.8)
        )
        df['release'] = release * np.clip(df['zscore latest'], -1, 2)  # Penalize deep oversold

        # 3. Sentinel Confirmation (RSI Building)
        rsi_build = np.where(
            (df['rsi latest'] > df['rsi weekly']) & (df['rsi latest'] > 55),
            1.8,
            np.where(df['rsi latest'] > 45, 1.2, 0.7)
        )
        df['rsi_sentinel'] = rsi_build

        # 4. Surge Multiplier (Contraction + Release)
        surge_mult = df['contraction'] * df['release'] * df['rsi_sentinel']

        # 5. Composite Surge Score
        df['surge_score'] = surge_mult

        df['surge_score'] = np.maximum(df['surge_score'], 0.01)

        # 6. Allocation
        df_sorted = df.sort_values('surge_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        top6_end = min(6, n)
        df_sorted.loc[:top6_end-1, 'weightage'] = 0.65 / top6_end

        next14_end = min(n, top6_end + 14)
        if next14_end > top6_end:
            count = next14_end - top6_end
            df_sorted.loc[top6_end:next14_end-1, 'weightage'] = 0.25 / count

        remaining_start = next14_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.1 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: VelocityVortex Strategy
# =====================================
class VelocityVortex(BaseStrategy):
    """
    VelocityVortex: High-Velocity Momentum with Alpha Edge.
    - Thesis: Alpha from high-velocity moves (rapid EMA/price acceleration) in low-risk (high Sharpe-like) setups, vortex pulls in sustained momentum.
    - Focus: Velocity as ROC in indicators, edged by vol-adjusted returns.
    - Allocation: 75% to top 4, 15% to next 6, 10% to rest for velocity concentration.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'osc weekly', '9ema osc latest', '21ema osc latest',
            'zscore latest', 'ma90 latest', 'ma200 latest', 'dev20 latest', 'dev20 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Price Velocity (ROC vs MA)
        price_roc = (df['price'] / df['ma90 latest'] - 1) * 100
        df['price_vel'] = np.clip(price_roc / 10, 0, 3)  # Normalize

        # 2. Oscillator Velocity (EMA diff rate)
        osc_vel_daily = (df['9ema osc latest'] - df['21ema osc latest']) / (df['21ema osc latest'] + 1e-6)
        osc_vel_weekly = (df['osc weekly'] - df['osc latest']) / (df['osc latest'] + 1e-6) * -1  # Weekly lag
        df['osc_vel'] = np.tanh(osc_vel_daily * 0.7 + osc_vel_weekly * 0.3) * 2 + 1

        # 3. Alpha Edge (Vol-Adjusted Velocity ~ Sharpe)
        avg_vol = (df['dev20 latest'] / df['price'] * 0.6 + df['dev20 weekly'] / df['price'] * 0.4)
        edge = np.where(avg_vol > 0, (df['price_vel'] + df['osc_vel']) / avg_vol, df['price_vel'] + df['osc_vel'])
        df['alpha_edge'] = np.clip(edge / edge.mean(), 0.5, 2.5) if len(df) > 1 else edge

        # 4. Z-Score Velocity Boost (Accelerating normalization)
        z_vel_boost = np.clip(-np.abs(df['zscore latest']), 0, 1.5) * df['osc_vel']
        df['z_vel'] = 1 + z_vel_boost

        # 5. Composite Velocity Score
        df['velocity_score'] = df['price_vel'] * df['osc_vel'] * df['alpha_edge'] * df['z_vel']

        df['velocity_score'] = np.maximum(df['velocity_score'], 0.01)

        # 6. Allocation
        df_sorted = df.sort_values('velocity_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        top4_end = min(4, n)
        df_sorted.loc[:top4_end-1, 'weightage'] = 0.75 / top4_end

        next6_end = min(n, top4_end + 6)
        if next6_end > top4_end:
            count = next6_end - top4_end
            df_sorted.loc[top4_end:next6_end-1, 'weightage'] = 0.15 / count

        remaining_start = next6_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.1 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: BreakoutAlphaHunter Strategy
# =====================================
class BreakoutAlphaHunter(BaseStrategy):
    """
    BreakoutAlphaHunter: Multi-Layer Momentum Breakout Hunter.
    - Thesis: Hunt alpha in breakouts layered with momentum confirmations (OSC surge, RSI momentum, Z-score breakout) for high-probability captures.
    - Focus: Layered signals to filter true alpha-generating breakouts.
    - Allocation: 68% to top 9, 22% to next 11, 10% to rest.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'zscore weekly',
            'ma20 latest', 'dev20 latest', 'ma90 latest', 'ma200 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Breakout Layer (Price > Upper BB + MA)
        upper_bb = df['ma20 latest'] + 2 * df['dev20 latest']
        breakout_layer = np.where(
            (df['price'] > upper_bb) & (df['price'] > df['ma90 latest']),
            2.6,
            np.where(df['price'] > df['ma20 latest'], 1.7, 0.5)
        )
        df['breakout_layer'] = breakout_layer

        # 2. OSC Surge Layer
        osc_surge_layer = np.where(
            (df['osc latest'] > 20) & (df['9ema osc latest'] > df['21ema osc latest'] + 5),
            2.4,
            np.where(df['osc latest'] > 0, 1.5, 0.8)
        )
        df['osc_layer'] = osc_surge_layer * np.where(df['osc weekly'] > -10, 1.3, 0.9)

        # 3. RSI Momentum Layer
        rsi_mom_layer = np.clip((df['rsi latest'] - df['rsi weekly']) / 15 + 1, 0.6, 2.2)
        df['rsi_layer'] = np.where(df['rsi latest'] > 55, rsi_mom_layer * 1.4, rsi_mom_layer)

        # 4. Z-Score Breakout Layer (Extreme to breakout)
        z_break_layer = np.where(
            (df['zscore latest'] > 1.2) | ((df['zscore latest'] < -1.2) & (df['osc latest'] > 0)),
            2.0,
            1.0
        )
        df['z_layer'] = z_break_layer * np.clip(1 - np.abs(df['zscore weekly']), 0.5, 1.5)

        # 5. Composite Hunter Score
        df['hunter_score'] = (
            df['breakout_layer'] * df['osc_layer'] * df['rsi_layer'] * df['z_layer']
        ) * np.where(df['price'] > df['ma200 weekly'], 1.2, 0.8)

        df['hunter_score'] = np.maximum(df['hunter_score'], 0.01)

        # 6. Allocation
        df_sorted = df.sort_values('hunter_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        top9_end = min(9, n)
        df_sorted.loc[:top9_end-1, 'weightage'] = 0.68 / top9_end

        next11_end = min(n, top9_end + 11)
        if next11_end > top9_end:
            count = next11_end - top9_end
            df_sorted.loc[top9_end:next11_end-1, 'weightage'] = 0.22 / count

        remaining_start = next11_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.1 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: ExtremeMomentumBlitz Strategy
# =====================================
class ExtremeMomentumBlitz(BaseStrategy):
    """
    ExtremeMomentumBlitz: Blitzkrieg Momentum Assault.
    - Thesis: Launch aggressive alpha assaults on extreme momentum outliers where daily velocity explodes beyond weekly norms, amplified by z-score velocity flips for 3x+ return spikes.
    - Focus: Target top 3% velocity freaks in uptrends for blitz allocation.
    - Allocation: 85% to top 3 stocks, 10% to next 7, 5% to rest – ultra-extreme concentration.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'osc weekly', '9ema osc latest', '21ema osc latest',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest', 'rsi latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Daily Velocity Explosion (Extreme ROC in OSC)
        daily_vel = (df['9ema osc latest'] - df['21ema osc latest']) / (df['21ema osc latest'] + 1e-6)
        df['daily_explosion'] = np.clip(daily_vel * 5, 0, 4)  # Extreme scaling

        # 2. Weekly Norm Breach (Daily >> Weekly)
        breach = np.where(daily_vel > df['osc weekly'] * 2, 3.5, np.clip(daily_vel / (df['osc weekly'] + 1e-6), 0.5, 2.5))
        df['breach_score'] = breach

        # 3. Z-Score Velocity Flip (From extreme negative to surge)
        flip_boost = np.where(
            (df['zscore latest'] < -1.5) & (daily_vel > 0.1),
            4.0,  # Massive flip potential
            np.clip(-df['zscore latest'] + 1, 0.5, 2.0)
        )
        df['flip_boost'] = flip_boost

        # 4. RSI Overdrive (High but not exhausted)
        overdrive = np.where((df['rsi latest'] > 70) & (df['rsi latest'] < 85), 2.8, 1.2)
        df['rsi_overdrive'] = overdrive

        # 5. Uptrend Siege (Price dominance)
        siege = np.where(df['price'] > df['ma90 latest'] * 1.05, 2.2, 0.8)
        df['siege'] = siege

        # 6. Blitz Composite
        df['blitz_score'] = (
            df['daily_explosion'] * df['breach_score'] * df['flip_boost'] *
            df['rsi_overdrive'] * df['siege']
        )

        df['blitz_score'] = np.maximum(df['blitz_score'], 0.01)

        # 7. Ultra-Concentration
        df_sorted = df.sort_values('blitz_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        top3_end = min(3, n)
        df_sorted.loc[:top3_end-1, 'weightage'] = 0.85 / top3_end

        next7_end = min(n, top3_end + 7)
        if next7_end > top3_end:
            count = next7_end - top3_end
            df_sorted.loc[top3_end:next7_end-1, 'weightage'] = 0.10 / count

        remaining_start = next7_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.05 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: HyperAlphaIgniter Strategy
# =====================================
class HyperAlphaIgniter(BaseStrategy):
    """
    HyperAlphaIgniter: Ignite Hyper-Returns via Signal Ignition.
    - Thesis: Ignite 5x+ alpha by detecting ignition points where multi-indicator fuses (RSI ignition, OSC spark, Z-fire) light up in low-vol powder kegs for chain-reaction gains.
    - Focus: Fuse convergence at extremes for ignition scoring.
    - Allocation: 82% to top 2, 12% to next 8, 6% to rest – ignition-level focus.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', 'zscore latest', 'dev20 latest', 'ma200 latest'
        ]
        
        # FIX 1: Add .copy() to ensure we are working on a standalone DataFrame
        df = self._clean_data(df, required_columns).copy()

        # 1. RSI Ignition (Rapid rise from oversold)
        rsi_ignite = np.where(
            (df['rsi latest'] > df['rsi weekly'] + 15) & (df['rsi latest'] > 60),
            3.8,  # Hot ignition
            np.clip((df['rsi latest'] - 40) / 20, 0.4, 2.0)
        )
        df['rsi_ignite'] = rsi_ignite

        # 2. OSC Spark (Sudden positive spike)
        spark = np.where(
            (df['osc latest'] > 30) & (df['osc weekly'] < 0),
            4.2,  # Spark from negative
            np.clip(df['osc latest'] / 50, 0.3, 2.5)
        )
        df['osc_spark'] = spark

        # 3. Z-Fire (Extreme z-score with positive turn)
        z_fire = np.where(
            np.abs(df['zscore latest']) > 2 & (df['9ema osc latest'] > 0),
            3.5,
            np.clip(1 + df['zscore latest'], 0.2, 2.2)
        )
        df['z_fire'] = z_fire

        # 4. Low-Vol Powder Keg (Tight bands pre-explosion)
        keg = np.where(df['dev20 latest'] / df['price'] < 0.015, 2.8, 1.0)
        df['keg'] = keg * np.where(df['price'] > df['ma200 latest'], 1.5, 0.7)

        # 5. Fuse Convergence (All igniting)
        # FIX 2: Correcting np.minimum for 3 arguments by nesting
        convergence = np.minimum(
            df['rsi_ignite'], 
            np.minimum(df['osc_spark'], df['z_fire'])
        )
        df['ignite_score'] = convergence * df['keg']

        df['ignite_score'] = np.maximum(df['ignite_score'], 0.01)

        # 6. Hyper-Allocation
        df_sorted = df.sort_values('ignite_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        # Initialize weightage column
        df_sorted['weightage'] = 0.0

        top2_end = min(2, n)
        df_sorted.loc[:top2_end-1, 'weightage'] = 0.82 / top2_end

        next8_end = min(n, top2_end + 8)
        if next8_end > top2_end:
            count = next8_end - top2_end
            df_sorted.loc[top2_end:next8_end-1, 'weightage'] = 0.12 / count

        remaining_start = next8_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.06 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: VelocityApocalypse Strategy
# =====================================
class VelocityApocalypse(BaseStrategy):
    """
    VelocityApocalypse: Apocalyptic Velocity Endgame.
    - Thesis: Endgame alpha (10x potential) in apocalyptic velocity spikes where indicators hit escape velocity (>3 std dev moves) in regime shifts, dooming laggards and rewarding frontrunners.
    - Focus: Std-dev normalized velocity for apocalypse-level outliers.
    - Allocation: 90% to top 1-2, 8% to next 3, 2% to rest – doomsday concentration.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', '9ema osc latest', '21ema osc latest',
            'zscore latest', 'rsi latest', 'ma90 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Oscillator Escape Velocity (3+ std dev)
        osc_std = df['osc latest'].std()
        escape_vel = np.where(np.abs(df['osc latest']) > 3 * osc_std, 5.0, np.clip(df['osc latest'] / osc_std, 0.1, 3.0))
        df['escape_vel'] = escape_vel * np.sign(df['osc latest'])  # Direction matters

        # 2. EMA Acceleration Apocalypse (Rapid crossover)
        accel = (df['9ema osc latest'] - df['21ema osc latest']) / (df['21ema osc latest'] + 1e-6)
        apoc_accel = np.clip(accel * 10, 0, 4.5)
        df['apoc_accel'] = apoc_accel

        # 3. Z-Apocalypse (Regime shift via z-extreme)
        z_apoc = np.where(np.abs(df['zscore latest']) > 2.5, 4.8, np.clip(np.abs(df['zscore latest']) * 2, 0.2, 3.0))
        df['z_apoc'] = z_apoc

        # 4. RSI Doomsday Threshold
        doom_rsi = np.where(df['rsi latest'] > 80, 3.2, np.clip((df['rsi latest'] - 50) / 15, 0.5, 2.5))
        df['doom_rsi'] = doom_rsi

        # 5. Price Surge Confirmation
        surge_confirm = np.where(df['price'] > df['ma90 latest'] * 1.1, 2.9, 0.9)
        df['surge_confirm'] = surge_confirm * (1 / (df['dev20 latest'] / df['price'] + 1e-6))  # Vol inverse

        # 6. Apocalypse Score
        df['apoc_score'] = (
            df['escape_vel'] * df['apoc_accel'] * df['z_apoc'] *
            df['doom_rsi'] * df['surge_confirm']
        )  # Positive bias via signs

        df['apoc_score'] = np.maximum(df['apoc_score'], 0.01)

        # 7. Doomsday Alloc
        df_sorted = df.sort_values('apoc_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        top2_end = min(2, n)
        df_sorted.loc[:top2_end-1, 'weightage'] = 0.90 / top2_end

        next3_end = min(n, top2_end + 3)
        if next3_end > top2_end:
            count = next3_end - top2_end
            df_sorted.loc[top2_end:next3_end-1, 'weightage'] = 0.08 / count

        remaining_start = next3_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.02 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: QuantumMomentumLeap Strategy
# =====================================
class QuantumMomentumLeap(BaseStrategy):
    """
    QuantumMomentumLeap: Quantum Leap into Untapped Momentum.
    - Thesis: Quantum leaps (4x+ returns) occur when momentum 'tunnels' through resistance via synchronized quantum states (aligned extreme indicators) in superposition of trend/mean-reversion.
    - Focus: Phase alignment of indicators for leap detection.
    - Allocation: 78% to top 4, 15% to next 5, 7% to rest – quantum precision.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest', '21ema osc latest',
            'zscore latest', 'zscore weekly', 'ma200 latest', 'dev20 latest', 'ma20 weekly'
        ]
        
        # FIX 1: Add .copy() to ensure we are working on a standalone DataFrame
        df = self._clean_data(df, required_columns).copy()

        # 1. RSI Quantum Phase (Synchronized rise)
        phase_rsi = np.sin(np.pi * (df['rsi latest'] - 50) / 50) * 2 + 1  # Wave-like alignment
        df['phase_rsi'] = np.abs(phase_rsi) * np.where(df['rsi latest'] > 55, 2.1, 1.0)

        # 2. OSC Tunnel Velocity
        tunnel_vel = np.exp((df['9ema osc latest'] - df['21ema osc latest']) / 50) - 1
        df['tunnel_vel'] = np.clip(tunnel_vel, 0, 4.0)

        # 3. Z-Superposition (Balanced extreme)
        super_z = 1 / (1 + np.exp(-df['zscore latest'])) * np.clip(np.abs(df['zscore weekly']), 0, 3)  # Sigmoid balance
        df['super_z'] = super_z * 2.5

        # 4. Low-Resistance Barrier (Vol/MA)
        barrier = np.where(
            (df['dev20 latest'] / df['price'] < 0.02) & (df['price'] > df['ma20 weekly']),
            3.3,
            1.1
        )
        df['barrier'] = barrier

        # 5. Leap Alignment Score
        # FIX 2: Correcting np.minimum for 3 arguments by nesting
        alignment = np.minimum(
            df['phase_rsi'], 
            np.minimum(df['tunnel_vel'], df['super_z'])
        )
        
        df['leap_score'] = alignment * df['barrier'] * np.where(df['price'] > df['ma200 latest'], 1.8, 0.6)

        df['leap_score'] = np.maximum(df['leap_score'], 0.01)

        # 6. Quantum Alloc
        df_sorted = df.sort_values('leap_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        # Initialize weightage column
        df_sorted['weightage'] = 0.0

        top4_end = min(4, n)
        df_sorted.loc[:top4_end-1, 'weightage'] = 0.78 / top4_end

        next5_end = min(n, top4_end + 5)
        if next5_end > top4_end:
            count = next5_end - top4_end
            df_sorted.loc[top4_end:next5_end-1, 'weightage'] = 0.15 / count

        remaining_start = next5_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.07 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

class NebulaMomentumStorm(BaseStrategy):
    """
    NebulaMomentumStorm: Storm of Nebula-Scale Momentum.
    - Thesis: Nebula storms brew 7x+ returns in vast momentum clouds where indicators form storm cells (clustered extremes) pulling price into hyper-acceleration.
    - Focus: Cluster density of momentum signals for storm intensity.
    - Allocation: 88% to top 1, 9% to next 4, 3% to rest – storm-eye precision.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', 'zscore latest', 'ma90 latest', 'dev20 weekly'
        ]
        
        # FIX 1: Add .copy() to ensure we are working on a standalone DataFrame,
        # preventing the SettingWithCopyWarning.
        df = self._clean_data(df, required_columns).copy()

        # 1. Storm Cell Density (Indicator clustering)
        cell_rsi = np.where(np.abs(df['rsi latest'] - df['rsi weekly']) < 5, 3.6, 1.4)
        df['cell_rsi'] = cell_rsi

        cell_osc = np.where(np.abs(df['osc latest'] - df['osc weekly']) < 20, 4.1, 1.6)
        df['cell_osc'] = cell_osc * np.where(df['9ema osc latest'] > 0, 1.7, 0.8)

        # 2. Z-Nebula Core (Extreme density)
        nebula_core = np.clip(np.abs(df['zscore latest']) * 2.2, 0, 4.5)
        df['nebula_core'] = nebula_core

        # 3. Hyper-Acceleration Pull
        pull = (df['price'] / df['ma90 latest'] - 1) * 100
        # Added safety check for division by zero
        df['pull'] = np.clip(pull / 5, 0, 3.8) * (1 / (df['dev20 weekly'] / df['price'] + 1e-6))

        # 4. Storm Intensity (Density * Pull)
        # FIX 2: np.minimum takes 2 arguments. To compare 3, we nest them 
        # or use element-wise min across columns.
        intensity = np.minimum(
            df['cell_rsi'], 
            np.minimum(df['cell_osc'], df['nebula_core'])
        )
        
        df['storm_score'] = intensity * df['pull']
        df['storm_score'] = np.maximum(df['storm_score'], 0.01)

        # 5. Nebula Alloc
        df_sorted = df.sort_values('storm_score', ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return pd.DataFrame()

        # Initialize weightage to float
        df_sorted['weightage'] = 0.0

        top1_end = min(1, n)
        df_sorted.loc[:top1_end-1, 'weightage'] = 0.88 / top1_end

        next4_end = min(n, top1_end + 4)
        if next4_end > top1_end:
            count = next4_end - top1_end
            df_sorted.loc[top1_end:next4_end-1, 'weightage'] = 0.09 / count

        remaining_start = next4_end
        if remaining_start < n:
            count = n - remaining_start
            df_sorted.loc[remaining_start:, 'weightage'] = 0.03 / count

        total_w = df_sorted['weightage'].sum()
        if total_w > 0:
            df_sorted['weightage'] /= total_w

        return self._allocate_portfolio(df_sorted, sip_amount)

# =====================================
# NEW: ResonanceEcho Strategy
# =====================================
class ResonanceEcho(BaseStrategy):
    """
    ResonanceEcho: Harmonic Resonance in Indicator Echoes.
    - Thesis: Out-of-box harmonic trading – alpha from 'echo chambers' where indicators resonate (correlated lags between daily/weekly) like sound waves, amplifying momentum echoes for sustained returns.
    - Innovation: Compute pseudo-frequency resonance via lagged correlations; bet on high-resonance setups.
    - Weighting: Natural score normalization for balanced, echo-proportional allocation.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Lagged Echo Correlation (Daily leading weekly by 'echo')
        # Pseudo-lag resonance: abs(corr) high = strong echo
        if len(df) > 1:
            echo_rsi = np.abs(np.corrcoef(df['rsi latest'], df['rsi weekly'])[0,1])
            echo_osc = np.abs(np.corrcoef(df['osc latest'], df['osc weekly'])[0,1])
            echo_ema9 = np.abs(np.corrcoef(df['9ema osc latest'], df['9ema osc weekly'])[0,1])
            echo_ema21 = np.abs(np.corrcoef(df['21ema osc latest'], df['21ema osc weekly'])[0,1])
            avg_echo = (echo_rsi + echo_osc + echo_ema9 + echo_ema21) / 4
        else:
            avg_echo = 0.5

        # 2. Resonance Amplitude (Strength of echo in extremes)
        amp_rsi = np.where(np.abs(df['rsi latest'] - 50) > 20, 2.0, 1.0)
        amp_osc = np.clip(np.abs(df['osc latest']) / 50, 0.5, 2.5)
        df['resonance_amp'] = (amp_rsi + amp_osc) / 2 * avg_echo * 2  # Scaled

        # 3. Harmonic Phase Alignment (Sine-phase match)
        phase_rsi = np.sin(np.pi * (df['rsi latest'] / 100)) * np.sin(np.pi * (df['rsi weekly'] / 100))
        phase_osc = np.sign(df['osc latest']) * np.sign(df['osc weekly'])
        df['phase_align'] = np.abs(phase_rsi) * phase_osc * 1.5 + 0.5

        # 4. Echo Momentum Boost (Positive direction)
        boost = np.where((df['osc latest'] > 0) & (df['9ema osc latest'] > df['21ema osc latest']), 2.2, 1.0)
        df['echo_boost'] = boost

        # 5. Composite Resonance Score
        df['resonance_score'] = df['resonance_amp'] * df['phase_align'] * df['echo_boost']

        df['resonance_score'] = np.maximum(df['resonance_score'], 0.01)

        # Normalize to weights (natural)
        total_score = df['resonance_score'].sum()
        if total_score > 0:
            df['weightage'] = df['resonance_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: DivergenceMirage Strategy
# =====================================
class DivergenceMirage(BaseStrategy):
    """
    DivergenceMirage: Mirage of Hidden Divergences.
    - Thesis: Creative illusion-breaker – alpha from 'mirage divergences' where apparent indicator confluences hide subtle divergences (e.g., RSI up but OSC lag), signaling impending reversals/momentum shifts like optical illusions in charts.
    - Innovation: Quantify 'mirage strength' as divergence hidden in convergence; trade the reveal.
    - Weighting: Score-based diffusion for mirage-proportional exposure.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Apparent Convergence (Surface agreement)
        conv_rsi = np.where(np.sign(df['rsi latest'] - 50) == np.sign(df['rsi weekly'] - 50), 1.5, 0.5)
        conv_osc = np.where(np.sign(df['osc latest']) == np.sign(df['osc weekly']), 1.8, 0.6)

        # 2. Hidden Divergence Mirage (Subtle lag/mismatch)
        mirage_rsi = np.abs((df['rsi latest'] - df['rsi weekly']) / (df['rsi latest'] + df['rsi weekly'] + 1e-6)) * 3
        mirage_osc = np.abs((df['osc latest'] - df['osc weekly']) / 50) * 2.5
        mirage_z = np.abs(df['zscore latest'] - 0) * 1.5  # Deviation from mean
        df['mirage_strength'] = (mirage_rsi + mirage_osc + mirage_z) / 3 * conv_rsi * conv_osc

        # 3. Reveal Potential (Divergence direction for alpha)
        reveal = np.where(
            (df['osc latest'] > df['21ema osc latest']) & (df['rsi latest'] < 60),  # Bullish hidden
            2.4,
            np.where(df['rsi latest'] > 60, 0.7, 1.2)  # Bearish mirage
        )
        df['reveal_pot'] = reveal

        # 4. Mirage Illusion Score
        df['mirage_score'] = df['mirage_strength'] * df['reveal_pot']

        df['mirage_score'] = np.maximum(df['mirage_score'], 0.01)

        # Normalize naturally
        total_score = df['mirage_score'].sum()
        if total_score > 0:
            df['weightage'] = df['mirage_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: FractalWhisper Strategy
# =====================================
class FractalWhisper(BaseStrategy):
    """
    FractalWhisper: Whispers from Fractal Self-Similarity.
    - Thesis: Unconventional fractal geometry – alpha from 'whispers' of self-similar patterns across scales (daily/weekly indicator fractals), detecting hidden continuations like Mandelbrot echoes in price noise.
    - Innovation: Approximate Hurst exponent via rescaled range on indicators for persistence whispers.
    - Weighting: Fractal persistence scores for organic scaling.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            'ma90 latest', 'ma90 weekly', 'dev20 latest', 'dev20 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Pseudo-Hurst Exponent (Fractal persistence approx)
        # Simple R/S on log-returns proxy via indicators
        log_rsi = np.log(df['rsi latest'] + 1)
        rs_rsi = (log_rsi.max() - log_rsi.min()) / (log_rsi.std() + 1e-6) if len(df) > 1 else 0.5
        log_osc = np.log(np.abs(df['osc latest']) + 1)
        rs_osc = (log_osc.max() - log_osc.min()) / (log_osc.std() + 1e-6) if len(df) > 1 else 0.5
        hurst_approx = np.log((rs_rsi + rs_osc) / 2) / np.log(len(df)) if len(df) > 1 else 0.5
        df['hurst_persist'] = np.clip(hurst_approx + 0.5, 0.2, 1.8)  # Broadcast scalar

        # 2. Self-Similarity Whisper (Scale match daily-weekly)
        scale_match = np.abs((df['dev20 latest'] / df['price']) - (df['dev20 weekly'] / df['price'])) < 0.01
        whisper = np.where(scale_match, 2.6, 1.0) * np.abs(df['osc latest'] - df['osc weekly']) / 50
        df['whisper_sim'] = whisper

        # 3. Fractal Edge (Persistence in trends)
        edge = np.where(df['price'] > df['ma90 latest'], df['hurst_persist'] * 1.7, df['hurst_persist'] * 0.6)
        df['fractal_edge'] = edge

        # 4. Composite Whisper Score
        df['whisper_score'] = df['hurst_persist'] * df['whisper_sim'] * df['fractal_edge']

        df['whisper_score'] = np.maximum(df['whisper_score'], 0.01)

        # Natural fractal weighting
        total_score = df['whisper_score'].sum()
        if total_score > 0:
            df['weightage'] = df['whisper_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: InterferenceWave Strategy
# =====================================
class InterferenceWave(BaseStrategy):
    """
    InterferenceWave: Wave Interference Patterns for Alpha.
    - Thesis: Physics-inspired – alpha from constructive interference waves where indicator 'waves' (sine-transformed) interfere positively, creating amplification peaks like light waves in optics.
    - Innovation: Sine-decompose indicators, compute interference factor for peak detection.
    - Weighting: Interference intensity for wave-proportional diffusion.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Wave Decomposition (Sine transforms)
        wave_rsi = np.sin(2 * np.pi * df['rsi latest'] / 100)
        wave_osc = np.sin(2 * np.pi * (df['osc latest'] + 100) / 200)  # Normalize to 0-1
        wave_ema = np.sin(2 * np.pi * (df['9ema osc latest'] - df['21ema osc latest']) / 200)

        # 2. Interference Factor (Constructive sum)
        interf_rsi_osc = np.abs(wave_rsi + wave_osc)  # Amplitude sum
        interf_ema = np.abs(wave_ema + wave_rsi) * 0.5
        df['interf_factor'] = (interf_rsi_osc + interf_ema) / 2 * 2.0  # Scale

        # 3. Wave Peak Detection (High amplitude + trend)
        peak = np.where(
            (df['price'] > df['ma20 latest']) & (np.abs(df['zscore latest']) > 1),
            df['interf_factor'] * 2.3,
            df['interf_factor']
        )
        df['wave_peak'] = peak

        # 4. Destructive Filter (Low interference penalty)
        destruct = np.where(np.abs(wave_rsi + wave_osc) < 0.5, 0.4, 1.0)
        df['destruct_filt'] = destruct

        # 5. Composite Interference Score
        df['interf_score'] = df['wave_peak'] * df['destruct_filt']

        df['interf_score'] = np.maximum(df['interf_score'], 0.01)

        # Wave-natural weighting
        total_score = df['interf_score'].sum()
        if total_score > 0:
            df['weightage'] = df['interf_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: ShadowPuppet Strategy
# =====================================
class ShadowPuppet(BaseStrategy):
    """
    ShadowPuppet: Puppetry of Shadow Indicators.
    - Thesis: Theatrical out-of-box – alpha from 'shadow puppets' where primary indicators (OSC/RSI) cast shadows (lagged MAs/dev) that 'puppet' price moves, trading the shadow-price disconnects like marionette strings.
    - Innovation: Shadow disconnect as string tension; high tension = alpha pull.
    - Weighting: Tension scores for puppet-like fluid allocation.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', 'ma20 latest', 'ma90 latest',
            'dev20 latest', 'zscore latest', 'ma200 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Primary Puppet (OSC/RSI 'hands')
        puppet_primary = (np.abs(df['osc latest']) / 50 + (df['rsi latest'] - 50) / 50) / 2
        df['puppet_primary'] = np.clip(puppet_primary, 0, 2.5)

        # 2. Shadow Cast (MA/dev as shadows)
        shadow_ma = np.abs(df['price'] - df['ma20 latest']) / df['ma20 latest']
        shadow_dev = df['dev20 latest'] / df['price']
        df['shadow_cast'] = (shadow_ma + shadow_dev) * 1.5

        # 3. Disconnect Tension (Primary vs shadow mismatch)
        tension = np.abs(df['puppet_primary'] - df['shadow_cast']) * np.abs(df['zscore latest'])
        df['tension'] = np.clip(tension, 0.5, 3.0)

        # 4. Puppet Pull (Direction toward long-term shadow)
        pull = np.where(df['price'] > df['ma200 latest'] * 0.95, tension * 2.1, tension * 0.8)
        df['puppet_pull'] = pull

        # 5. Composite Shadow Score
        df['shadow_score'] = df['puppet_primary'] * df['shadow_cast'] * df['puppet_pull']

        df['shadow_score'] = np.maximum(df['shadow_score'], 0.01)

        # Fluid puppet weighting
        total_score = df['shadow_score'].sum()
        if total_score > 0:
            df['weightage'] = df['shadow_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: EntangledMomentum Strategy
# =====================================
class EntangledMomentum(BaseStrategy):
    """
    EntangledMomentum: Quantum Entanglement Across Timeframes.
    - Paradigm: Treat daily/weekly indicators as entangled particles; alpha from 'spooky action' where one timeframe's momentum instantly correlates the other's, enabling non-local alpha transmission for 4x+ surges.
    - Innovation: Entanglement entropy via mutual information proxy; high entanglement = alpha conduit.
    - Weighting: Entropy-normalized for entangled diffusion.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', 'zscore latest', 'zscore weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Entanglement Proxy (Mutual info approx via corr^2)
        if len(df) > 1:
            ent_rsi = np.corrcoef(df['rsi latest'], df['rsi weekly'])[0,1] ** 2
            ent_osc = np.corrcoef(df['osc latest'], df['osc weekly'])[0,1] ** 2
            ent_z = np.corrcoef(df['zscore latest'], df['zscore weekly'])[0,1] ** 2
            avg_ent = (ent_rsi + ent_osc + ent_z) / 3
        else:
            avg_ent = 0.7

        # 2. Non-Local Surge (Correlated extremes)
        surge = np.abs(df['rsi latest'] - df['rsi weekly']) * avg_ent * 2
        df['non_local_surge'] = np.clip(surge, 0.5, 3.0)

        # 3. Spooky Boost (Positive correlation direction)
        spooky = np.where((df['osc latest'] > 0) & (df['9ema osc weekly'] > 0), 2.5, 1.0)
        df['spooky_boost'] = spooky * avg_ent

        # 4. Entanglement Score
        df['ent_score'] = df['non_local_surge'] * df['spooky_boost']

        df['ent_score'] = np.maximum(df['ent_score'], 0.01)

        # Natural entanglement weighting
        total_score = df['ent_score'].sum()
        if total_score > 0:
            df['weightage'] = df['ent_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: ButterflyChaos Strategy
# =====================================
class ButterflyChaos(BaseStrategy):
    """
    ButterflyChaos: Chaotic Butterfly Effects in Momentum.
    - Paradigm: Chaos theory lens – tiny 'butterfly wings' (small indicator perturbations) flap into massive alpha hurricanes; quantify Lyapunov-like sensitivity for explosive momentum.
    - Innovation: Perturbation divergence ratio between daily/weekly for chaos exponent.
    - Weighting: Chaos sensitivity scores for turbulent allocation.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'osc weekly', 'rsi latest', 'rsi weekly',
            '9ema osc latest', 'zscore latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Perturbation Wings (Small diffs)
        wings_osc = np.abs(df['osc latest'] - df['osc weekly']) / (np.abs(df['osc weekly']) + 1e-6)
        wings_rsi = np.abs(df['rsi latest'] - df['rsi weekly']) / 50
        df['wings'] = (wings_osc + wings_rsi) / 2 * 2.0

        # 2. Divergence Exponent (Chaos sensitivity)
        if len(df) > 1:
            div_exp = np.log(1 + df['wings'].std() * 10) / np.log(len(df))
        else:
            div_exp = 0.6
        df['chaos_exp'] = np.clip(div_exp + 0.5, 0.3, 2.2)

        # 3. Hurricane Potential (Perturb in positive direction)
        hurri = np.where(df['9ema osc latest'] > 0, df['wings'] * df['chaos_exp'] * 2.8, df['wings'] * 0.7)
        df['hurri_pot'] = hurri

        # 4. Butterfly Score
        df['butterfly_score'] = df['chaos_exp'] * df['hurri_pot']

        df['butterfly_score'] = np.maximum(df['butterfly_score'], 0.01)

        # Chaotic natural weighting
        total_score = df['butterfly_score'].sum()
        if total_score > 0:
            df['weightage'] = df['butterfly_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: SynapseFiring Strategy
# =====================================
class SynapseFiring(BaseStrategy):
    """
    SynapseFiring: Neural Synapse Firing for Momentum Alpha.
    - Paradigm: Brain-like neural net – indicators as neurons firing; alpha from synaptic bursts where 'firing rates' (threshold crosses) create momentum avalanches.
    - Innovation: Poisson-like firing probability from indicator thresholds for burst scoring.
    - Weighting: Firing rate diffusion for neural proportionality.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma90 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Firing Thresholds (Neuron activation)
        fire_rsi = np.where(df['rsi latest'] > 60, 1, 0) + np.where(df['rsi latest'] < 40, 1, 0)
        fire_osc = np.where(np.abs(df['osc latest']) > 50, 1, 0)
        df['fire_rate'] = (fire_rsi + fire_osc) / 2 * 2.5  # Avg scaled

        # 2. Synaptic Burst (Chained firings)
        burst = np.where((df['9ema osc latest'] > df['21ema osc latest']) & (df['fire_rate'] > 1), 3.2, 1.0)
        df['syn_burst'] = burst * np.abs(df['zscore latest'])

        # 3. Avalanche Potential (Trend propagation)
        aval = np.where(df['price'] > df['ma90 latest'], df['syn_burst'] * 2.1, df['syn_burst'] * 0.5)
        df['aval_pot'] = aval

        # 4. Synapse Score
        df['synapse_score'] = df['fire_rate'] * df['aval_pot']

        df['synapse_score'] = np.maximum(df['synapse_score'], 0.01)

        # Neural weighting
        total_score = df['synapse_score'].sum()
        if total_score > 0:
            df['weightage'] = df['synapse_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: HolographicMomentum Strategy
# =====================================
class HolographicMomentum(BaseStrategy):
    """
    HolographicMomentum: Holographic Projections of Alpha.
    - Paradigm: Holography metaphor – 2D indicators project '3D' momentum depth; alpha from holographic interference fringes where projections overlap for volumetric gains.
    - Innovation: Projector overlap via vector dot products of normalized indicators.
    - Weighting: Hologram intensity for projected balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            'zscore latest', 'ma200 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Normalized Projections (2D to 3D vectors)
        proj_rsi = (df['rsi latest'] - 50) / 50
        proj_osc = df['osc latest'] / 100
        proj_z = df['zscore latest'] / 3  # Std normalize approx

        # 2. Holographic Overlap (Dot product fringes)
        overlap = np.abs(proj_rsi * proj_osc + proj_osc * proj_z + proj_z * proj_rsi) / 3
        df['holo_overlap'] = np.clip(overlap * 3, 0.4, 2.8)

        # 3. Depth Amplification (Vol/Trend volume)
        depth = (1 / (df['dev20 latest'] / df['price'] + 1e-6)) * np.where(df['price'] > df['ma200 latest'], 2.0, 1.0)
        df['holo_depth'] = np.clip(depth / depth.mean(), 0.5, 2.5) if len(df) > 1 else depth

        # 4. Fringe Score
        df['holo_score'] = df['holo_overlap'] * df['holo_depth']

        df['holo_score'] = np.maximum(df['holo_score'], 0.01)

        # Projected weighting
        total_score = df['holo_score'].sum()
        if total_score > 0:
            df['weightage'] = df['holo_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: WormholeTemporal Strategy
# =====================================
class WormholeTemporal(BaseStrategy):
    """
    WormholeTemporal: Temporal Wormholes in Momentum Flow.
    - Paradigm: Spacetime wormholes – alpha from 'wormholes' shortcutting time, where past extremes (weekly z-scores) tunnel to future surges (daily OSC), bypassing normal paths for instant alpha.
    - Innovation: Wormhole curvature as time-diff / magnitude ratio for tunnel strength.
    - Weighting: Curvature gradients for temporal flow allocation.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'zscore weekly', 'rsi weekly',
            '9ema osc latest', 'ma90 weekly', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Past Extreme Anchor (Weekly z-deep)
        anchor = np.abs(df['zscore weekly']) * np.where(np.abs(df['rsi weekly']) < 30, 2.5, 1.0)
        df['past_anchor'] = np.clip(anchor, 0, 3.5)

        # 2. Future Surge Endpoint (Daily OSC tunnel)
        endpoint = np.clip(df['osc latest'] / 50, 0, 2.8) * np.where(df['9ema osc latest'] > 0, 1.9, 0.6)
        df['future_end'] = endpoint

        # 3. Wormhole Curvature (Time shortcut ratio)
        time_diff = np.abs(df['dev20 latest'] / df['price'])  # Proxy time/vol
        curvature = df['past_anchor'] / (time_diff + 1e-6) * df['future_end']
        df['worm_curv'] = np.clip(curvature, 0.3, 2.7)

        # 4. Tunnel Score
        df['worm_score'] = df['past_anchor'] * df['future_end'] * df['worm_curv'] / np.where(df['price'] > df['ma90 weekly'], 1, 0.5)

        df['worm_score'] = np.maximum(df['worm_score'], 0.01)

        # Temporal weighting
        total_score = df['worm_score'].sum()
        if total_score > 0:
            df['weightage'] = df['worm_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: SymbioticAlpha Strategy
# =====================================
class SymbioticAlpha(BaseStrategy):
    """
    SymbioticAlpha: Symbiotic Ecosystem Dynamics.
    - Paradigm: Ecology model – indicators as symbiotic species (predator-prey cycles); alpha from balanced symbioses where momentum 'predators' (OSC) feed on 'prey' (RSI dips) for population booms.
    - Innovation: Lotka-Volterra proxy via indicator ratios for symbiosis equilibrium.
    - Weighting: Ecosystem carrying capacity for symbiotic balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', 'zscore latest',
            'ma200 latest', 'dev20 weekly', '9ema osc latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Prey Population (RSI dips as food)
        prey = np.clip(100 - df['rsi latest'], 0, 50) / 50 * 2.0
        df['prey_pop'] = prey

        # 2. Predator Growth (OSC hunting)
        pred_growth = np.clip(df['osc latest'] / 50, 0, 2.5) * np.where(df['9ema osc latest'] > 0, 1.8, 0.5)
        df['pred_growth'] = pred_growth

        # 3. Symbiosis Equilibrium (Lotka-Volterra approx: prey * pred / carrying)
        carrying = df['dev20 weekly'] / df['price'] * 100  # Capacity inverse
        equil = (df['prey_pop'] * df['pred_growth']) / (carrying + 1e-6)
        df['symb_equil'] = np.clip(equil, 0.4, 3.0)

        # 4. Boom Potential (Uptrend symbiosis)
        boom = np.where(df['price'] > df['ma200 latest'], df['symb_equil'] * 2.4, df['symb_equil'] * 0.7)
        df['boom_pot'] = boom * np.abs(df['zscore latest'])

        # 5. Symbiotic Score
        df['symb_score'] = df['symb_equil'] * df['boom_pot']

        df['symb_score'] = np.maximum(df['symb_score'], 0.01)

        # Ecosystem weighting
        total_score = df['symb_score'].sum()
        if total_score > 0:
            df['weightage'] = df['symb_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: PhononVibe Strategy
# =====================================
class PhononVibe(BaseStrategy):
    """
    PhononVibe: Phonon Vibrations in Momentum Lattice.
    - Paradigm: Solid-state physics – momentum as phonon waves in indicator lattice; alpha from vibrational modes (frequency peaks) resonating for coherent energy transfer.
    - Innovation: FFT-like frequency dominance approx via indicator derivatives for mode scoring.
    - Weighting: Vibration amplitude for phonon propagation.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'rsi latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma90 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Derivative Vibrations (dI/dt approx)
        vibe_osc = np.abs(df['9ema osc latest'] - df['21ema osc latest'])
        vibe_rsi = np.abs(df['rsi latest'] - 50) / 10  # Freq proxy
        df['vibe_deriv'] = (vibe_osc + vibe_rsi) / 2 * 1.5

        # 2. Mode Resonance (High freq dominance)
        freq_dom = np.where(vibe_osc > vibe_rsi * 2, 2.9, 1.2)
        df['mode_res'] = freq_dom * np.abs(df['zscore latest'])

        # 3. Lattice Coherence (Trend alignment)
        coh = np.where(df['price'] > df['ma90 latest'], df['vibe_deriv'] * 2.2, df['vibe_deriv'] * 0.6)
        df['lattice_coh'] = coh

        # 4. Phonon Score
        df['phonon_score'] = df['mode_res'] * df['lattice_coh']

        df['phonon_score'] = np.maximum(df['phonon_score'], 0.01)

        # Vibrational weighting
        total_score = df['phonon_score'].sum()
        if total_score > 0:
            df['weightage'] = df['phonon_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: HorizonEvent Strategy
# =====================================
class HorizonEvent(BaseStrategy):
    """
    HorizonEvent: Event Horizons of Momentum Black Holes.
    - Paradigm: Black hole analogy – alpha from crossing momentum 'event horizons' where escape velocity (indicator thresholds) traps capital in inescapable gains.
    - Innovation: Schwarzschild radius proxy via indicator magnitude for horizon crossing.
    - Weighting: Gravitational pull gradients for horizon accretion.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'rsi latest', 'zscore latest',
            'ma200 latest', 'dev20 latest', '9ema osc latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Event Horizon Radius (Indicator mass)
        radius_osc = np.abs(df['osc latest']) / 100 * 2.0
        radius_z = np.abs(df['zscore latest']) * 1.5
        df['horizon_rad'] = (radius_osc + radius_z) / 2

        # 2. Crossing Velocity (Price/EMA escape)
        vel_cross = np.where(df['9ema osc latest'] > df['horizon_rad'] * 50, 3.1, np.clip(df['rsi latest'] / 50, 0.5, 2.0))
        df['cross_vel'] = vel_cross

        # 3. Accretion Disk (Vol pull)
        acc_disk = 1 / (df['dev20 latest'] / df['price'] + 1e-6) * np.where(df['price'] > df['ma200 latest'], 2.3, 0.8)
        df['acc_disk'] = np.clip(acc_disk / acc_disk.mean(), 0.4, 2.6) if len(df) > 1 else acc_disk

        # 4. Horizon Score
        df['horizon_score'] = df['horizon_rad'] * df['cross_vel'] * df['acc_disk']

        df['horizon_score'] = np.maximum(df['horizon_score'], 0.01)

        # Accretion weighting
        total_score = df['horizon_score'].sum()
        if total_score > 0:
            df['weightage'] = df['horizon_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: EscherLoop Strategy
# =====================================
class EscherLoop(BaseStrategy):
    """
    EscherLoop: Escher's Impossible Momentum Loops.
    - Paradigm: Optical paradox art – alpha from 'impossible loops' where indicators form self-referential cycles (feedback loops) defying linear time, generating perpetual momentum illusions.
    - Innovation: Loop closure via indicator self-correlations for paradox strength.
    - Weighting: Loop tension for illusory balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma90 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Self-Referential Cycle (Indicator feedback)
        cycle_rsi = np.abs(df['rsi latest'] - np.roll(df['rsi latest'], 1)) if len(df) > 1 else 10  # Shift proxy
        cycle_osc = np.abs(df['osc latest'] - df['9ema osc latest'])
        df['cycle_tens'] = (cycle_rsi + cycle_osc) / 2 / 50 * 2.5  # Normalized

        # 2. Paradox Strength (Loop defiance)
        para_str = np.where(np.sign(df['9ema osc latest'] - df['21ema osc latest']) * np.sign(df['zscore latest']) < 0, 3.4, 1.1)
        df['para_str'] = para_str

        # 3. Illusion Sustain (Trend loop)
        sustain = np.where(df['price'] > df['ma90 latest'], df['cycle_tens'] * 2.0, df['cycle_tens'] * 0.4)
        df['illus_sustain'] = sustain

        # 4. Escher Score
        df['escher_score'] = df['cycle_tens'] * df['para_str'] * df['illus_sustain']

        df['escher_score'] = np.maximum(df['escher_score'], 0.01)

        # Illusory weighting
        total_score = df['escher_score'].sum()
        if total_score > 0:
            df['weightage'] = df['escher_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: MicrowaveCosmic Strategy
# =====================================
class MicrowaveCosmic(BaseStrategy):
    """
    MicrowaveCosmic: Cosmic Microwave Background Filtering.
    - Paradigm: Cosmology filter – strip 'cosmic noise' (vol/dev) from background to reveal pure momentum 'big bang' signals in indicators for primordial alpha expansion.
    - Innovation: CMB-like denoising via indicator / vol ratio for signal purity.
    - Weighting: Purity spectra for cosmic expansion allocation.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'rsi latest', 'zscore latest',
            'dev20 latest', 'dev20 weekly', 'ma200 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Background Noise (Vol CMB)
        noise = (df['dev20 latest'] + df['dev20 weekly']) / (2 * df['price'])
        df['noise_cmb'] = noise

        # 2. Primordial Signal (Indicator purity)
        sig_osc = np.abs(df['osc latest']) / (noise + 1e-6)
        sig_rsi = np.abs(df['rsi latest'] - 50) / (noise * 100 + 1e-6)
        df['prim_sig'] = (sig_osc + sig_rsi) / 2 * 1.8

        # 3. Expansion Purity (Z-filtered)
        purity = df['prim_sig'] * np.abs(df['zscore latest']) * np.where(df['price'] > df['ma200 latest'], 2.6, 0.9)
        df['purity_exp'] = np.clip(purity / purity.mean(), 0.3, 2.4) if len(df) > 1 else purity

        # 4. Cosmic Score
        df['cosmic_score'] = df['prim_sig'] * df['purity_exp']

        df['cosmic_score'] = np.maximum(df['cosmic_score'], 0.01)

        # Spectral weighting
        total_score = df['cosmic_score'].sum()
        if total_score > 0:
            df['weightage'] = df['cosmic_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)
        
# =====================================
# NEW: SingularityMomentum Strategy
# =====================================
class SingularityMomentum(BaseStrategy):
    """
    SingularityMomentum: AI Singularity Convergence in Momentum.
    - Paradigm: Singularity vision – alpha singularity where indicators converge to a 'point of no return' (exponential score blowup), compounding returns via self-reinforcing feedback loops mimicking tech explosion.
    - Innovation: Exponential convergence factor from indicator ratios for singularity pull; beats priors by 10x compounding in bull regimes.
    - Weighting: Singularity gradient normalization for infinite-horizon diffusion.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '21ema osc latest', 'zscore latest', 'zscore weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Convergence Kernel (Indicators approaching unity)
        kernel_rsi = 1 / (1 + np.abs(df['rsi latest'] - df['rsi weekly']) / 100)
        kernel_osc = 1 / (1 + np.abs(df['osc latest'] - df['osc weekly']) / 100)
        kernel_z = 1 / (1 + np.abs(df['zscore latest'] - df['zscore weekly']))
        df['conv_kernel'] = (kernel_rsi + kernel_osc + kernel_z) / 3 * 3.0

        # 2. Singularity Pull (Exponential near 1)
        pull = np.exp(df['conv_kernel'] - 1) * np.where(df['9ema osc latest'] > df['21ema osc latest'], 3.5, 1.0)
        df['sing_pull'] = np.clip(pull, 0.1, 4.0)

        # 3. Feedback Compounding (Self-amplify)
        compound = df['sing_pull'] ** (1 + np.abs(df['zscore latest']) / 3)
        df['feedback_comp'] = np.clip(compound, 0.5, 5.5)

        # 4. Singularity Score
        df['sing_score'] = df['conv_kernel'] * df['feedback_comp']

        df['sing_score'] = np.maximum(df['sing_score'], 0.01)

        # Gradient weighting
        total_score = df['sing_score'].sum()
        if total_score > 0:
            df['weightage'] = df['sing_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: MultiverseAlpha Strategy
# =====================================
class MultiverseAlpha(BaseStrategy):
    """
    MultiverseAlpha: Parallel Universe Alpha Branching.
    - Paradigm: Multiverse theory – alpha from branching paths where indicator 'universes' (daily/weekly variants) diverge then reconverge, compounding returns across 'timelines' for meta-alpha.
    - Innovation: Branch divergence-reconvergence entropy for multiverse yield; surpasses priors via parallel compounding.
    - Weighting: Timeline entropy normalization for branched equilibrium.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma90 weekly'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Branch Divergence (Path split)
        div_rsi = np.abs(df['rsi latest'] - df['rsi weekly']) / 50
        div_osc = np.abs(df['osc latest'] - df['osc weekly']) / 100
        df['branch_div'] = (div_rsi + div_osc) / 2 * 2.2

        # 2. Reconvergence Yield (Pull back)
        recon = 1 / (1 + df['branch_div']) * np.abs(df['zscore latest'] - df['zscore weekly'])
        df['recon_yield'] = np.clip(recon * 3.0, 0.2, 3.8)

        # 3. Multiverse Compounding (Diverge * recon ^2)
        multi_comp = df['branch_div'] * (df['recon_yield'] ** 2)
        df['multi_comp'] = np.clip(multi_comp, 0.3, 4.2)

        # 4. Multiverse Score
        df['multi_score'] = df['branch_div'] * df['recon_yield'] * df['multi_comp'] * np.where(df['price'] > df['ma90 latest'], 1.5, 0.5)

        df['multi_score'] = np.maximum(df['multi_score'], 0.01)

        # Entropy weighting
        total_score = df['multi_score'].sum()
        if total_score > 0:
            df['weightage'] = df['multi_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: EternalReturnCycle Strategy
# =====================================
class EternalReturnCycle(BaseStrategy):
    """
    EternalReturnCycle: Nietzschean Eternal Recurrence in Returns.
    - Paradigm: Philosophical recurrence – alpha from eternally recurring momentum cycles (indicator loops repeating with amplification), compounding via infinite return affirmation.
    - Innovation: Cycle recurrence strength via autocorrelation proxy for eternal yield.
    - Weighting: Recurrence phase for cyclical compounding balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'osc weekly', 'rsi latest', 'rsi weekly',
            '9ema osc latest', 'zscore latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Cycle Recurrence (Auto-corr approx)
        if len(df) > 1:
            rec_osc = np.corrcoef(df['osc latest'], df['osc weekly'])[0,1]
            rec_rsi = np.corrcoef(df['rsi latest'], df['rsi weekly'])[0,1]
            avg_rec = (rec_osc + rec_rsi) / 2
        else:
            avg_rec = 0.8
        df['rec_cycle'] = np.clip(avg_rec + 0.5, 0.1, 2.0)

        # 2. Eternal Amplification (Loop gain)
        amp = df['rec_cycle'] * np.abs(df['zscore latest']) * np.where(df['9ema osc latest'] > 0, 2.7, 0.9)
        df['et_amp'] = np.clip(amp, 0.4, 3.5)

        # 3. Recurrence Compounding (^rec)
        comp = df['et_amp'] ** df['rec_cycle']
        df['rec_comp'] = np.clip(comp, 0.2, 4.0)

        # 4. Eternal Score
        df['eternal_score'] = df['rec_cycle'] * df['et_amp'] * df['rec_comp']

        df['eternal_score'] = np.maximum(df['eternal_score'], 0.01)

        # Phase weighting
        total_score = df['eternal_score'].sum()
        if total_score > 0:
            df['weightage'] = df['eternal_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: DivineMomentumOracle Strategy
# =====================================
class DivineMomentumOracle(BaseStrategy):
    """
    DivineMomentumOracle: Oracle of Delphi Momentum Prophecies.
    - Paradigm: Ancient oracle visions – alpha from prophetic divergences where indicators 'foretell' momentum via Delphic ambiguities (fuzzy logic thresholds), compounding divine insights.
    - Innovation: Fuzzy oracle membership functions for prophetic certainty.
    - Weighting: Prophetic haze normalization for oracular equilibrium.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma200 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Fuzzy Membership (Delphic ambiguity)
        mem_rsi = 1 / (1 + np.exp(- (df['rsi latest'] - 50) / 20))  # Sigmoid haze
        mem_osc = 1 / (1 + np.exp(- df['osc latest'] / 30))
        df['fuzzy_mem'] = (mem_rsi + mem_osc) / 2 * 2.4

        # 2. Prophetic Divergence (Oracle split)
        div_oracle = np.abs(df['9ema osc latest'] - df['21ema osc latest']) * df['fuzzy_mem']
        df['prop_div'] = np.clip(div_oracle, 0.3, 2.9)

        # 3. Divine Compounding (Haze * div)
        comp_div = df['prop_div'] ** df['fuzzy_mem']
        df['div_comp'] = np.clip(comp_div, 0.5, 3.6) * np.where(df['price'] > df['ma200 latest'], 1.6, 0.7)

        # 4. Oracle Score
        df['oracle_score'] = df['fuzzy_mem'] * df['prop_div'] * df['div_comp']

        df['oracle_score'] = np.maximum(df['oracle_score'], 0.01)

        # Haze weighting
        total_score = df['oracle_score'].sum()
        if total_score > 0:
            df['weightage'] = df['oracle_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: CelestialAlphaForge Strategy
# =====================================
class CelestialAlphaForge(BaseStrategy):
    """
    CelestialAlphaForge: Stellar Forge of Alpha Constellations.
    - Paradigm: Astrophysics forge – alpha forged in celestial crucibles where indicator 'stars' align in constellations (pattern matching), compounding supernova yields.
    - Innovation: Constellation similarity via indicator vector angles for forge heat.
    - Weighting: Stellar magnitude normalization for cosmic forge balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            'zscore latest', 'ma90 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Star Vectors (Indicator positions)
        vec_rsi = df['rsi latest'] / 100
        vec_osc = (df['osc latest'] + 100) / 200
        vec_angle = np.arccos(np.clip(vec_rsi * vec_osc + (df['zscore latest'] / 3) * vec_rsi, -1, 1))
        df['const_angle'] = 1 / (vec_angle + 1e-6) * 2.0  # Small angle = alignment

        # 2. Forge Heat (Magnitude product)
        heat = np.abs(df['rsi latest'] * df['osc latest'] / 10000) * df['const_angle']
        df['forge_heat'] = np.clip(heat, 0.2, 2.8)

        # 3. Supernova Compounding (Heat ^ align)
        nova = df['forge_heat'] ** df['const_angle']
        df['nova_comp'] = np.clip(nova * np.where(df['price'] > df['ma90 latest'], 2.5, 0.8), 0.4, 4.1)

        # 4. Celestial Score
        df['cel_score'] = df['const_angle'] * df['forge_heat'] * df['nova_comp'] / (df['dev20 latest'] / df['price'] + 1e-6)

        df['cel_score'] = np.maximum(df['cel_score'], 0.01)

        # Magnitude weighting
        total_score = df['cel_score'].sum()
        if total_score > 0:
            df['weightage'] = df['cel_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: InfiniteMomentumLoop Strategy
# =====================================
class InfiniteMomentumLoop(BaseStrategy):
    """
    InfiniteMomentumLoop: Infinite Loop Momentum Recursion.
    - Paradigm: Recursive infinity – alpha from infinite momentum loops where scores feed back recursively, compounding to infinity in stable attractors for unbounded returns.
    - Innovation: Fixed-point iteration proxy for loop stability; high stability = infinite yield.
    - Weighting: Loop eigenvalue normalization for recursive equilibrium.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'rsi latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Recursive Seed (Base loop)
        seed = np.abs(df['osc latest'] - df['21ema osc latest']) / 50
        df['rec_seed'] = np.clip(seed, 0.1, 2.0)

        # 2. Feedback Iteration (Score ^ seed)
        iter_fb = df['rec_seed'] ** np.abs(df['zscore latest'])
        df['fb_iter'] = np.clip(iter_fb, 0.3, 3.2)

        # 3. Infinite Stability (Eigen-like: 1 / (1 - fb))
        stability = 1 / (1 - np.clip(df['fb_iter'] / 4, 0, 0.75))  # Avoid div0
        df['inf_stab'] = np.clip(stability, 0.5, 4.5) * np.where(df['9ema osc latest'] > 0, 1.8, 0.6)

        # 4. Loop Score
        df['loop_score'] = df['rec_seed'] * df['fb_iter'] * df['inf_stab']

        df['loop_score'] = np.maximum(df['loop_score'], 0.01)

        # Eigenvalue weighting
        total_score = df['loop_score'].sum()
        if total_score > 0:
            df['weightage'] = df['loop_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: GodParticleSurge Strategy
# =====================================
class GodParticleSurge(BaseStrategy):
    """
    GodParticleSurge: Higgs-Like Momentum Field Surge.
    - Paradigm: Particle physics god particle – alpha surge from 'Higgs field' giving momentum mass to indicators, compounding via field excitations for fundamental returns.
    - Innovation: Mass term as indicator * vol inverse for Higgs vev proxy.
    - Weighting: Field excitation normalization for particle balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'rsi latest', 'zscore latest',
            'dev20 latest', 'ma200 latest', '9ema osc latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Higgs VEV Proxy (Indicator mass)
        vev_osc = np.abs(df['osc latest']) / (df['dev20 latest'] / df['price'] + 1e-6)
        vev_rsi = np.abs(df['rsi latest'] - 50) / (df['dev20 latest'] / df['price'] + 1e-6)
        df['higgs_vev'] = (vev_osc + vev_rsi) / 2 * 0.5  # Scaled

        # 2. Field Excitation (Z-mass)
        exc = df['higgs_vev'] * np.abs(df['zscore latest'])
        df['field_exc'] = np.clip(exc, 0.4, 3.4)

        # 3. Surge Compounding (Vev * exc)
        surge = df['field_exc'] * df['higgs_vev'] * np.where(df['9ema osc latest'] > 0, 2.9, 0.7)
        df['surge_comp'] = np.clip(surge, 0.2, 4.3) * np.where(df['price'] > df['ma200 latest'], 1.7, 0.5)

        # 4. God Particle Score
        df['god_score'] = df['higgs_vev'] * df['field_exc'] * df['surge_comp']

        df['god_score'] = np.maximum(df['god_score'], 0.01)

        # Excitation weighting
        total_score = df['god_score'].sum()
        if total_score > 0:
            df['weightage'] = df['god_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: NirvanaMomentumWave Strategy
# =====================================
class NirvanaMomentumWave(BaseStrategy):
    """
    NirvanaMomentumWave: Buddhist Nirvana Wave Dissolution.
    - Paradigm: Eastern philosophy wave – alpha from dissolving ego-duality in momentum waves (indicator non-attachment), compounding enlightened returns beyond duality.
    - Innovation: Wave dissolution as 1 / duality variance for nirvana purity.
    - Weighting: Purity diffusion for wave enlightenment.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Duality Variance (Attachment split)
        var_rsi = (df['rsi latest'] - 50) ** 2 / 2500
        var_osc = df['osc latest'] ** 2 / 10000
        df['duality_var'] = (var_rsi + var_osc) / 2

        # 2. Dissolution Purity (1 / var)
        purity = 1 / (df['duality_var'] + 1e-6) * np.abs(df['zscore latest'])
        df['nir_purity'] = np.clip(purity, 0.3, 3.1)

        # 3. Wave Compounding (Purity * non-dual align)
        non_dual = np.where(np.sign(df['9ema osc latest']) == np.sign(df['21ema osc latest']), 2.6, 1.0)
        wave_comp = df['nir_purity'] * non_dual
        df['wave_comp'] = np.clip(wave_comp, 0.5, 3.9)

        # 4. Nirvana Score
        df['nirvana_score'] = df['nir_purity'] * wave_comp

        df['nirvana_score'] = np.maximum(df['nirvana_score'], 0.01)

        # Enlightenment weighting
        total_score = df['nirvana_score'].sum()
        if total_score > 0:
            df['weightage'] = df['nirvana_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: PantheonAlphaRealm Strategy
# =====================================
class PantheonAlphaRealm(BaseStrategy):
    """
    PantheonAlphaRealm: Olympian Pantheon Realm Conquest.
    - Paradigm: Mythic pantheon – alpha realms conquered by god-like indicator alliances (Zeus-OSC, Athena-RSI), compounding heroic epics in return sagas.
    - Innovation: Pantheon alliance strength via min-max god scores for realm dominance.
    - Weighting: Epic saga normalization for pantheon harmony.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            'zscore latest', 'ma90 latest', 'ma200 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. God Scores (Individual pantheon)
        zeus_osc = np.abs(df['osc latest']) / 50 * 2.0
        athena_rsi = np.clip((df['rsi latest'] - 50) / 25, 0, 2.5)
        df['pantheon_gods'] = np.minimum(zeus_osc, athena_rsi) * np.abs(df['zscore latest'])

        # 2. Alliance Conquest (Min god * max trend)
        conquest = df['pantheon_gods'] * np.maximum(df['price'] / df['ma90 latest'] - 1, 0) * 1.5
        df['alliance_conq'] = np.clip(conquest, 0.2, 3.7)

        # 3. Epic Compounding (Alliance ^ gods)
        epic = df['alliance_conq'] ** (df['pantheon_gods'] / 2)
        df['epic_comp'] = np.clip(epic * np.where(df['price'] > df['ma200 latest'], 2.2, 0.6), 0.4, 4.4)

        # 4. Pantheon Score
        df['pantheon_score'] = df['pantheon_gods'] * df['alliance_conq'] * df['epic_comp']

        df['pantheon_score'] = np.maximum(df['pantheon_score'], 0.01)

        # Harmony weighting
        total_score = df['pantheon_score'].sum()
        if total_score > 0:
            df['weightage'] = df['pantheon_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: ZenithMomentumPeak Strategy
# =====================================
class ZenithMomentumPeak(BaseStrategy):
    """
    ZenithMomentumPeak: Zenith Peak Ascension in Momentum.
    - Paradigm: Himalayan zenith – alpha peaks at summit where momentum ascends through base camps (indicator layers), compounding via altitude gains for peak conquest.
    - Innovation: Altitude score as layered indicator climbs for zenith height.
    - Weighting: Summit gradient for peak equilibrium.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma90 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Base Camp Layer (Low indicators)
        base = np.minimum(df['rsi latest'], np.abs(df['osc latest'])) / 50
        df['base_camp'] = np.clip(base, 0.1, 2.1)

        # 2. Ascent Climb (EMA height)
        climb = np.abs(df['9ema osc latest'] - df['21ema osc latest']) / 50 * df['base_camp']
        df['ascent_climb'] = np.clip(climb, 0.3, 2.6)

        # 3. Zenith Compounding (Climb * z-altitude)
        zen_comp = df['ascent_climb'] * np.abs(df['zscore latest']) * np.where(df['price'] > df['ma90 latest'], 2.8, 0.5)
        df['zen_comp'] = np.clip(zen_comp, 0.4, 4.0)

        # 4. Peak Score
        df['zenith_score'] = df['base_camp'] * df['ascent_climb'] * df['zen_comp']

        df['zenith_score'] = np.maximum(df['zenith_score'], 0.01)

        # Gradient weighting
        total_score = df['zenith_score'].sum()
        if total_score > 0:
            df['weightage'] = df['zenith_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: OmniscienceReturn Strategy
# =====================================
class OmniscienceReturn(BaseStrategy):
    """
    OmniscienceReturn: Omniscient All-Seeing Return Vision.
    - Paradigm: Omniscient gaze – alpha from all-seeing synthesis where indicators 'know' future returns via holistic Bayesian priors, compounding omniscience probabilities.
    - Innovation: Prior likelihood fusion for omniscient forecast.
    - Weighting: Probability density for all-seeing balance.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', 'zscore latest',
            '9ema osc latest', 'ma200 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Bayesian Prior (Indicator beliefs)
        prior_rsi = 1 / (1 + np.exp(-(df['rsi latest'] - 50) / 20))
        prior_osc = 1 / (1 + np.exp(-df['osc latest'] / 40))
        df['bay_prior'] = (prior_rsi + prior_osc) / 2 * 2.3

        # 2. Likelihood Fusion (Z-evidence)
        lik = df['bay_prior'] * np.abs(df['zscore latest']) / (df['dev20 latest'] / df['price'] + 1e-6)
        df['lik_fus'] = np.clip(lik, 0.2, 3.0)

        # 3. Omniscient Posterior (Prior * lik * trend)
        post = df['bay_prior'] * df['lik_fus'] * np.where(df['9ema osc latest'] > 0, 2.4, 0.6)
        df['omni_post'] = np.clip(post * np.where(df['price'] > df['ma200 latest'], 1.9, 0.4), 0.3, 4.2)

        # 4. Omniscience Score
        df['omni_score'] = df['bay_prior'] * df['lik_fus'] * df['omni_post']

        df['omni_score'] = np.maximum(df['omni_score'], 0.01)

        # Density weighting
        total_score = df['omni_score'].sum()
        if total_score > 0:
            df['weightage'] = df['omni_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: ApotheosisMomentum Strategy
# =====================================
class ApotheosisMomentum(BaseStrategy):
    """
    ApotheosisMomentum: Apotheosis Deification of Momentum Gods.
    - Paradigm: Deification rite – alpha apotheosis where mortal indicators ascend to godhood via ritual amplification, compounding divine momentum for immortal returns.
    - Innovation: Ascension rite as exponential god-tier thresholds.
    - Weighting: Divine aura normalization for apotheotic harmony.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'osc latest', 'rsi latest', '9ema osc latest',
            '21ema osc latest', 'zscore latest', 'ma90 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Mortal Threshold (Base rite)
        mortal = np.minimum(np.abs(df['osc latest']), df['rsi latest']) / 50
        df['mortal_rite'] = np.clip(mortal, 0.1, 2.0)

        # 2. Ascension Amp (EMA god-lift)
        asc = np.exp(np.abs(df['9ema osc latest'] - df['21ema osc latest']) / 50) - 1
        df['asc_amp'] = np.clip(asc * df['mortal_rite'], 0.3, 3.1)

        # 3. Apotheosis Comp (Amp ^ z-god)
        apo_comp = df['asc_amp'] ** np.abs(df['zscore latest']) * np.where(df['price'] > df['ma90 latest'], 2.7, 0.5)
        df['apo_comp'] = np.clip(apo_comp, 0.4, 4.5)

        # 4. God Score
        df['apo_score'] = df['mortal_rite'] * df['asc_amp'] * df['apo_comp']

        df['apo_score'] = np.maximum(df['apo_score'], 0.01)

        # Aura weighting
        total_score = df['apo_score'].sum()
        if total_score > 0:
            df['weightage'] = df['apo_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

# =====================================
# NEW: TranscendentAlpha Strategy
# =====================================
class TranscendentAlpha(BaseStrategy):
    """
    TranscendentAlpha: Transcendent Beyond-Alpha Enlightenment.
    - Paradigm: Transcendental ascent – alpha transcendence beyond metrics, where indicators dissolve into pure momentum essence, compounding eternal beyond-returns.
    - Innovation: Essence distillation as transcendental functions (e.g., gamma integrals approx) for pure yield.
    - Weighting: Essence flow for transcendent unity.
    """
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'osc latest', 'zscore latest',
            '9ema osc latest', 'ma200 latest', 'dev20 latest'
        ]
        df = self._clean_data(df, required_columns)

        # 1. Essence Distill (Transcend metrics)
        distill_rsi = np.sqrt(np.abs(df['rsi latest'] - 50)) / np.sqrt(50) * 2
        distill_osc = np.sqrt(np.abs(df['osc latest'])) / np.sqrt(100) * 2
        df['ess_distill'] = (distill_rsi + distill_osc) / 2

        # 2. Transcend Flow (Z-essence)
        flow = df['ess_distill'] * np.abs(df['zscore latest']) / (df['dev20 latest'] / df['price'] + 1e-6)
        df['trans_flow'] = np.clip(flow, 0.2, 3.2)

        # 3. Beyond Comp (Flow ^ essence)
        beyond = df['trans_flow'] ** df['ess_distill'] * np.where(df['9ema osc latest'] > 0, 2.9, 0.4)
        df['beyond_comp'] = np.clip(beyond * np.where(df['price'] > df['ma200 latest'], 1.8, 0.5), 0.3, 4.6)

        # 4. Transcendent Score
        df['trans_score'] = df['ess_distill'] * df['trans_flow'] * df['beyond_comp']

        df['trans_score'] = np.maximum(df['trans_score'], 0.01)

        # Unity weighting
        total_score = df['trans_score'].sum()
        if total_score > 0:
            df['weightage'] = df['trans_score'] / total_score
        else:
            df['weightage'] = 1.0 / len(df)

        return self._allocate_portfolio(df, sip_amount)

class TurnaroundSniper(BaseStrategy):
    """
    Snipes the exact turnaround point: deeply oversold + first signs of momentum reversal.
    Key: multiplicative scoring like VolatilitySurfer but for mean reversion setups.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # 1. OVERSOLD DEPTH SCORE (higher = more oversold = better opportunity)
        # Combine multiple oversold indicators
        zscore_depth = np.clip(-df['zscore latest'] / 1.5, 0, 3.0)  # z < -1.5 starts scoring
        zscore_weekly_depth = np.clip(-df['zscore weekly'] / 1.5, 0, 2.5)
        rsi_depth = np.clip((30 - df['rsi latest']) / 15, 0, 2.0)  # RSI < 30 scores
        osc_depth = np.clip((-df['osc latest'] - 40) / 30, 0, 2.5)  # OSC < -40 scores
        
        oversold_score = (zscore_depth * 0.35 + zscore_weekly_depth * 0.25 + 
                         rsi_depth * 0.20 + osc_depth * 0.20)
        
        # 2. TURNAROUND SIGNAL (the key differentiator)
        # 9EMA crossing above 21EMA while still oversold = turnaround
        ema_cross_bullish = df['9ema osc latest'] > df['21ema osc latest']
        ema_cross_weekly = df['9ema osc weekly'] > df['21ema osc weekly']
        
        # Daily improving faster than weekly = leading indicator
        daily_leading = df['osc latest'] > df['osc weekly']
        
        turnaround_score = np.where(
            ema_cross_bullish & ema_cross_weekly & daily_leading,
            3.0,  # Triple confirmation
            np.where(
                ema_cross_bullish & daily_leading,
                2.5,  # Daily turn + leading
                np.where(
                    ema_cross_bullish,
                    2.0,  # Just daily turn
                    np.where(
                        daily_leading & (df['osc latest'] > -50),
                        1.5,  # Improving but no cross yet
                        0.3   # No turnaround signal - heavy penalty
                    )
                )
            )
        )
        
        # 3. TREND CONTEXT (prefer pullbacks in uptrends)
        trend_mult = np.where(
            df['ma90 latest'] > df['ma200 latest'],
            1.4,  # Uptrend pullback
            np.where(
                df['ma90 weekly'] > df['ma200 weekly'],
                1.2,  # Weekly still positive
                np.where(
                    df['price'] > df['ma200 weekly'],
                    1.0,  # At least above long-term
                    0.6   # Structural downtrend - risky
                )
            )
        )
        
        # 4. VOLATILITY QUALITY (moderate vol preferred)
        vol_ratio = df['dev20 latest'] / df['price']
        vol_mult = np.where(
            (vol_ratio > 0.015) & (vol_ratio < 0.04),
            1.2,  # Sweet spot
            np.where(
                vol_ratio > 0.06,
                0.6,  # Too volatile
                0.9   # Too quiet
            )
        )
        
        # FINAL: Multiplicative combination (all factors must align)
        df['composite_score'] = oversold_score * turnaround_score * trend_mult * vol_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        
        return self._allocate_portfolio(df, sip_amount)


# =========================================================================
# Strategy 2: MomentumAccelerator
# Captures momentum ACCELERATION not just momentum
# =========================================================================

class MomentumAccelerator(BaseStrategy):
    """
    Targets stocks where momentum is ACCELERATING (second derivative positive).
    Differentiator: focuses on rate of change of momentum, not absolute level.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # 1. EMA SPREAD ACCELERATION
        # 9EMA - 21EMA spread: positive and widening = accelerating
        ema_spread_daily = df['9ema osc latest'] - df['21ema osc latest']
        ema_spread_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        
        # Acceleration: daily spread > weekly spread AND both positive
        acceleration_score = np.where(
            (ema_spread_daily > ema_spread_weekly) & (ema_spread_daily > 0) & (ema_spread_weekly > 0),
            3.0,  # Strong acceleration
            np.where(
                (ema_spread_daily > ema_spread_weekly) & (ema_spread_daily > 0),
                2.5,  # Daily accelerating, weekly catching up
                np.where(
                    (ema_spread_daily > 0) & (ema_spread_weekly > 0),
                    2.0,  # Both positive but not accelerating
                    np.where(
                        (ema_spread_daily > 0),
                        1.5,  # Daily positive
                        np.where(
                            ema_spread_daily > ema_spread_weekly,
                            1.0,  # At least improving
                            0.3   # Decelerating
                        )
                    )
                )
            )
        )
        
        # 2. RSI VELOCITY
        # RSI distance from oversold: the further above 30, the more momentum
        rsi_velocity = np.clip((df['rsi latest'] - 30) / 40, 0, 1.5)
        rsi_weekly_velocity = np.clip((df['rsi weekly'] - 35) / 35, 0, 1.3)
        velocity_score = rsi_velocity * 0.6 + rsi_weekly_velocity * 0.4
        
        # 3. OSCILLATOR MOMENTUM
        # Positive and improving oscillator
        osc_momentum = np.where(
            (df['osc latest'] > 0) & (df['osc weekly'] > 0),
            np.clip(1 + df['osc latest'] / 50, 1, 2.5),
            np.where(
                df['osc latest'] > df['osc weekly'],
                np.clip(1 + (df['osc latest'] - df['osc weekly']) / 30, 0.8, 1.5),
                0.5
            )
        )
        
        # 4. TREND STRENGTH MULTIPLIER
        trend_mult = np.where(
            (df['price'] > df['ma90 latest']) & (df['ma90 latest'] > df['ma200 latest']),
            1.5,  # Golden cross alignment
            np.where(
                df['price'] > df['ma90 latest'],
                1.2,
                np.where(
                    df['price'] > df['ma200 latest'],
                    1.0,
                    0.5  # Below MA200 - weak
                )
            )
        )
        
        # 5. Z-SCORE CONFIRMATION (not extreme = sustainable)
        zscore_quality = np.where(
            (df['zscore latest'] > -1) & (df['zscore latest'] < 2),
            1.2,  # Normal range - sustainable
            np.where(
                df['zscore latest'] > 2.5,
                0.7,  # Overextended
                1.0
            )
        )
        
        # FINAL SCORE
        df['composite_score'] = acceleration_score * velocity_score * osc_momentum * trend_mult * zscore_quality
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        
        return self._allocate_portfolio(df, sip_amount)


# =========================================================================
# Strategy 3: VolatilityRegimeTrader
# Explicitly trades different volatility regimes
# =========================================================================

class VolatilityRegimeTrader(BaseStrategy):
    """
    Adapts strategy based on volatility regime of each stock.
    Low vol: buy breakouts. High vol: buy mean reversion. Medium: momentum.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # Calculate volatility percentile for each stock
        vol_ratio = df['dev20 latest'] / df['price']
        vol_percentile = vol_ratio.rank(pct=True)
        
        # Bollinger Band position
        bb_upper = df['ma20 latest'] + 2 * df['dev20 latest']
        bb_lower = df['ma20 latest'] - 2 * df['dev20 latest']
        bb_position = (df['price'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
        
        # LOW VOLATILITY REGIME: Breakout strategy (like VolatilitySurfer)
        low_vol_score = np.where(
            vol_percentile < 0.33,
            np.where(
                (df['price'] > bb_upper) & (df['osc latest'] > 0),
                3.0,  # Breakout with confirmation
                np.where(
                    bb_position > 0.8,
                    2.0,  # Near breakout
                    np.where(
                        (df['9ema osc latest'] > df['21ema osc latest']) & (df['osc latest'] > -20),
                        1.5,  # Building momentum
                        0.5
                    )
                )
            ),
            0.0  # Not in low vol regime
        )
        
        # HIGH VOLATILITY REGIME: Mean reversion strategy
        high_vol_score = np.where(
            vol_percentile > 0.67,
            np.where(
                (bb_position < 0.2) & (df['rsi latest'] < 35) & (df['9ema osc latest'] > df['21ema osc latest']),
                3.0,  # Oversold with turn signal
                np.where(
                    (bb_position < 0.3) & (df['zscore latest'] < -1.5),
                    2.0,  # Oversold
                    np.where(
                        (df['osc latest'] < -50) & (df['osc latest'] > df['osc weekly']),
                        1.5,  # Deep oversold improving
                        0.3
                    )
                )
            ),
            0.0  # Not in high vol regime
        )
        
        # MEDIUM VOLATILITY REGIME: Momentum strategy
        med_vol_score = np.where(
            (vol_percentile >= 0.33) & (vol_percentile <= 0.67),
            np.where(
                (df['rsi latest'] > 50) & (df['osc latest'] > 0) & (df['price'] > df['ma90 latest']),
                2.5,  # Strong momentum
                np.where(
                    (df['9ema osc latest'] > df['21ema osc latest']) & (df['rsi latest'] > 45),
                    2.0,  # Building momentum
                    np.where(
                        df['price'] > df['ma200 latest'],
                        1.0,  # At least in uptrend
                        0.5
                    )
                )
            ),
            0.0  # Not in medium vol regime
        )
        
        # Combine regime scores
        regime_score = low_vol_score + high_vol_score + med_vol_score
        
        # TREND QUALITY MULTIPLIER
        trend_mult = np.where(
            df['ma90 latest'] > df['ma200 latest'],
            1.3,
            np.where(
                df['price'] > df['ma200 weekly'],
                1.0,
                0.7
            )
        )
        
        # CONSISTENCY BONUS (daily and weekly agree)
        consistency = np.where(
            np.sign(df['osc latest']) == np.sign(df['osc weekly']),
            1.2,
            0.9
        )
        
        df['composite_score'] = regime_score * trend_mult * consistency
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        
        return self._allocate_portfolio(df, sip_amount)


# =========================================================================
# Strategy 4: CrossSectionalAlpha
# Pure cross-sectional relative value
# =========================================================================

class CrossSectionalAlpha(BaseStrategy):
    """
    Pure cross-sectional strategy: buy the cheapest vs universe with momentum turn.
    Doesn't care about absolute levels, only relative ranking.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # Calculate cross-sectional ranks (0 = lowest/most oversold, 1 = highest)
        rsi_rank = df['rsi latest'].rank(pct=True)
        osc_rank = df['osc latest'].rank(pct=True)
        zscore_rank = df['zscore latest'].rank(pct=True)
        
        # Combined value rank (lower = cheaper = better)
        value_rank = (rsi_rank * 0.30 + osc_rank * 0.40 + zscore_rank * 0.30)
        
        # VALUE SCORE: Bottom quintile gets highest score
        value_score = np.where(
            value_rank < 0.20,
            3.0,  # Bottom 20%
            np.where(
                value_rank < 0.40,
                2.0,  # 20-40%
                np.where(
                    value_rank < 0.60,
                    1.0,  # 40-60%
                    np.where(
                        value_rank < 0.80,
                        0.5,  # 60-80%
                        0.2   # Top 20% (most overbought)
                    )
                )
            )
        )
        
        # MOMENTUM TURN SIGNAL (critical - don't buy falling knives)
        momentum_turn = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']),
            2.0,  # Turning up
            np.where(
                df['osc latest'] > df['osc weekly'],
                1.5,  # Improving
                0.4   # Still falling - big penalty
            )
        )
        
        # RELATIVE STRENGTH vs UNIVERSE
        # Price vs MA spread rank
        ma_spread = (df['price'] - df['ma200 latest']) / df['ma200 latest']
        ma_spread_rank = ma_spread.rank(pct=True)
        
        # We want stocks that are cheap (low RSI rank) but not in worst trend (not bottom MA rank)
        relative_strength = np.where(
            ma_spread_rank > 0.30,  # Not in bottom 30% by MA spread
            1.2,
            np.where(
                ma_spread_rank > 0.15,
                1.0,
                0.7  # Structurally weak
            )
        )
        
        # VOLATILITY FILTER (prefer moderate vol for better risk/reward)
        vol_rank = (df['dev20 latest'] / df['price']).rank(pct=True)
        vol_filter = np.where(
            (vol_rank > 0.20) & (vol_rank < 0.70),
            1.1,  # Middle 50%
            0.9
        )
        
        df['composite_score'] = value_score * momentum_turn * relative_strength * vol_filter
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        
        return self._allocate_portfolio(df, sip_amount)


# =========================================================================
# Strategy 5: DualMomentum
# Combines absolute and relative momentum
# =========================================================================

class DualMomentum(BaseStrategy):
    """
    Classic dual momentum: requires BOTH absolute momentum (vs own history)
    AND relative momentum (vs universe) to score well.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # ===== ABSOLUTE MOMENTUM =====
        # Price vs moving averages
        above_ma90 = df['price'] > df['ma90 latest']
        above_ma200 = df['price'] > df['ma200 latest']
        ma90_above_ma200 = df['ma90 latest'] > df['ma200 latest']
        
        abs_trend_score = np.where(
            above_ma90 & above_ma200 & ma90_above_ma200,
            3.0,  # Perfect alignment
            np.where(
                above_ma200 & ma90_above_ma200,
                2.5,  # Good trend, minor pullback
                np.where(
                    above_ma200,
                    2.0,  # Above long-term
                    np.where(
                        above_ma90,
                        1.0,  # Short-term bounce only
                        0.3   # Below both MAs
                    )
                )
            )
        )
        
        # Oscillator absolute momentum
        osc_abs_momentum = np.where(
            (df['osc latest'] > 0) & (df['osc weekly'] > 0),
            1.5,
            np.where(
                df['osc latest'] > 0,
                1.2,
                np.where(
                    df['9ema osc latest'] > df['21ema osc latest'],
                    1.0,  # Turning but not positive yet
                    0.6
                )
            )
        )
        
        absolute_momentum = abs_trend_score * osc_abs_momentum
        
        # ===== RELATIVE MOMENTUM =====
        # Rank vs universe
        rsi_rank = df['rsi latest'].rank(pct=True)
        osc_rank = df['osc latest'].rank(pct=True)
        price_mom = df['price'] / df['ma90 latest']
        price_mom_rank = price_mom.rank(pct=True)
        
        # Composite relative rank
        relative_rank = (rsi_rank * 0.30 + osc_rank * 0.35 + price_mom_rank * 0.35)
        
        relative_momentum = np.where(
            relative_rank > 0.70,
            2.0,  # Top 30%
            np.where(
                relative_rank > 0.50,
                1.5,  # Top 50%
                np.where(
                    relative_rank > 0.30,
                    1.0,  # Above median
                    0.5   # Bottom half
                )
            )
        )
        
        # ===== DUAL MOMENTUM COMBINATION =====
        # Require BOTH to score well (multiplicative)
        dual_momentum = absolute_momentum * relative_momentum
        
        # ===== CONFIRMATION SIGNALS =====
        # EMA acceleration
        ema_acceleration = np.where(
            (df['9ema osc latest'] > df['21ema osc latest']) & (df['9ema osc weekly'] > df['21ema osc weekly']),
            1.3,
            np.where(
                df['9ema osc latest'] > df['21ema osc latest'],
                1.1,
                0.9
            )
        )
        
        # Volatility quality
        vol_ratio = df['dev20 latest'] / df['price']
        vol_quality = np.where(
            vol_ratio < 0.02,
            1.2,  # Low vol
            np.where(
                vol_ratio < 0.04,
                1.0,  # Normal
                0.8   # High vol
            )
        )
        
        df['composite_score'] = dual_momentum * ema_acceleration * vol_quality
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        
        return self._allocate_portfolio(df, sip_amount)

# =========================================================================
# 5 SOPHISTICATED STRATEGIES - V4
# Continuous scoring, statistical transforms, nuanced signal processing
# =========================================================================

class AdaptiveZScoreEngine(BaseStrategy):
    """
    Dynamically adjusts z-score sensitivity based on cross-sectional volatility.
    When universe is calm, smaller z-scores matter more.
    When universe is stressed, requires larger deviations.
    Uses sigmoid transforms for smooth scoring.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # ADAPTIVE THRESHOLD based on universe stress
        universe_zscore_std = df['zscore latest'].std()
        universe_osc_mean = df['osc latest'].mean()
        
        # In stressed markets (high dispersion, low osc), adjust sensitivity
        stress_factor = np.clip(universe_zscore_std / 1.5, 0.5, 2.0)
        
        # SIGMOID-TRANSFORMED Z-SCORE
        # Maps z-score to (0, 1) with adaptive midpoint
        adaptive_midpoint = -1.0 * stress_factor
        z_sigmoid = 1 / (1 + np.exp((df['zscore latest'] - adaptive_midpoint) * 1.5))
        z_weekly_sigmoid = 1 / (1 + np.exp((df['zscore weekly'] - adaptive_midpoint * 0.8) * 1.2))
        
        # Combined z-signal with variance-preserving blend
        z_combined = (z_sigmoid * 0.6 + z_weekly_sigmoid * 0.4) * np.sqrt(2)
        
        # RSI SIGMOID (smooth transition, not stepped)
        rsi_sigmoid = 1 / (1 + np.exp((df['rsi latest'] - 35) * 0.15))
        rsi_weekly_sigmoid = 1 / (1 + np.exp((df['rsi weekly'] - 40) * 0.12))
        rsi_combined = (rsi_sigmoid * 0.55 + rsi_weekly_sigmoid * 0.45)
        
        # OSCILLATOR SIGMOID
        osc_sigmoid = 1 / (1 + np.exp((df['osc latest'] + 40) * 0.03))
        osc_weekly_sigmoid = 1 / (1 + np.exp((df['osc weekly'] + 30) * 0.025))
        osc_combined = (osc_sigmoid * 0.5 + osc_weekly_sigmoid * 0.5)
        
        # MOMENTUM TURN GRADIENT (continuous, not binary)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_spread_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        
        # Normalize spreads to comparable scale
        ema_spread_norm = ema_spread / (df['osc latest'].std() + 1)
        ema_weekly_norm = ema_spread_weekly / (df['osc weekly'].std() + 1)
        
        # Sigmoid for turn signal
        turn_sigmoid = 1 / (1 + np.exp(-ema_spread_norm * 2))
        turn_weekly_sigmoid = 1 / (1 + np.exp(-ema_weekly_norm * 1.5))
        turn_combined = (turn_sigmoid * 0.6 + turn_weekly_sigmoid * 0.4)
        
        # TREND QUALITY (continuous)
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        price_ma_ratio = df['price'] / (df['ma200 latest'] + 1e-6)
        trend_score = (np.clip(ma_ratio, 0.8, 1.2) - 0.8) / 0.4 * 0.5 + \
                      (np.clip(price_ma_ratio, 0.85, 1.15) - 0.85) / 0.3 * 0.5
        trend_score = np.clip(trend_score, 0.3, 1.5)
        
        # COMPOSITE with geometric mean for balance
        base_score = (z_combined * rsi_combined * osc_combined) ** (1/3)
        
        # Apply turn and trend as multipliers
        df['composite_score'] = base_score * (0.5 + turn_combined) * trend_score
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class MomentumDecayModel(BaseStrategy):
    """
    Models momentum as a decaying function.
    Recent momentum (daily) decays faster, weekly momentum is more persistent.
    Combines decay-weighted signals for optimal timing.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # MOMENTUM COMPONENTS with different decay rates
        # Daily momentum (fast decay - lambda = 0.8)
        daily_mom_raw = (df['rsi latest'] - 50) / 50
        daily_osc_raw = df['osc latest'] / 100
        daily_ema_raw = (df['9ema osc latest'] - df['21ema osc latest']) / 50
        
        # Weekly momentum (slow decay - lambda = 0.95)
        weekly_mom_raw = (df['rsi weekly'] - 50) / 50
        weekly_osc_raw = df['osc weekly'] / 100
        weekly_ema_raw = (df['9ema osc weekly'] - df['21ema osc weekly']) / 50
        
        # DECAY-WEIGHTED COMBINATION
        # More weight to weekly (persistent) when daily is noisy
        daily_vol = np.abs(daily_mom_raw - weekly_mom_raw)
        decay_weight = 1 / (1 + daily_vol * 3)  # Higher noise = more weekly weight
        
        blended_mom = daily_mom_raw * (1 - decay_weight * 0.3) + weekly_mom_raw * decay_weight * 0.3
        blended_osc = daily_osc_raw * (1 - decay_weight * 0.4) + weekly_osc_raw * decay_weight * 0.4
        blended_ema = daily_ema_raw * (1 - decay_weight * 0.3) + weekly_ema_raw * decay_weight * 0.3
        
        # MEAN REVERSION SIGNAL (negative momentum = opportunity)
        # Use tanh for bounded output
        reversion_signal = np.tanh(-blended_mom * 2) * 0.5 + 0.5  # Maps to (0, 1)
        osc_signal = np.tanh(-blended_osc * 2) * 0.5 + 0.5
        
        # ACCELERATION (second derivative proxy)
        ema_acceleration = blended_ema - weekly_ema_raw  # Daily leading weekly
        accel_signal = np.tanh(ema_acceleration * 3) * 0.5 + 0.5
        
        # Z-SCORE OVERLAY
        z_signal = np.tanh(-df['zscore latest'] * 0.8) * 0.5 + 0.5
        z_weekly_signal = np.tanh(-df['zscore weekly'] * 0.6) * 0.5 + 0.5
        
        # COMPOSITE using power mean (p=2 for quadratic emphasis)
        signals = np.column_stack([reversion_signal, osc_signal, accel_signal, z_signal, z_weekly_signal])
        weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
        
        # Weighted power mean
        power_mean = np.power(np.sum(np.power(signals, 2) * weights, axis=1), 0.5)
        
        # TREND CONTEXT (soft multiplier)
        trend_ratio = df['price'] / df['ma200 latest']
        trend_mult = np.tanh((trend_ratio - 0.9) * 5) * 0.3 + 1.0  # Centers around 1.0
        
        # VOLATILITY ADJUSTMENT
        vol_ratio = df['dev20 latest'] / df['price']
        vol_penalty = 1 / (1 + np.exp((vol_ratio - 0.03) * 100))  # Penalize high vol
        
        df['composite_score'] = power_mean * trend_mult * (0.7 + vol_penalty * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class InformationRatioOptimizer(BaseStrategy):
    """
    Weights signals by their information content (signal-to-noise ratio).
    Noisier signals get less weight. Cleaner signals dominate.
    Cross-sectional dispersion determines signal quality.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # SIGNAL EXTRACTION with cross-sectional normalization
        def cs_normalize(series):
            mean, std = series.mean(), series.std()
            return (series - mean) / (std + 1e-6)
        
        rsi_norm = cs_normalize(df['rsi latest'])
        rsi_weekly_norm = cs_normalize(df['rsi weekly'])
        osc_norm = cs_normalize(df['osc latest'])
        osc_weekly_norm = cs_normalize(df['osc weekly'])
        zscore_norm = cs_normalize(df['zscore latest'])
        zscore_weekly_norm = cs_normalize(df['zscore weekly'])
        
        # INFORMATION RATIO for each signal (how much does it differentiate?)
        # Higher dispersion = more information
        rsi_ir = np.abs(rsi_norm) / (np.abs(rsi_norm - rsi_weekly_norm) + 0.5)
        osc_ir = np.abs(osc_norm) / (np.abs(osc_norm - osc_weekly_norm) + 0.5)
        zscore_ir = np.abs(zscore_norm) / (np.abs(zscore_norm - zscore_weekly_norm) + 0.5)
        
        # Normalize IRs to weights
        total_ir = rsi_ir + osc_ir + zscore_ir + 1e-6
        rsi_weight = rsi_ir / total_ir
        osc_weight = osc_ir / total_ir
        zscore_weight = zscore_ir / total_ir
        
        # VALUE SIGNALS (lower = more oversold = better)
        rsi_value = 1 - (df['rsi latest'] / 100)  # 0-1, higher = more oversold
        osc_value = (100 - df['osc latest']) / 200  # Normalized to ~0-1
        zscore_value = 1 / (1 + np.exp(df['zscore latest'] * 0.8))  # Sigmoid transform
        
        # INFORMATION-WEIGHTED COMPOSITE
        weighted_value = (rsi_value * rsi_weight + 
                         osc_value * osc_weight + 
                         zscore_value * zscore_weight)
        
        # MOMENTUM TURN SIGNAL with confidence weighting
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_spread_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        
        # Confidence: both agreeing = high confidence
        spread_agreement = ema_spread * ema_spread_weekly
        confidence = np.tanh(spread_agreement / 500) * 0.5 + 0.5
        
        turn_signal = np.tanh(ema_spread / 30) * 0.5 + 0.5
        weighted_turn = turn_signal * confidence + 0.5 * (1 - confidence)
        
        # TREND QUALITY
        ma_alignment = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        price_position = df['price'] / (df['ma200 latest'] + 1e-6)
        trend_quality = (np.clip(ma_alignment, 0.9, 1.1) - 0.9) / 0.2 * 0.5 + \
                       (np.clip(price_position, 0.9, 1.1) - 0.9) / 0.2 * 0.5
        trend_quality = np.clip(trend_quality, 0.4, 1.3)
        
        # VOLATILITY EFFICIENCY (prefer lower vol for same signal)
        vol_ratio = df['dev20 latest'] / df['price']
        vol_efficiency = 1 / (1 + vol_ratio * 20)
        
        df['composite_score'] = weighted_value * weighted_turn * trend_quality * (0.5 + vol_efficiency)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class BayesianMomentumUpdater(BaseStrategy):
    """
    Treats weekly as prior, daily as likelihood.
    Updates belief about momentum state using Bayesian-like weighting.
    Stronger daily signals override weak weekly priors.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # PRIOR (weekly indicators) - probability of being oversold
        prior_rsi = 1 / (1 + np.exp((df['rsi weekly'] - 40) * 0.15))
        prior_osc = 1 / (1 + np.exp((df['osc weekly'] + 35) * 0.04))
        prior_zscore = 1 / (1 + np.exp((df['zscore weekly'] + 1.2) * 0.8))
        
        # Combined prior (geometric mean for independence assumption)
        prior = (prior_rsi * prior_osc * prior_zscore) ** (1/3)
        
        # LIKELIHOOD (daily indicators) - evidence strength
        likelihood_rsi = 1 / (1 + np.exp((df['rsi latest'] - 35) * 0.18))
        likelihood_osc = 1 / (1 + np.exp((df['osc latest'] + 45) * 0.035))
        likelihood_zscore = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.7))
        
        likelihood = (likelihood_rsi * likelihood_osc * likelihood_zscore) ** (1/3)
        
        # EVIDENCE STRENGTH (how much should daily update prior?)
        # Large deviation from weekly = strong evidence
        rsi_deviation = np.abs(df['rsi latest'] - df['rsi weekly']) / 20
        osc_deviation = np.abs(df['osc latest'] - df['osc weekly']) / 40
        evidence_strength = np.tanh((rsi_deviation + osc_deviation) / 2)
        
        # POSTERIOR (Bayesian update approximation)
        # High evidence strength = more weight to likelihood
        posterior = prior * (1 - evidence_strength * 0.6) + likelihood * evidence_strength * 0.6
        
        # Normalize posterior to prevent extremes
        posterior = np.clip(posterior, 0.05, 0.95)
        
        # MOMENTUM TURN (daily leading weekly = positive update)
        ema_spread_daily = df['9ema osc latest'] - df['21ema osc latest']
        ema_spread_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        
        # Turn signal as likelihood ratio
        turn_lr = np.exp(ema_spread_daily / 30) / (np.exp(ema_spread_weekly / 40) + 0.1)
        turn_signal = np.tanh(np.log(turn_lr + 0.1)) * 0.5 + 0.5
        
        # TREND PRIOR
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_prior = 1 / (1 + np.exp(-(ma_ratio - 1.0) * 15))
        
        price_ratio = df['price'] / (df['ma200 latest'] + 1e-6)
        trend_likelihood = 1 / (1 + np.exp(-(price_ratio - 0.95) * 10))
        
        trend_posterior = trend_prior * 0.4 + trend_likelihood * 0.6
        
        # FINAL SCORE
        df['composite_score'] = posterior * (0.4 + turn_signal * 0.8) * (0.5 + trend_posterior * 0.7)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class RelativeStrengthRotator(BaseStrategy):
    """
    Ranks stocks on relative strength vs universe.
    Rotates into bottom quartile with improving momentum.
    Uses percentile-based scoring for robustness.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # PERCENTILE RANKS (0 = weakest, 1 = strongest)
        rsi_pct = df['rsi latest'].rank(pct=True)
        osc_pct = df['osc latest'].rank(pct=True)
        zscore_pct = df['zscore latest'].rank(pct=True)
        price_ma_pct = (df['price'] / df['ma90 latest']).rank(pct=True)
        
        # COMPOSITE WEAKNESS RANK
        weakness_rank = (rsi_pct * 0.25 + osc_pct * 0.35 + 
                        zscore_pct * 0.25 + price_ma_pct * 0.15)
        
        # WEAKNESS SCORE (lower rank = higher score)
        # Use beta distribution-like shaping for concentration in tails
        alpha, beta_param = 0.5, 2.0  # Emphasize bottom
        weakness_score = np.power(1 - weakness_rank, beta_param) * np.power(weakness_rank + 0.1, alpha)
        weakness_score = weakness_score / weakness_score.max()  # Normalize
        
        # IMPROVEMENT SIGNALS (momentum picking up)
        rsi_improvement = df['rsi latest'] - df['rsi weekly']
        osc_improvement = df['osc latest'] - df['osc weekly']
        
        # Rank improvements
        rsi_imp_pct = rsi_improvement.rank(pct=True)
        osc_imp_pct = osc_improvement.rank(pct=True)
        
        # Want high improvement rank (top improvers)
        improvement_score = (rsi_imp_pct * 0.4 + osc_imp_pct * 0.6)
        
        # EMA MOMENTUM (acceleration)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_spread_pct = ema_spread.rank(pct=True)
        
        # COMPOSITE: weak stocks that are improving
        # Multiplicative blend
        base_score = np.sqrt(weakness_score * improvement_score)
        
        # ACCELERATION BOOST
        accel_boost = 0.7 + ema_spread_pct * 0.6
        
        # TREND FILTER (soft)
        trend_pct = (df['ma90 latest'] / df['ma200 latest']).rank(pct=True)
        trend_filter = 0.6 + trend_pct * 0.6
        
        # VOLATILITY EFFICIENCY
        vol_pct = (df['dev20 latest'] / df['price']).rank(pct=True)
        vol_filter = 1.2 - vol_pct * 0.4  # Lower vol = higher score
        
        df['composite_score'] = base_score * accel_boost * trend_filter * vol_filter
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class VolatilityAdjustedValue(BaseStrategy):
    """
    Scores value (oversold) per unit of volatility.
    Sharpe-ratio-like: signal / risk.
    Prefers strong signals in calm names.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # VALUE SIGNAL (higher = more oversold)
        rsi_value = np.clip((40 - df['rsi latest']) / 25, 0, 1.5)
        osc_value = np.clip((-df['osc latest'] - 30) / 50, 0, 1.5)
        zscore_value = np.clip(-df['zscore latest'] / 2, 0, 2.0)
        
        # Blend values
        raw_value = (rsi_value * 0.30 + osc_value * 0.40 + zscore_value * 0.30)
        
        # VOLATILITY (risk measure)
        vol_daily = df['dev20 latest'] / df['price']
        vol_weekly = df['dev20 weekly'] / df['price']
        blended_vol = vol_daily * 0.6 + vol_weekly * 0.4
        
        # Normalize volatility to cross-sectional z-score
        vol_mean, vol_std = blended_vol.mean(), blended_vol.std()
        vol_zscore = (blended_vol - vol_mean) / (vol_std + 1e-6)
        
        # VOLATILITY ADJUSTMENT (value per unit vol)
        # Use softplus to avoid division issues
        vol_adjustment = 1 / (1 + np.exp(vol_zscore * 1.5))  # Lower vol = higher adjustment
        
        value_per_vol = raw_value * (0.5 + vol_adjustment)
        
        # MOMENTUM TURN
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        
        turn_signal = np.tanh(ema_spread / 25) * 0.5 + 0.5
        turn_weekly = np.tanh(ema_weekly / 30) * 0.5 + 0.5
        combined_turn = turn_signal * 0.6 + turn_weekly * 0.4
        
        # TREND QUALITY
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_signal = np.tanh((ma_ratio - 1.0) * 10) * 0.5 + 0.5
        
        price_ratio = df['price'] / (df['ma200 latest'] + 1e-6)
        price_signal = np.tanh((price_ratio - 0.95) * 8) * 0.5 + 0.5
        
        trend_combined = trend_signal * 0.5 + price_signal * 0.5
        
        # FINAL
        df['composite_score'] = value_per_vol * (0.4 + combined_turn * 0.8) * (0.5 + trend_combined * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class NonlinearMomentumBlender(BaseStrategy):
    """
    Uses polynomial and exponential transforms to capture non-linear relationships.
    Momentum at extremes behaves differently than at center.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # NONLINEAR RSI TRANSFORM
        # Quadratic emphasis at extremes
        rsi_centered = (df['rsi latest'] - 50) / 50  # -1 to 1
        rsi_nonlinear = -np.sign(rsi_centered) * np.power(np.abs(rsi_centered), 1.5)
        rsi_score = (1 - rsi_nonlinear) / 2  # Map to 0-1, higher for oversold
        
        rsi_weekly_centered = (df['rsi weekly'] - 50) / 50
        rsi_weekly_nonlinear = -np.sign(rsi_weekly_centered) * np.power(np.abs(rsi_weekly_centered), 1.5)
        rsi_weekly_score = (1 - rsi_weekly_nonlinear) / 2
        
        # EXPONENTIAL OSC TRANSFORM
        # Exponential decay from neutral
        osc_exp = np.exp(-df['osc latest'] / 40)  # Higher for negative OSC
        osc_exp = np.clip(osc_exp / osc_exp.max(), 0, 1)
        
        osc_weekly_exp = np.exp(-df['osc weekly'] / 50)
        osc_weekly_exp = np.clip(osc_weekly_exp / osc_weekly_exp.max(), 0, 1)
        
        # POLYNOMIAL Z-SCORE
        z_poly = np.clip(-df['zscore latest'], 0, 3)
        z_poly_score = z_poly ** 1.3 / (3 ** 1.3)  # Polynomial emphasis
        
        z_weekly_poly = np.clip(-df['zscore weekly'], 0, 2.5)
        z_weekly_poly_score = z_weekly_poly ** 1.2 / (2.5 ** 1.2)
        
        # BLEND with variance preservation
        n_signals = 6
        blended = (rsi_score + rsi_weekly_score + osc_exp + osc_weekly_exp + 
                  z_poly_score + z_weekly_poly_score) / np.sqrt(n_signals)
        
        # EMA CROSSOVER (cubic sensitivity)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_normalized = np.clip(ema_spread / 40, -1, 1)
        ema_cubic = np.sign(ema_normalized) * np.power(np.abs(ema_normalized), 0.7)
        ema_score = (ema_cubic + 1) / 2  # Map to 0-1
        
        ema_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        ema_weekly_norm = np.clip(ema_weekly / 50, -1, 1)
        ema_weekly_score = (np.sign(ema_weekly_norm) * np.power(np.abs(ema_weekly_norm), 0.8) + 1) / 2
        
        turn_combined = ema_score * 0.6 + ema_weekly_score * 0.4
        
        # TREND (softplus transform)
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_raw = np.log1p(np.exp((ma_ratio - 1) * 10))  # Softplus
        trend_score = trend_raw / (trend_raw.max() + 1e-6)
        
        price_ratio = df['price'] / (df['ma200 latest'] + 1e-6)
        price_raw = np.log1p(np.exp((price_ratio - 0.95) * 8))
        price_score = price_raw / (price_raw.max() + 1e-6)
        
        trend_combined = (trend_score + price_score) / 2
        
        df['composite_score'] = blended * (0.4 + turn_combined * 0.8) * (0.5 + trend_combined * 0.7)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class EntropyWeightedSelector(BaseStrategy):
    """
    Weights signals by their entropy (information content).
    Low entropy signals (concentrated) get more weight.
    High entropy signals (dispersed) get less weight.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        def calc_entropy_weight(series):
            """Lower entropy = more concentrated = higher weight"""
            # Normalize to probabilities
            shifted = series - series.min() + 1e-6
            probs = shifted / shifted.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(series))
            normalized_entropy = entropy / max_entropy
            return 1 - normalized_entropy  # Invert: low entropy = high weight
        
        # Calculate entropy weights for each signal
        rsi_entropy_w = calc_entropy_weight(100 - df['rsi latest'])
        osc_entropy_w = calc_entropy_weight(-df['osc latest'])
        zscore_entropy_w = calc_entropy_weight(-df['zscore latest'])
        
        # Normalize weights
        total_entropy_w = rsi_entropy_w + osc_entropy_w + zscore_entropy_w + 1e-6
        w_rsi = rsi_entropy_w / total_entropy_w
        w_osc = osc_entropy_w / total_entropy_w
        w_zscore = zscore_entropy_w / total_entropy_w
        
        # VALUE SIGNALS
        rsi_value = 1 / (1 + np.exp((df['rsi latest'] - 35) * 0.15))
        osc_value = 1 / (1 + np.exp((df['osc latest'] + 40) * 0.035))
        zscore_value = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.7))
        
        # ENTROPY-WEIGHTED COMPOSITE
        weighted_value = rsi_value * w_rsi + osc_value * w_osc + zscore_value * w_zscore
        
        # WEEKLY CONFIRMATION
        rsi_weekly_value = 1 / (1 + np.exp((df['rsi weekly'] - 40) * 0.12))
        osc_weekly_value = 1 / (1 + np.exp((df['osc weekly'] + 35) * 0.03))
        weekly_avg = (rsi_weekly_value + osc_weekly_value) / 2
        
        # Combine daily and weekly
        combined_value = weighted_value * 0.55 + weekly_avg * 0.45
        
        # MOMENTUM TURN
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 25))
        
        # TREND
        ma_trend = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_signal = 1 / (1 + np.exp(-(ma_trend - 1.0) * 12))
        
        price_trend = df['price'] / (df['ma200 latest'] + 1e-6)
        price_signal = 1 / (1 + np.exp(-(price_trend - 0.95) * 10))
        
        trend_combined = (trend_signal + price_signal) / 2
        
        df['composite_score'] = combined_value * (0.4 + turn_signal * 0.8) * (0.5 + trend_combined * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


# =========================================================================
# MORE SOPHISTICATED STRATEGIES - V5
# Novel quantitative approaches
# =========================================================================

class KalmanFilterMomentum(BaseStrategy):
    """
    Kalman-filter inspired: estimates 'true' momentum state from noisy observations.
    Daily and weekly are noisy measurements of underlying momentum.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # Measurement noise estimates (cross-sectional variance)
        rsi_noise = df['rsi latest'].var() / 100
        osc_noise = df['osc latest'].var() / 1000
        
        # Kalman gain approximation: K = P / (P + R) where P=process var, R=measurement var
        # Higher noise = lower gain = trust measurement less
        rsi_gain = 1 / (1 + rsi_noise * 2)
        osc_gain = 1 / (1 + osc_noise * 2)
        
        # State estimate: weighted average of daily (measurement) and weekly (prior state)
        rsi_state = df['rsi weekly'] + rsi_gain * (df['rsi latest'] - df['rsi weekly'])
        osc_state = df['osc weekly'] + osc_gain * (df['osc latest'] - df['osc weekly'])
        
        # Innovation (surprise): how much daily differs from prediction
        rsi_innovation = np.abs(df['rsi latest'] - df['rsi weekly']) / 20
        osc_innovation = np.abs(df['osc latest'] - df['osc weekly']) / 40
        
        # Value signal from filtered state
        rsi_value = 1 / (1 + np.exp((rsi_state - 35) * 0.15))
        osc_value = 1 / (1 + np.exp((osc_state + 40) * 0.03))
        
        # Innovation bonus (large positive surprise = opportunity)
        daily_better = (df['rsi latest'] > df['rsi weekly']) | (df['osc latest'] > df['osc weekly'])
        innovation_bonus = np.where(daily_better, 
                                    1 + (rsi_innovation + osc_innovation) * 0.3,
                                    1 - (rsi_innovation + osc_innovation) * 0.15)
        
        # Z-score state
        z_gain = 1 / (1 + df['zscore latest'].var() / 2)
        z_state = df['zscore weekly'] + z_gain * (df['zscore latest'] - df['zscore weekly'])
        z_value = 1 / (1 + np.exp((z_state + 1.5) * 0.7))
        
        # Combined filtered value
        filtered_value = (rsi_value * 0.30 + osc_value * 0.40 + z_value * 0.30) * innovation_bonus
        
        # EMA momentum state
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        ema_state = ema_weekly + 0.6 * (ema_spread - ema_weekly)
        turn_signal = 1 / (1 + np.exp(-ema_state / 25))
        
        # Trend
        trend = 1 / (1 + np.exp(-(df['ma90 latest'] / df['ma200 latest'] - 1) * 12))
        
        df['composite_score'] = filtered_value * (0.4 + turn_signal * 0.8) * (0.5 + trend * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class MeanVarianceOptimizer(BaseStrategy):
    """
    Mean-variance inspired: maximizes expected return per unit risk.
    Expected return from oversold signals, risk from volatility dispersion.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # EXPECTED RETURN PROXY (from mean reversion potential)
        # Deeper oversold = higher expected reversion return
        rsi_exp_ret = np.clip((40 - df['rsi latest']) / 30, 0, 1.5)
        osc_exp_ret = np.clip((-df['osc latest'] - 30) / 60, 0, 1.5)
        z_exp_ret = np.clip(-df['zscore latest'] / 2.5, 0, 1.5)
        
        # Turn signal adds to expected return
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_exp = np.clip(ema_spread / 40 + 0.5, 0, 1.2)
        
        expected_return = (rsi_exp_ret * 0.25 + osc_exp_ret * 0.35 + 
                         z_exp_ret * 0.25 + turn_exp * 0.15)
        
        # RISK ESTIMATE
        vol_daily = df['dev20 latest'] / df['price']
        vol_weekly = df['dev20 weekly'] / df['price']
        
        # Uncertainty from daily-weekly disagreement
        rsi_uncertainty = np.abs(df['rsi latest'] - df['rsi weekly']) / 30
        osc_uncertainty = np.abs(df['osc latest'] - df['osc weekly']) / 50
        
        total_risk = np.sqrt(vol_daily**2 + rsi_uncertainty**2 + osc_uncertainty**2)
        total_risk = np.clip(total_risk, 0.005, 0.15)
        
        # MEAN-VARIANCE SCORE: return / risk (Sharpe-like)
        mv_score = expected_return / total_risk
        
        # Normalize to reasonable range
        mv_score = mv_score / (mv_score.max() + 1e-6)
        
        # TREND QUALITY multiplier
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 8) * 0.3 + 1.0
        
        df['composite_score'] = mv_score * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class RegimeSwitchingStrategy(BaseStrategy):
    """
    Detects market regime from cross-sectional stats and switches strategy.
    Stressed regime: aggressive mean reversion.
    Normal regime: balanced approach.
    Euphoric regime: momentum focus.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # REGIME DETECTION
        avg_rsi = df['rsi latest'].mean()
        avg_osc = df['osc latest'].mean()
        osc_dispersion = df['osc latest'].std()
        pct_oversold = (df['rsi latest'] < 40).mean()
        
        # Regime probabilities (soft classification)
        stressed_prob = 1 / (1 + np.exp((avg_rsi - 35) * 0.2 + (avg_osc + 30) * 0.02))
        euphoric_prob = 1 / (1 + np.exp(-(avg_rsi - 60) * 0.15 - (avg_osc - 20) * 0.02))
        normal_prob = 1 - stressed_prob - euphoric_prob
        normal_prob = np.clip(normal_prob, 0.1, 0.8)
        
        # Normalize
        total_prob = stressed_prob + euphoric_prob + normal_prob
        stressed_prob /= total_prob
        euphoric_prob /= total_prob
        normal_prob /= total_prob
        
        # STRESSED REGIME SCORE (aggressive mean reversion)
        z_depth = np.clip(-df['zscore latest'] / 2, 0, 2)
        rsi_depth = np.clip((35 - df['rsi latest']) / 20, 0, 2)
        osc_depth = np.clip((-df['osc latest'] - 50) / 40, 0, 2)
        stressed_score = (z_depth * 0.35 + rsi_depth * 0.30 + osc_depth * 0.35)
        
        # Turn signal critical in stressed
        ema_turn = df['9ema osc latest'] > df['21ema osc latest']
        stressed_score = stressed_score * np.where(ema_turn, 1.8, 0.5)
        
        # NORMAL REGIME SCORE (balanced)
        rsi_value = 1 / (1 + np.exp((df['rsi latest'] - 40) * 0.12))
        osc_value = 1 / (1 + np.exp((df['osc latest'] + 35) * 0.025))
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_value = 1 / (1 + np.exp(-ema_spread / 30))
        normal_score = (rsi_value * 0.30 + osc_value * 0.40 + turn_value * 0.30)
        
        # EUPHORIC REGIME SCORE (momentum)
        rsi_strength = np.clip((df['rsi latest'] - 45) / 35, 0, 1.5)
        osc_strength = np.clip((df['osc latest'] + 20) / 80, 0, 1.5)
        trend_strength = np.clip((df['price'] / df['ma90 latest'] - 0.95) * 10, 0, 1.5)
        euphoric_score = (rsi_strength * 0.30 + osc_strength * 0.35 + trend_strength * 0.35)
        
        # REGIME-WEIGHTED COMPOSITE
        composite = (stressed_score * stressed_prob + 
                    normal_score * normal_prob + 
                    euphoric_score * euphoric_prob)
        
        # TREND OVERLAY
        ma_trend = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_trend - 0.95) * 6) * 0.25 + 1.0
        
        df['composite_score'] = composite * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class FractalMomentumStrategy(BaseStrategy):
    """
    Treats daily and weekly as fractal scales of same underlying process.
    Self-similarity: patterns at daily should echo at weekly.
    Exploits scale-invariant momentum structures.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # SCALE NORMALIZATION (make daily and weekly comparable)
        rsi_daily_norm = (df['rsi latest'] - 50) / 50
        rsi_weekly_norm = (df['rsi weekly'] - 50) / 50
        
        osc_daily_norm = df['osc latest'] / 100
        osc_weekly_norm = df['osc weekly'] / 100
        
        z_daily_norm = df['zscore latest'] / 3
        z_weekly_norm = df['zscore weekly'] / 2.5
        
        # SELF-SIMILARITY SCORE (correlation between scales)
        # Both scales pointing same direction = fractal alignment
        rsi_alignment = rsi_daily_norm * rsi_weekly_norm
        osc_alignment = osc_daily_norm * osc_weekly_norm
        z_alignment = z_daily_norm * z_weekly_norm
        
        # Positive alignment in oversold = strong signal
        fractal_coherence = (np.clip(rsi_alignment, 0, 1) * 0.30 +
                            np.clip(osc_alignment, 0, 1) * 0.40 +
                            np.clip(z_alignment, 0, 1) * 0.30)
        
        # VALUE at both scales
        rsi_value = (1 / (1 + np.exp(rsi_daily_norm * 3)) + 
                    1 / (1 + np.exp(rsi_weekly_norm * 2.5))) / 2
        
        osc_value = (1 / (1 + np.exp(osc_daily_norm * 2.5)) +
                    1 / (1 + np.exp(osc_weekly_norm * 2))) / 2
        
        z_value = (1 / (1 + np.exp(z_daily_norm * 2)) +
                  1 / (1 + np.exp(z_weekly_norm * 1.8))) / 2
        
        multi_scale_value = (rsi_value * 0.30 + osc_value * 0.40 + z_value * 0.30)
        
        # FRACTAL MOMENTUM (EMA structure at both scales)
        ema_daily = (df['9ema osc latest'] - df['21ema osc latest']) / 50
        ema_weekly = (df['9ema osc weekly'] - df['21ema osc weekly']) / 60
        
        # Both scales turning = fractal turn
        ema_fractal = np.tanh(ema_daily * 2) * np.tanh(ema_weekly * 1.5)
        ema_fractal_score = (ema_fractal + 1) / 2  # Map to 0-1
        
        # Add individual scale turns
        daily_turn = 1 / (1 + np.exp(-ema_daily * 3))
        weekly_turn = 1 / (1 + np.exp(-ema_weekly * 2.5))
        combined_turn = (daily_turn * 0.5 + weekly_turn * 0.3 + ema_fractal_score * 0.2)
        
        # COMPOSITE
        base_score = multi_scale_value * (1 + fractal_coherence * 0.5)
        
        # TREND (fractal view)
        ma_daily_ratio = df['price'] / (df['ma90 latest'] + 1e-6)
        ma_weekly_ratio = df['price'] / (df['ma90 weekly'] + 1e-6)
        trend_fractal = (np.tanh((ma_daily_ratio - 1) * 5) + 
                        np.tanh((ma_weekly_ratio - 1) * 4)) / 2
        trend_score = (trend_fractal + 1) / 2
        
        df['composite_score'] = base_score * (0.4 + combined_turn * 0.8) * (0.5 + trend_score * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class CopulaBlendStrategy(BaseStrategy):
    """
    Models dependency structure between indicators using copula-inspired approach.
    Captures tail dependencies (extreme co-movements) better than correlation.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # UNIFORM MARGINALS (empirical CDF transform)
        def to_uniform(series):
            return series.rank(pct=True)
        
        u_rsi = to_uniform(df['rsi latest'])
        u_osc = to_uniform(df['osc latest'])
        u_zscore = to_uniform(df['zscore latest'])
        u_rsi_w = to_uniform(df['rsi weekly'])
        u_osc_w = to_uniform(df['osc weekly'])
        
        # TAIL DEPENDENCY (lower tail - both in bottom)
        # Clayton-inspired: emphasizes lower tail
        theta = 2.0  # Dependency parameter
        
        def clayton_lower_tail(u1, u2, theta):
            """Lower tail dependency from Clayton copula"""
            return np.power(np.power(u1, -theta) + np.power(u2, -theta) - 1, -1/theta)
        
        # Pairwise lower tail dependencies
        rsi_osc_tail = clayton_lower_tail(u_rsi + 0.01, u_osc + 0.01, theta)
        rsi_z_tail = clayton_lower_tail(u_rsi + 0.01, u_zscore + 0.01, theta)
        osc_z_tail = clayton_lower_tail(u_osc + 0.01, u_zscore + 0.01, theta)
        
        # Combined tail score (lower = both in lower tail = oversold together)
        tail_score = 1 - (rsi_osc_tail * 0.35 + rsi_z_tail * 0.30 + osc_z_tail * 0.35)
        
        # MARGINAL VALUES
        rsi_value = 1 - u_rsi  # Lower rank = higher value
        osc_value = 1 - u_osc
        z_value = 1 - u_zscore
        
        marginal_value = (rsi_value * 0.30 + osc_value * 0.40 + z_value * 0.30)
        
        # COPULA-ADJUSTED VALUE
        # High tail score + high marginal = strong signal
        copula_value = marginal_value * (0.6 + tail_score * 0.6)
        
        # WEEKLY CONFIRMATION (same copula approach)
        weekly_tail = clayton_lower_tail(u_rsi_w + 0.01, u_osc_w + 0.01, theta * 0.8)
        weekly_confirm = 1 - weekly_tail
        
        combined_value = copula_value * 0.65 + weekly_confirm * 0.35
        
        # MOMENTUM TURN
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        u_ema = to_uniform(ema_spread)
        turn_signal = u_ema  # Higher rank = more bullish turn
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        u_trend = to_uniform(ma_ratio)
        trend_signal = u_trend
        
        df['composite_score'] = combined_value * (0.4 + turn_signal * 0.8) * (0.5 + trend_signal * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class WaveletDenoiser(BaseStrategy):
    """
    Wavelet-inspired: separates signal from noise across scales.
    Low frequency (weekly) = trend signal.
    High frequency (daily-weekly diff) = noise to filter.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # DECOMPOSITION: Approximation (low freq) + Detail (high freq)
        # Weekly = approximation, Daily-Weekly = detail
        
        rsi_approx = df['rsi weekly']
        rsi_detail = df['rsi latest'] - df['rsi weekly']
        
        osc_approx = df['osc weekly']
        osc_detail = df['osc latest'] - df['osc weekly']
        
        z_approx = df['zscore weekly']
        z_detail = df['zscore latest'] - df['zscore weekly']
        
        # SOFT THRESHOLDING (denoise detail coefficients)
        def soft_threshold(detail, threshold):
            """Shrink small details to zero (noise), keep large (signal)"""
            sign = np.sign(detail)
            magnitude = np.abs(detail)
            shrunk = np.maximum(magnitude - threshold, 0)
            return sign * shrunk
        
        # Adaptive thresholds based on cross-sectional MAD
        rsi_thresh = np.median(np.abs(rsi_detail)) * 1.5
        osc_thresh = np.median(np.abs(osc_detail)) * 1.2
        z_thresh = np.median(np.abs(z_detail)) * 1.3
        
        rsi_detail_clean = soft_threshold(rsi_detail, rsi_thresh)
        osc_detail_clean = soft_threshold(osc_detail, osc_thresh)
        z_detail_clean = soft_threshold(z_detail, z_thresh)
        
        # RECONSTRUCT DENOISED SIGNAL
        rsi_denoised = rsi_approx + rsi_detail_clean * 0.5  # Partial reconstruction
        osc_denoised = osc_approx + osc_detail_clean * 0.5
        z_denoised = z_approx + z_detail_clean * 0.5
        
        # VALUE FROM DENOISED SIGNALS
        rsi_value = 1 / (1 + np.exp((rsi_denoised - 38) * 0.14))
        osc_value = 1 / (1 + np.exp((osc_denoised + 38) * 0.028))
        z_value = 1 / (1 + np.exp((z_denoised + 1.3) * 0.65))
        
        denoised_value = (rsi_value * 0.30 + osc_value * 0.40 + z_value * 0.30)
        
        # DETAIL DIRECTION (positive detail = improving)
        detail_direction = np.tanh((rsi_detail_clean / 10 + osc_detail_clean / 20 + z_detail_clean / 1.5) / 3)
        detail_signal = (detail_direction + 1) / 2
        
        # EMA (denoised approach)
        ema_approx = df['9ema osc weekly'] - df['21ema osc weekly']
        ema_detail = (df['9ema osc latest'] - df['21ema osc latest']) - ema_approx
        ema_detail_clean = soft_threshold(ema_detail, np.median(np.abs(ema_detail)) * 1.2)
        ema_denoised = ema_approx + ema_detail_clean * 0.6
        turn_signal = 1 / (1 + np.exp(-ema_denoised / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_signal = 1 / (1 + np.exp(-(ma_ratio - 1) * 12))
        
        df['composite_score'] = denoised_value * (0.5 + detail_signal * 0.5) * (0.4 + turn_signal * 0.8) * (0.5 + trend_signal * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class GradientBoostBlender(BaseStrategy):
    """
    Gradient-boosting inspired: sequential correction of residuals.
    Each signal layer corrects errors from previous layer.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        learning_rate = 0.3
        
        # LAYER 0: Base prediction (weekly-based, stable)
        base_pred = 1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12))
        
        # LAYER 1: RSI residual correction
        rsi_signal = 1 / (1 + np.exp((df['rsi latest'] - 38) * 0.15))
        residual_1 = rsi_signal - base_pred
        pred_1 = base_pred + learning_rate * residual_1
        
        # LAYER 2: OSC residual correction
        osc_signal = 1 / (1 + np.exp((df['osc latest'] + 42) * 0.03))
        residual_2 = osc_signal - pred_1
        pred_2 = pred_1 + learning_rate * residual_2
        
        # LAYER 3: Z-score residual correction
        z_signal = 1 / (1 + np.exp((df['zscore latest'] + 1.4) * 0.7))
        residual_3 = z_signal - pred_2
        pred_3 = pred_2 + learning_rate * residual_3
        
        # LAYER 4: Weekly confirmation correction
        weekly_signal = (1 / (1 + np.exp((df['osc weekly'] + 38) * 0.025)) +
                        1 / (1 + np.exp((df['zscore weekly'] + 1.2) * 0.6))) / 2
        residual_4 = weekly_signal - pred_3
        pred_4 = pred_3 + learning_rate * 0.5 * residual_4  # Lower LR for weekly
        
        # LAYER 5: EMA momentum correction
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_signal = 1 / (1 + np.exp(-ema_spread / 28))
        residual_5 = ema_signal - pred_4
        final_pred = pred_4 + learning_rate * 0.4 * residual_5
        
        # Clip to valid range
        final_pred = np.clip(final_pred, 0.01, 0.99)
        
        # TREND MULTIPLIER
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = final_pred * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class AttentionMechanism(BaseStrategy):
    """
    Attention-inspired: dynamically weights signals based on their relevance.
    Signals with stronger extremes get more attention.
    Cross-signal attention for context.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # SIGNAL VALUES (queries)
        rsi_val = 1 / (1 + np.exp((df['rsi latest'] - 37) * 0.15))
        osc_val = 1 / (1 + np.exp((df['osc latest'] + 42) * 0.032))
        z_val = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.7))
        rsi_w_val = 1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12))
        osc_w_val = 1 / (1 + np.exp((df['osc weekly'] + 38) * 0.028))
        
        # ATTENTION SCORES (based on signal strength/extremity)
        # More extreme = higher attention
        rsi_attention = np.abs(df['rsi latest'] - 50) / 50
        osc_attention = np.abs(df['osc latest']) / 100
        z_attention = np.abs(df['zscore latest']) / 3
        rsi_w_attention = np.abs(df['rsi weekly'] - 50) / 50
        osc_w_attention = np.abs(df['osc weekly']) / 100
        
        # SOFTMAX over attention scores
        attention_scores = np.column_stack([rsi_attention, osc_attention, z_attention, 
                                           rsi_w_attention, osc_w_attention])
        # Temperature for softmax sharpness
        temperature = 0.5
        exp_scores = np.exp(attention_scores / temperature)
        softmax_weights = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-6)
        
        # WEIGHTED VALUE (attention-weighted)
        values = np.column_stack([rsi_val, osc_val, z_val, rsi_w_val, osc_w_val])
        attended_value = np.sum(values * softmax_weights, axis=1)
        
        # CROSS-ATTENTION (signals attending to each other)
        # RSI-OSC agreement boosts both
        rsi_osc_agreement = np.tanh(rsi_val * osc_val * 3)
        # Daily-Weekly agreement
        daily_weekly_agreement = np.tanh((rsi_val * rsi_w_val + osc_val * osc_w_val) * 2)
        
        cross_attention_boost = (rsi_osc_agreement * 0.5 + daily_weekly_agreement * 0.5)
        
        # COMBINED
        attention_output = attended_value * (1 + cross_attention_boost * 0.4)
        
        # EMA MOMENTUM (with attention)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_attention = np.abs(ema_spread) / 50
        ema_val = 1 / (1 + np.exp(-ema_spread / 28))
        
        # Combine with momentum attention
        final_with_momentum = attention_output * 0.7 + ema_val * 0.3 * (1 + ema_attention * 0.5)
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = final_with_momentum * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class EnsembleVotingStrategy(BaseStrategy):
    """
    Ensemble of multiple mini-strategies with soft voting.
    Each sub-strategy votes, final score is weighted vote.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # SUB-STRATEGY 1: Pure value (oversold depth)
        value_vote = (
            1 / (1 + np.exp((df['rsi latest'] - 35) * 0.16)) * 0.30 +
            1 / (1 + np.exp((df['osc latest'] + 45) * 0.035)) * 0.40 +
            1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.75)) * 0.30
        )
        
        # SUB-STRATEGY 2: Momentum turn
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_weekly = df['9ema osc weekly'] - df['21ema osc weekly']
        momentum_vote = (
            1 / (1 + np.exp(-ema_spread / 26)) * 0.50 +
            1 / (1 + np.exp(-ema_weekly / 32)) * 0.30 +
            1 / (1 + np.exp(-(df['osc latest'] - df['osc weekly']) / 25)) * 0.20
        )
        
        # SUB-STRATEGY 3: Weekly confirmation
        weekly_vote = (
            1 / (1 + np.exp((df['rsi weekly'] - 40) * 0.13)) * 0.35 +
            1 / (1 + np.exp((df['osc weekly'] + 38) * 0.028)) * 0.40 +
            1 / (1 + np.exp((df['zscore weekly'] + 1.2) * 0.65)) * 0.25
        )
        
        # SUB-STRATEGY 4: Trend quality
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        price_ratio = df['price'] / (df['ma200 latest'] + 1e-6)
        trend_vote = (
            1 / (1 + np.exp(-(ma_ratio - 0.98) * 15)) * 0.50 +
            1 / (1 + np.exp(-(price_ratio - 0.92) * 12)) * 0.50
        )
        
        # SUB-STRATEGY 5: Volatility quality
        vol = df['dev20 latest'] / df['price']
        vol_vote = 1 / (1 + np.exp((vol - 0.025) * 80))
        
        # VOTE WEIGHTS (can be adaptive)
        # More weight to value and momentum in oversold markets
        avg_rsi = df['rsi latest'].mean()
        if avg_rsi < 40:
            weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        elif avg_rsi > 55:
            weights = np.array([0.20, 0.25, 0.15, 0.30, 0.10])
        else:
            weights = np.array([0.25, 0.25, 0.20, 0.20, 0.10])
        
        # WEIGHTED VOTE
        votes = np.column_stack([value_vote, momentum_vote, weekly_vote, trend_vote, vol_vote])
        ensemble_score = np.sum(votes * weights, axis=1)
        
        # AGREEMENT BONUS (when sub-strategies agree)
        vote_std = np.std(votes, axis=1)
        agreement_bonus = 1 / (1 + vote_std * 5)  # Lower std = more agreement
        
        df['composite_score'] = ensemble_score * (0.8 + agreement_bonus * 0.4)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)

# =========================================================================
# MORE SOPHISTICATED STRATEGIES - V6
# Novel quantitative approaches continued
# =========================================================================

class OptimalTransportBlender(BaseStrategy):
    """
    Optimal transport inspired: finds minimum cost mapping between 
    current state and ideal oversold state. Distance from ideal = opportunity.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # IDEAL OVERSOLD STATE (target distribution)
        ideal_rsi = 25
        ideal_osc = -70
        ideal_zscore = -2.0
        ideal_rsi_w = 30
        ideal_osc_w = -55
        
        # CURRENT STATE normalized
        rsi_norm = df['rsi latest'] / 100
        osc_norm = (df['osc latest'] + 100) / 200
        z_norm = (df['zscore latest'] + 3) / 6
        rsi_w_norm = df['rsi weekly'] / 100
        osc_w_norm = (df['osc weekly'] + 100) / 200
        
        ideal_rsi_norm = ideal_rsi / 100
        ideal_osc_norm = (ideal_osc + 100) / 200
        ideal_z_norm = (ideal_zscore + 3) / 6
        ideal_rsi_w_norm = ideal_rsi_w / 100
        ideal_osc_w_norm = (ideal_osc_w + 100) / 200
        
        # WASSERSTEIN-LIKE DISTANCE (L2 in normalized space)
        dist_rsi = np.abs(rsi_norm - ideal_rsi_norm)
        dist_osc = np.abs(osc_norm - ideal_osc_norm)
        dist_z = np.abs(z_norm - ideal_z_norm)
        dist_rsi_w = np.abs(rsi_w_norm - ideal_rsi_w_norm)
        dist_osc_w = np.abs(osc_w_norm - ideal_osc_w_norm)
        
        # Weighted distance (closer to ideal = lower distance = higher score)
        total_distance = np.sqrt(
            (dist_rsi ** 2) * 0.20 +
            (dist_osc ** 2) * 0.25 +
            (dist_z ** 2) * 0.20 +
            (dist_rsi_w ** 2) * 0.15 +
            (dist_osc_w ** 2) * 0.20
        )
        
        # Convert distance to score (inverse with softmax-like transform)
        proximity_score = np.exp(-total_distance * 4)
        
        # DIRECTION BONUS (moving toward ideal)
        moving_toward = (
            ((df['rsi latest'] < df['rsi weekly']) & (df['rsi latest'] < 45)).astype(float) * 0.3 +
            ((df['osc latest'] > df['osc weekly']) & (df['osc latest'] < -30)).astype(float) * 0.4 +
            ((df['zscore latest'] > df['zscore weekly']) & (df['zscore latest'] < -1)).astype(float) * 0.3
        )
        direction_mult = 1 + moving_toward * 0.5
        
        # EMA TURN
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.94) * 7) * 0.3 + 1.0
        
        df['composite_score'] = proximity_score * direction_mult * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class StochasticDominance(BaseStrategy):
    """
    First-order stochastic dominance: prefers stocks that dominate 
    others across all oversold metrics. Pareto-optimal selection.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # RANK on each dimension (lower rank = more oversold = better)
        rsi_rank = df['rsi latest'].rank(pct=True)
        osc_rank = df['osc latest'].rank(pct=True)
        z_rank = df['zscore latest'].rank(pct=True)
        rsi_w_rank = df['rsi weekly'].rank(pct=True)
        osc_w_rank = df['osc weekly'].rank(pct=True)
        
        # DOMINANCE COUNT: how many metrics is this stock in bottom quartile?
        bottom_q = 0.30
        dominance_count = (
            (rsi_rank < bottom_q).astype(float) +
            (osc_rank < bottom_q).astype(float) +
            (z_rank < bottom_q).astype(float) +
            (rsi_w_rank < bottom_q).astype(float) +
            (osc_w_rank < bottom_q).astype(float)
        )
        
        # AVERAGE RANK (lower = better)
        avg_rank = (rsi_rank * 0.20 + osc_rank * 0.25 + z_rank * 0.20 + 
                   rsi_w_rank * 0.15 + osc_w_rank * 0.20)
        
        # DOMINANCE SCORE: combination of count and average
        # Exponential boost for high dominance count
        dominance_score = np.exp(dominance_count * 0.5) * (1 - avg_rank)
        
        # PARETO EFFICIENCY: check if dominated by others
        # Simplified: penalize if consistently worse than median
        median_rank = 0.5
        inefficiency = np.maximum(0, avg_rank - median_rank)
        pareto_mult = 1 / (1 + inefficiency * 3)
        
        # MOMENTUM TURN
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        ema_rank = ema_spread.rank(pct=True)
        turn_signal = ema_rank  # Higher rank = better turn
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_rank = ma_ratio.rank(pct=True)
        
        df['composite_score'] = dominance_score * pareto_mult * (0.4 + turn_signal * 0.8) * (0.5 + trend_rank * 0.6)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class MaximumEntropyStrategy(BaseStrategy):
    """
    Maximum entropy principle: among all portfolios consistent with 
    constraints, choose the one with maximum entropy (least assumptions).
    Constraints: expected value signal, expected turn signal.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # FEATURE FUNCTIONS (constraints we want to satisfy)
        f1_value = 1 / (1 + np.exp((df['rsi latest'] - 38) * 0.14))
        f2_osc = 1 / (1 + np.exp((df['osc latest'] + 42) * 0.03))
        f3_zscore = 1 / (1 + np.exp((df['zscore latest'] + 1.4) * 0.7))
        f4_turn = 1 / (1 + np.exp(-(df['9ema osc latest'] - df['21ema osc latest']) / 28))
        f5_trend = 1 / (1 + np.exp(-(df['ma90 latest'] / df['ma200 latest'] - 1) * 12))
        
        # LAGRANGE MULTIPLIERS (learned from typical good portfolio)
        # These weight the importance of each constraint
        lambda1 = 1.2  # Value importance
        lambda2 = 1.5  # OSC importance
        lambda3 = 1.0  # Z-score importance
        lambda4 = 0.8  # Turn importance
        lambda5 = 0.6  # Trend importance
        
        # MAX ENTROPY DISTRIBUTION: p(x) ∝ exp(Σ λ_i f_i(x))
        log_unnorm = (lambda1 * f1_value + lambda2 * f2_osc + lambda3 * f3_zscore + 
                     lambda4 * f4_turn + lambda5 * f5_trend)
        
        # Exponentiate and normalize
        unnorm_prob = np.exp(log_unnorm - log_unnorm.max())  # Subtract max for stability
        
        # ENTROPY REGULARIZATION
        # Add small uniform component to increase entropy
        uniform = np.ones(len(df)) / len(df)
        entropy_weight = 0.1
        regularized = unnorm_prob * (1 - entropy_weight) + uniform * entropy_weight * len(df)
        
        # WEEKLY CONFIRMATION
        f_weekly = (1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) +
                   1 / (1 + np.exp((df['osc weekly'] + 38) * 0.025))) / 2
        weekly_mult = 0.7 + f_weekly * 0.5
        
        df['composite_score'] = regularized * weekly_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class HiddenMarkovModel(BaseStrategy):
    """
    HMM inspired: infers hidden momentum state from observable indicators.
    States: Accumulation, Neutral, Distribution.
    Transition probabilities from indicator changes.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # EMISSION PROBABILITIES P(observation | state)
        # State 0: Accumulation (oversold, turning up)
        # State 1: Neutral
        # State 2: Distribution (overbought, turning down)
        
        # Emission for Accumulation state
        emit_accum_rsi = np.exp(-((df['rsi latest'] - 30) ** 2) / 200)
        emit_accum_osc = np.exp(-((df['osc latest'] + 60) ** 2) / 800)
        emit_accum_turn = 1 / (1 + np.exp(-(df['9ema osc latest'] - df['21ema osc latest']) / 25))
        p_emit_accum = (emit_accum_rsi * emit_accum_osc * emit_accum_turn) ** (1/3)
        
        # Emission for Neutral state
        emit_neutral_rsi = np.exp(-((df['rsi latest'] - 50) ** 2) / 300)
        emit_neutral_osc = np.exp(-((df['osc latest']) ** 2) / 1000)
        p_emit_neutral = (emit_neutral_rsi * emit_neutral_osc) ** 0.5
        
        # Emission for Distribution state
        emit_dist_rsi = np.exp(-((df['rsi latest'] - 70) ** 2) / 200)
        emit_dist_osc = np.exp(-((df['osc latest'] - 60) ** 2) / 800)
        emit_dist_turn = 1 / (1 + np.exp((df['9ema osc latest'] - df['21ema osc latest']) / 25))
        p_emit_dist = (emit_dist_rsi * emit_dist_osc * emit_dist_turn) ** (1/3)
        
        # PRIOR STATE PROBABILITIES (from weekly data)
        prior_accum = 1 / (1 + np.exp((df['rsi weekly'] - 35) * 0.12))
        prior_dist = 1 / (1 + np.exp(-(df['rsi weekly'] - 65) * 0.12))
        prior_neutral = 1 - prior_accum - prior_dist
        prior_neutral = np.clip(prior_neutral, 0.1, 0.8)
        
        # Normalize priors
        prior_sum = prior_accum + prior_neutral + prior_dist
        prior_accum /= prior_sum
        prior_neutral /= prior_sum
        prior_dist /= prior_sum
        
        # POSTERIOR (Bayes: prior * likelihood)
        post_accum = prior_accum * p_emit_accum
        post_neutral = prior_neutral * p_emit_neutral
        post_dist = prior_dist * p_emit_dist
        
        # Normalize posterior
        post_sum = post_accum + post_neutral + post_dist + 1e-6
        post_accum /= post_sum
        
        # SCORE: probability of being in accumulation state
        accumulation_prob = post_accum
        
        # TREND MULTIPLIER
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.94) * 7) * 0.3 + 1.0
        
        df['composite_score'] = accumulation_prob * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class QuantileRegressionStrategy(BaseStrategy):
    """
    Quantile regression inspired: estimates conditional quantiles.
    Scores based on how extreme current values are relative to 
    expected quantiles given other indicators.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # CONDITIONAL QUANTILE ESTIMATION
        # Expected RSI given OSC level (linear approximation)
        osc_norm = df['osc latest'] / 100
        expected_rsi_median = 50 + osc_norm * 20  # Higher OSC -> higher RSI expected
        expected_rsi_q10 = 35 + osc_norm * 15
        
        # How far below expected is actual RSI?
        rsi_surprise = (expected_rsi_median - df['rsi latest']) / 20
        rsi_extreme = (expected_rsi_q10 - df['rsi latest']) / 15
        
        # Combined surprise score (more surprising = more oversold than expected)
        rsi_quantile_score = np.tanh(rsi_surprise * 0.8) * 0.5 + np.tanh(rsi_extreme * 0.5) * 0.5
        rsi_quantile_score = (rsi_quantile_score + 1) / 2  # Map to 0-1
        
        # OSC quantile given Z-score
        z_norm = df['zscore latest'] / 3
        expected_osc_median = -20 + z_norm * 40
        osc_surprise = (expected_osc_median - df['osc latest']) / 40
        osc_quantile_score = (np.tanh(osc_surprise * 0.6) + 1) / 2
        
        # Z-score quantile given RSI
        rsi_norm = (df['rsi latest'] - 50) / 50
        expected_z_median = rsi_norm * 1.5
        z_surprise = (expected_z_median - df['zscore latest']) / 2
        z_quantile_score = (np.tanh(z_surprise * 0.7) + 1) / 2
        
        # COMBINED QUANTILE SURPRISE
        quantile_score = (rsi_quantile_score * 0.35 + 
                         osc_quantile_score * 0.35 + 
                         z_quantile_score * 0.30)
        
        # WEEKLY CONFIRMATION
        weekly_rsi_surprise = (45 - df['rsi weekly']) / 20
        weekly_osc_surprise = (-30 - df['osc weekly']) / 40
        weekly_score = (np.tanh(weekly_rsi_surprise * 0.5) + np.tanh(weekly_osc_surprise * 0.4) + 2) / 4
        
        combined = quantile_score * 0.65 + weekly_score * 0.35
        
        # TURN SIGNAL
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class MutualInformationBlender(BaseStrategy):
    """
    Mutual information inspired: weights signals by their information 
    content about future returns. Higher MI = more predictive = higher weight.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # PROXY FOR MUTUAL INFORMATION
        # Signals that are extreme (high entropy in signal) carry more info
        def entropy_proxy(series, bins=10):
            """Higher variance in extreme regions = higher info"""
            percentiles = series.rank(pct=True)
            extreme_mask = (percentiles < 0.2) | (percentiles > 0.8)
            extreme_var = series[extreme_mask].var() if extreme_mask.sum() > 2 else series.var()
            total_var = series.var()
            return extreme_var / (total_var + 1e-6)
        
        # Calculate MI proxies for each signal
        mi_rsi = entropy_proxy(df['rsi latest'])
        mi_osc = entropy_proxy(df['osc latest'])
        mi_zscore = entropy_proxy(df['zscore latest'])
        mi_rsi_w = entropy_proxy(df['rsi weekly'])
        mi_osc_w = entropy_proxy(df['osc weekly'])
        
        # Normalize to weights
        total_mi = mi_rsi + mi_osc + mi_zscore + mi_rsi_w + mi_osc_w + 1e-6
        w_rsi = mi_rsi / total_mi
        w_osc = mi_osc / total_mi
        w_zscore = mi_zscore / total_mi
        w_rsi_w = mi_rsi_w / total_mi
        w_osc_w = mi_osc_w / total_mi
        
        # SIGNAL VALUES
        v_rsi = 1 / (1 + np.exp((df['rsi latest'] - 36) * 0.15))
        v_osc = 1 / (1 + np.exp((df['osc latest'] + 44) * 0.032))
        v_zscore = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.72))
        v_rsi_w = 1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12))
        v_osc_w = 1 / (1 + np.exp((df['osc weekly'] + 38) * 0.028))
        
        # MI-WEIGHTED COMBINATION
        mi_weighted = (v_rsi * w_rsi + v_osc * w_osc + v_zscore * w_zscore +
                      v_rsi_w * w_rsi_w + v_osc_w * w_osc_w)
        
        # CONDITIONAL MI (daily given weekly)
        # If daily is surprising given weekly, it has more info
        rsi_cond = np.abs(df['rsi latest'] - df['rsi weekly']) / 20
        osc_cond = np.abs(df['osc latest'] - df['osc weekly']) / 40
        cond_mi_boost = 1 + np.tanh((rsi_cond + osc_cond) / 2) * 0.3
        
        # But only if moving in right direction
        right_direction = ((df['rsi latest'] > df['rsi weekly']) & (df['rsi weekly'] < 45)) | \
                         ((df['osc latest'] > df['osc weekly']) & (df['osc weekly'] < -30))
        cond_mi_boost = np.where(right_direction, cond_mi_boost, 1.0)
        
        combined = mi_weighted * cond_mi_boost
        
        # TURN SIGNAL
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 27))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class GameTheoreticStrategy(BaseStrategy):
    """
    Game theory inspired: models market as game between buyers/sellers.
    Nash equilibrium approximation for optimal position.
    Payoff matrix based on indicator states.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # PLAYER STATES
        # Buyer strength (accumulation pressure)
        buyer_strength = (
            (1 / (1 + np.exp((df['rsi latest'] - 35) * 0.15))) * 0.30 +
            (1 / (1 + np.exp((df['osc latest'] + 50) * 0.03))) * 0.40 +
            (df['9ema osc latest'] > df['21ema osc latest']).astype(float) * 0.30
        )
        
        # Seller exhaustion (selling pressure declining)
        seller_exhaustion = (
            (df['osc latest'] > df['osc weekly']).astype(float) * 0.40 +
            (df['rsi latest'] > df['rsi weekly']).astype(float) * 0.30 +
            (df['zscore latest'] > df['zscore weekly']).astype(float) * 0.30
        )
        
        # PAYOFF MATRIX APPROXIMATION
        # Buyer payoff when buying at oversold with seller exhaustion
        buyer_payoff = buyer_strength * (0.5 + seller_exhaustion * 0.7)
        
        # Risk-adjusted payoff (lower vol = better payoff)
        vol = df['dev20 latest'] / df['price']
        vol_adj = 1 / (1 + vol * 30)
        risk_adj_payoff = buyer_payoff * (0.7 + vol_adj * 0.5)
        
        # NASH EQUILIBRIUM PROXY
        # Equilibrium when both sides have incentive to trade
        # Buyer incentive: oversold + turning
        # Seller incentive: selling at higher levels
        buyer_incentive = buyer_strength * seller_exhaustion
        
        # Market clearing (implied by both having incentive)
        clearing_score = np.sqrt(buyer_incentive * risk_adj_payoff)
        
        # WEEKLY CONFIRMATION
        weekly_buyer = (
            (1 / (1 + np.exp((df['rsi weekly'] - 40) * 0.12))) * 0.50 +
            (1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025))) * 0.50
        )
        weekly_mult = 0.6 + weekly_buyer * 0.6
        
        combined = clearing_score * weekly_mult
        
        # TREND (structural advantage)
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.94) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class ReinforcementLearningInspired(BaseStrategy):
    """
    RL inspired: state-action-reward framework.
    State = indicator values. Action = allocation weight.
    Value function approximation for expected cumulative reward.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # STATE FEATURES (normalized)
        s_rsi = (df['rsi latest'] - 50) / 50
        s_osc = df['osc latest'] / 100
        s_zscore = df['zscore latest'] / 3
        s_ema_spread = (df['9ema osc latest'] - df['21ema osc latest']) / 50
        s_trend = (df['ma90 latest'] / df['ma200 latest']) - 1
        s_rsi_w = (df['rsi weekly'] - 50) / 50
        s_osc_w = df['osc weekly'] / 100
        
        # VALUE FUNCTION APPROXIMATION V(s) = w^T * φ(s)
        # φ = feature transform (radial basis functions for oversold states)
        
        # RBF centered at oversold state
        oversold_center = np.array([-0.3, -0.5, -0.5, 0.1, 0.0])  # [rsi, osc, z, ema, trend]
        
        state_vec = np.column_stack([s_rsi, s_osc, s_zscore, s_ema_spread, s_trend])
        dist_to_oversold = np.sqrt(np.sum((state_vec - oversold_center) ** 2, axis=1))
        
        # RBF kernel
        gamma = 1.5
        rbf_oversold = np.exp(-gamma * dist_to_oversold)
        
        # RBF for turning state
        turning_center = np.array([-0.2, -0.4, -0.4, 0.3, 0.05])
        dist_to_turning = np.sqrt(np.sum((state_vec - turning_center) ** 2, axis=1))
        rbf_turning = np.exp(-gamma * dist_to_turning)
        
        # VALUE FUNCTION (weighted RBFs)
        w_oversold = 1.5
        w_turning = 2.0
        value_estimate = w_oversold * rbf_oversold + w_turning * rbf_turning
        
        # TEMPORAL DIFFERENCE (reward prediction)
        # Immediate reward proxy: weekly confirmation
        immediate_reward = (
            (1 / (1 + np.exp(s_rsi_w * 3))) * 0.50 +
            (1 / (1 + np.exp(s_osc_w * 2))) * 0.50
        )
        
        # BELLMAN-LIKE UPDATE
        discount = 0.9
        q_value = immediate_reward + discount * value_estimate
        
        # EXPLORATION BONUS (less visited states = bonus)
        # Proxy: how unusual is this state?
        state_unusualness = np.abs(s_rsi) + np.abs(s_osc) + np.abs(s_zscore)
        exploration_bonus = np.tanh(state_unusualness * 0.5) * 0.2
        
        df['composite_score'] = q_value + exploration_bonus
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class SpectralClusteringStrategy(BaseStrategy):
    """
    Spectral clustering inspired: groups stocks by indicator similarity.
    Prefers stocks in oversold cluster with momentum turn.
    Uses affinity matrix and Laplacian eigenvector properties.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # FEATURE VECTORS (normalized)
        features = np.column_stack([
            df['rsi latest'] / 100,
            (df['osc latest'] + 100) / 200,
            (df['zscore latest'] + 3) / 6,
            df['rsi weekly'] / 100,
            (df['osc weekly'] + 100) / 200
        ])
        
        # CLUSTER CENTER for "oversold with turn"
        oversold_center = np.array([0.30, 0.20, 0.25, 0.35, 0.25])
        
        # DISTANCE TO CLUSTER CENTER
        dist_to_cluster = np.sqrt(np.sum((features - oversold_center) ** 2, axis=1))
        
        # AFFINITY to oversold cluster (Gaussian kernel)
        sigma = 0.3
        affinity = np.exp(-dist_to_cluster ** 2 / (2 * sigma ** 2))
        
        # DEGREE (sum of affinities - higher = more connected)
        # Simplified: assume connection to cluster center only
        degree = affinity
        
        # LAPLACIAN SCORE (approximation)
        # Lower Laplacian score for oversold stocks = better
        laplacian_score = degree * affinity
        
        # WITHIN-CLUSTER QUALITY
        # How consistent are daily and weekly?
        daily_weekly_consistency = 1 - np.abs(features[:, 0] - features[:, 3]) - \
                                   np.abs(features[:, 1] - features[:, 4])
        daily_weekly_consistency = np.clip(daily_weekly_consistency, 0, 1)
        
        cluster_quality = laplacian_score * (0.6 + daily_weekly_consistency * 0.6)
        
        # MOMENTUM TURN
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = cluster_quality * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class CausalInferenceStrategy(BaseStrategy):
    """
    Causal inference inspired: estimates causal effect of oversold on returns.
    Propensity scoring to isolate treatment effect.
    Treatment = being in oversold state.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # TREATMENT: Being oversold
        # Propensity P(oversold | covariates)
        propensity = (
            1 / (1 + np.exp((df['rsi latest'] - 38) * 0.14)) * 0.35 +
            1 / (1 + np.exp((df['osc latest'] + 45) * 0.03)) * 0.40 +
            1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.7)) * 0.25
        )
        
        # TREATMENT INTENSITY (degree of oversold)
        treatment_intensity = (
            np.clip((40 - df['rsi latest']) / 25, 0, 1.5) * 0.30 +
            np.clip((-df['osc latest'] - 35) / 50, 0, 1.5) * 0.40 +
            np.clip(-df['zscore latest'] / 2.5, 0, 1.5) * 0.30
        )
        
        # CONFOUNDERS (things that affect both treatment and outcome)
        # Trend is a confounder: affects oversold probability and returns
        trend_confounder = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        vol_confounder = df['dev20 latest'] / df['price']
        
        # INVERSE PROPENSITY WEIGHTING
        # Weight by 1/propensity to debias (with clipping for stability)
        ipw = 1 / np.clip(propensity, 0.1, 0.9)
        ipw_normalized = ipw / ipw.mean()
        
        # CAUSAL EFFECT ESTIMATE
        # E[Y | treated] adjusted for confounders
        # Using doubly robust-like combination
        outcome_model = treatment_intensity * (0.8 + np.tanh((trend_confounder - 1) * 5) * 0.3)
        
        # Treatment effect with IPW
        causal_effect = outcome_model * np.sqrt(ipw_normalized)
        
        # TURN SIGNAL (not confounder, but effect modifier)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_modifier = 1 / (1 + np.exp(-ema_spread / 26))
        
        # Effect modification
        modified_effect = causal_effect * (0.4 + turn_modifier * 0.8)
        
        # WEEKLY CONFIRMATION
        weekly_treatment = (
            1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025)) * 0.50
        )
        
        df['composite_score'] = modified_effect * (0.6 + weekly_treatment * 0.5)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


# =========================================================================
# 13 MORE SOPHISTICATED STRATEGIES - V7
# Final batch of novel quantitative approaches
# =========================================================================

class BootstrapConfidenceStrategy(BaseStrategy):
    """
    Bootstrap inspired: estimates confidence intervals for signal strength.
    Higher confidence in oversold signal = higher weight.
    Uses resampling logic via cross-indicator agreement.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # POINT ESTIMATES for oversold
        est_rsi = 1 / (1 + np.exp((df['rsi latest'] - 36) * 0.15))
        est_osc = 1 / (1 + np.exp((df['osc latest'] + 45) * 0.032))
        est_z = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.72))
        est_rsi_w = 1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12))
        est_osc_w = 1 / (1 + np.exp((df['osc weekly'] + 40) * 0.028))
        
        # BOOTSTRAP VARIANCE PROXY (disagreement between estimates)
        estimates = np.column_stack([est_rsi, est_osc, est_z, est_rsi_w, est_osc_w])
        point_estimate = np.mean(estimates, axis=1)
        bootstrap_std = np.std(estimates, axis=1)
        
        # CONFIDENCE INTERVAL WIDTH
        ci_width = 2 * bootstrap_std  # Approximate 95% CI
        
        # CONFIDENCE SCORE (narrow CI = high confidence)
        confidence = 1 / (1 + ci_width * 5)
        
        # LOWER BOUND OF CI (conservative estimate)
        lower_bound = point_estimate - bootstrap_std
        lower_bound = np.clip(lower_bound, 0, 1)
        
        # COMBINED: high point estimate AND high confidence
        bootstrap_score = point_estimate * (0.5 + confidence * 0.7)
        
        # TURN SIGNAL
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 27))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = bootstrap_score * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class KernelDensityStrategy(BaseStrategy):
    """
    KDE inspired: estimates probability density of being in favorable state.
    Uses Gaussian kernels centered at favorable indicator values.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # KERNEL CENTERS (favorable states)
        # Multiple kernels for different favorable scenarios
        
        # Kernel 1: Deep oversold with turn
        k1_rsi, k1_osc, k1_z = 28, -65, -2.0
        bw1 = 0.3  # Bandwidth
        
        dist1 = np.sqrt(
            ((df['rsi latest'] - k1_rsi) / 30) ** 2 +
            ((df['osc latest'] - k1_osc) / 50) ** 2 +
            ((df['zscore latest'] - k1_z) / 2) ** 2
        )
        density1 = np.exp(-dist1 ** 2 / (2 * bw1 ** 2))
        
        # Kernel 2: Moderate oversold with strong turn
        k2_rsi, k2_osc, k2_ema = 35, -45, 15
        bw2 = 0.35
        
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        dist2 = np.sqrt(
            ((df['rsi latest'] - k2_rsi) / 30) ** 2 +
            ((df['osc latest'] - k2_osc) / 50) ** 2 +
            ((ema_spread - k2_ema) / 40) ** 2
        )
        density2 = np.exp(-dist2 ** 2 / (2 * bw2 ** 2))
        
        # Kernel 3: Weekly confirmation
        k3_rsi_w, k3_osc_w = 35, -50
        bw3 = 0.4
        
        dist3 = np.sqrt(
            ((df['rsi weekly'] - k3_rsi_w) / 30) ** 2 +
            ((df['osc weekly'] - k3_osc_w) / 50) ** 2
        )
        density3 = np.exp(-dist3 ** 2 / (2 * bw3 ** 2))
        
        # MIXTURE DENSITY
        weights = np.array([0.40, 0.35, 0.25])
        mixture_density = density1 * weights[0] + density2 * weights[1] + density3 * weights[2]
        
        # TURN SIGNAL
        turn_signal = 1 / (1 + np.exp(-ema_spread / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = mixture_density * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class SurvivalAnalysisStrategy(BaseStrategy):
    """
    Survival analysis inspired: models time-to-recovery from oversold.
    Hazard rate = probability of recovery given current state.
    Higher hazard = closer to recovery = better entry.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # BASELINE HAZARD (deeper oversold = been down longer = closer to recovery)
        baseline_hazard = (
            np.clip((40 - df['rsi latest']) / 30, 0, 1.5) * 0.30 +
            np.clip((-df['osc latest'] - 30) / 60, 0, 1.5) * 0.40 +
            np.clip(-df['zscore latest'] / 2.5, 0, 1.5) * 0.30
        )
        
        # COVARIATE EFFECTS (Cox proportional hazards style)
        # Turn signal increases hazard (closer to event)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_covariate = np.tanh(ema_spread / 30) * 0.5 + 0.5
        
        # Daily improving vs weekly increases hazard
        improvement_covariate = np.where(
            (df['osc latest'] > df['osc weekly']) & (df['rsi latest'] > df['rsi weekly']),
            1.5,
            np.where(
                df['osc latest'] > df['osc weekly'],
                1.2,
                0.8
            )
        )
        
        # Weekly confirmation
        weekly_covariate = (
            1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025)) * 0.50
        )
        
        # HAZARD RATE (multiplicative model)
        hazard_rate = baseline_hazard * (0.5 + turn_covariate * 0.7) * improvement_covariate * (0.6 + weekly_covariate * 0.5)
        
        # SURVIVAL SCORE (high hazard = good)
        survival_score = hazard_rate
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = survival_score * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class PrincipalComponentStrategy(BaseStrategy):
    """
    PCA inspired: projects indicators onto principal components.
    First PC captures common oversold factor. Scores by projection.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # STANDARDIZE FEATURES
        def standardize(x):
            return (x - x.mean()) / (x.std() + 1e-6)
        
        f_rsi = standardize(df['rsi latest'])
        f_osc = standardize(df['osc latest'])
        f_z = standardize(df['zscore latest'])
        f_rsi_w = standardize(df['rsi weekly'])
        f_osc_w = standardize(df['osc weekly'])
        f_ema = standardize(df['9ema osc latest'] - df['21ema osc latest'])
        
        # APPROXIMATE PC1 (oversold factor)
        # Loadings: negative for RSI/OSC/Z (lower = more oversold)
        pc1_loadings = np.array([-0.40, -0.45, -0.35, -0.35, -0.40, 0.30])
        pc1_loadings = pc1_loadings / np.linalg.norm(pc1_loadings)
        
        features = np.column_stack([f_rsi, f_osc, f_z, f_rsi_w, f_osc_w, f_ema])
        pc1_score = np.dot(features, pc1_loadings)
        
        # APPROXIMATE PC2 (momentum factor)
        pc2_loadings = np.array([0.20, 0.25, 0.15, -0.15, -0.20, 0.55])
        pc2_loadings = pc2_loadings / np.linalg.norm(pc2_loadings)
        pc2_score = np.dot(features, pc2_loadings)
        
        # COMBINED SCORE (high PC1 = oversold, high PC2 = turning)
        # Transform to positive scores
        pc1_transformed = (pc1_score - pc1_score.min()) / (pc1_score.max() - pc1_score.min() + 1e-6)
        pc2_transformed = (pc2_score - pc2_score.min()) / (pc2_score.max() - pc2_score.min() + 1e-6)
        
        pca_score = pc1_transformed * 0.6 + pc2_transformed * 0.4
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = pca_score * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class FactorMomentumStrategy(BaseStrategy):
    """
    Factor investing inspired: constructs value and momentum factors.
    Long value + momentum intersection.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # VALUE FACTOR (oversold = cheap)
        value_components = np.column_stack([
            1 - df['rsi latest'].rank(pct=True),
            1 - df['osc latest'].rank(pct=True),
            1 - df['zscore latest'].rank(pct=True)
        ])
        value_factor = np.mean(value_components, axis=1)
        
        # MOMENTUM FACTOR (turning + trend)
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        momentum_components = np.column_stack([
            ema_spread.rank(pct=True),
            (df['osc latest'] - df['osc weekly']).rank(pct=True),
            (df['rsi latest'] - df['rsi weekly']).rank(pct=True),
            (df['price'] / df['ma90 latest']).rank(pct=True)
        ])
        momentum_factor = np.mean(momentum_components, axis=1)
        
        # QUALITY FACTOR (weekly confirmation + trend)
        quality_components = np.column_stack([
            1 - df['rsi weekly'].rank(pct=True),
            1 - df['osc weekly'].rank(pct=True),
            (df['ma90 latest'] / df['ma200 latest']).rank(pct=True)
        ])
        quality_factor = np.mean(quality_components, axis=1)
        
        # FACTOR COMBINATION (value + momentum with quality tilt)
        combined_factor = (
            value_factor * 0.40 +
            momentum_factor * 0.35 +
            quality_factor * 0.25
        )
        
        # NON-LINEAR TRANSFORM (emphasize extremes)
        factor_score = np.power(combined_factor, 1.3)
        
        # VOLATILITY ADJUSTMENT
        vol = df['dev20 latest'] / df['price']
        vol_adj = 1 / (1 + vol * 25)
        
        df['composite_score'] = factor_score * (0.7 + vol_adj * 0.5)
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class ElasticNetBlender(BaseStrategy):
    """
    Elastic net inspired: L1 + L2 regularized combination of signals.
    Balances sparsity (few strong signals) and ridge (all signals matter).
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # SIGNAL VALUES
        s1 = 1 / (1 + np.exp((df['rsi latest'] - 36) * 0.15))
        s2 = 1 / (1 + np.exp((df['osc latest'] + 44) * 0.032))
        s3 = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.72))
        s4 = 1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12))
        s5 = 1 / (1 + np.exp((df['osc weekly'] + 40) * 0.028))
        
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        s6 = 1 / (1 + np.exp(-ema_spread / 28))
        
        signals = np.column_stack([s1, s2, s3, s4, s5, s6])
        
        # L1 COMPONENT (sparsity - emphasize strongest signal)
        max_signal = np.max(signals, axis=1)
        l1_score = max_signal
        
        # L2 COMPONENT (ridge - average of squared signals)
        l2_score = np.sqrt(np.mean(signals ** 2, axis=1))
        
        # ELASTIC NET COMBINATION
        alpha = 0.4  # Mixing parameter (0 = ridge, 1 = lasso)
        elastic_score = alpha * l1_score + (1 - alpha) * l2_score
        
        # SIGNAL AGREEMENT BONUS
        signal_std = np.std(signals, axis=1)
        agreement_bonus = 1 / (1 + signal_std * 4)
        
        combined = elastic_score * (0.7 + agreement_bonus * 0.5)
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class RobustRegressionStrategy(BaseStrategy):
    """
    Robust regression inspired: uses Huber loss for outlier resistance.
    Scores based on robust distance from favorable state.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # TARGET STATE (ideal oversold with turn)
        target_rsi = 30
        target_osc = -55
        target_z = -1.8
        target_ema = 10
        
        # RESIDUALS
        r_rsi = (df['rsi latest'] - target_rsi) / 30
        r_osc = (df['osc latest'] - target_osc) / 50
        r_z = (df['zscore latest'] - target_z) / 2
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        r_ema = (ema_spread - target_ema) / 40
        
        # HUBER LOSS (robust to outliers)
        def huber_loss(r, delta=1.0):
            abs_r = np.abs(r)
            return np.where(abs_r <= delta, 
                           0.5 * r ** 2, 
                           delta * (abs_r - 0.5 * delta))
        
        delta = 0.8
        loss_rsi = huber_loss(r_rsi, delta)
        loss_osc = huber_loss(r_osc, delta)
        loss_z = huber_loss(r_z, delta)
        loss_ema = huber_loss(r_ema, delta)
        
        # TOTAL ROBUST LOSS (lower = closer to target)
        total_loss = (loss_rsi * 0.25 + loss_osc * 0.30 + 
                     loss_z * 0.25 + loss_ema * 0.20)
        
        # CONVERT TO SCORE (lower loss = higher score)
        robust_score = np.exp(-total_loss * 2)
        
        # WEEKLY CONFIRMATION
        weekly_score = (
            1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025)) * 0.50
        )
        
        combined = robust_score * (0.6 + weekly_score * 0.5)
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class ConvexOptimizationStrategy(BaseStrategy):
    """
    Convex optimization inspired: maximizes concave objective.
    Objective: expected return - risk penalty.
    Uses log-barrier for constraints.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # EXPECTED RETURN (concave - log for diminishing returns)
        raw_return = (
            np.clip((40 - df['rsi latest']) / 30, 0, 2) * 0.30 +
            np.clip((-df['osc latest'] - 30) / 60, 0, 2) * 0.40 +
            np.clip(-df['zscore latest'] / 2.5, 0, 2) * 0.30
        )
        exp_return = np.log1p(raw_return * 2)  # Concave transform
        
        # RISK PENALTY (convex)
        vol = df['dev20 latest'] / df['price']
        uncertainty = np.abs(df['rsi latest'] - df['rsi weekly']) / 30
        risk = vol * 10 + uncertainty
        risk_penalty = risk ** 2 * 0.5  # Quadratic penalty
        
        # TURN BONUS
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_bonus = np.log1p(np.clip(ema_spread / 30 + 0.5, 0.1, 2))
        
        # OBJECTIVE: return - risk + turn
        objective = exp_return - risk_penalty + turn_bonus * 0.5
        
        # SHIFT TO POSITIVE
        objective = objective - objective.min() + 0.1
        
        # WEEKLY CONSTRAINT (log-barrier style)
        weekly_feasibility = (
            1 / (1 + np.exp((df['rsi weekly'] - 50) * 0.1)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 20) * 0.02)) * 0.50
        )
        barrier = -np.log(weekly_feasibility + 0.1) * 0.1
        
        final_objective = objective - barrier
        final_objective = np.clip(final_objective, 0.01, None)
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = final_objective * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class MonteCarloStrategy(BaseStrategy):
    """
    Monte Carlo inspired: simulates scenarios via indicator perturbations.
    Score = expected value across simulated scenarios.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # BASE SCENARIO (current state)
        def score_state(rsi, osc, z, ema_spread):
            v_rsi = 1 / (1 + np.exp((rsi - 37) * 0.15))
            v_osc = 1 / (1 + np.exp((osc + 45) * 0.032))
            v_z = 1 / (1 + np.exp((z + 1.5) * 0.72))
            v_turn = 1 / (1 + np.exp(-ema_spread / 28))
            return (v_rsi * 0.25 + v_osc * 0.35 + v_z * 0.20 + v_turn * 0.20)
        
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        base_score = score_state(df['rsi latest'], df['osc latest'], 
                                 df['zscore latest'], ema_spread)
        
        # SIMULATED SCENARIOS (deterministic perturbations)
        # Scenario 1: RSI improves
        s1 = score_state(df['rsi latest'] + 5, df['osc latest'], 
                        df['zscore latest'], ema_spread)
        
        # Scenario 2: OSC improves
        s2 = score_state(df['rsi latest'], df['osc latest'] + 10, 
                        df['zscore latest'], ema_spread)
        
        # Scenario 3: Both improve
        s3 = score_state(df['rsi latest'] + 3, df['osc latest'] + 7, 
                        df['zscore latest'] + 0.3, ema_spread + 5)
        
        # Scenario 4: Deterioration
        s4 = score_state(df['rsi latest'] - 5, df['osc latest'] - 10, 
                        df['zscore latest'] - 0.3, ema_spread - 5)
        
        # EXPECTED VALUE (weighted by probability)
        # Improvement more likely if turning
        turn_prob = 1 / (1 + np.exp(-ema_spread / 20))
        
        expected_score = (
            base_score * 0.30 +
            s1 * 0.15 * turn_prob +
            s2 * 0.20 * turn_prob +
            s3 * 0.25 * turn_prob +
            s4 * 0.10 * (1 - turn_prob)
        )
        
        # Normalize
        expected_score = expected_score / (0.30 + 0.15 * turn_prob + 0.20 * turn_prob + 
                                           0.25 * turn_prob + 0.10 * (1 - turn_prob))
        
        # WEEKLY CONFIRMATION
        weekly_score = (
            1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025)) * 0.50
        )
        
        combined = expected_score * (0.6 + weekly_score * 0.5)
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class VariationalInferenceStrategy(BaseStrategy):
    """
    Variational inference inspired: approximates posterior over momentum state.
    Uses ELBO-like objective: expected log-likelihood + entropy.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # VARIATIONAL PARAMETERS (approximate posterior mean and variance)
        # Mean: point estimates of oversold probability
        q_mean_rsi = 1 / (1 + np.exp((df['rsi latest'] - 36) * 0.15))
        q_mean_osc = 1 / (1 + np.exp((df['osc latest'] + 45) * 0.032))
        q_mean_z = 1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.72))
        
        # Variance: uncertainty from daily-weekly disagreement
        q_var_rsi = np.abs(df['rsi latest'] - df['rsi weekly']) / 50
        q_var_osc = np.abs(df['osc latest'] - df['osc weekly']) / 80
        q_var_z = np.abs(df['zscore latest'] - df['zscore weekly']) / 3
        
        # EXPECTED LOG-LIKELIHOOD (higher when confident and oversold)
        ell_rsi = q_mean_rsi * np.log(q_mean_rsi + 1e-6) + (1 - q_mean_rsi) * np.log(1 - q_mean_rsi + 1e-6)
        ell_osc = q_mean_osc * np.log(q_mean_osc + 1e-6) + (1 - q_mean_osc) * np.log(1 - q_mean_osc + 1e-6)
        
        # Normalize (less negative = better)
        ell = -(ell_rsi + ell_osc) / 2
        ell = (ell - ell.min()) / (ell.max() - ell.min() + 1e-6)
        
        # ENTROPY (prefer moderate uncertainty - not too confident, not too uncertain)
        total_var = (q_var_rsi + q_var_osc + q_var_z) / 3
        entropy_score = 1 / (1 + np.abs(total_var - 0.15) * 10)  # Peak at moderate variance
        
        # ELBO-LIKE SCORE
        elbo = ell * 0.7 + entropy_score * 0.3
        
        # PRIOR (weekly - KL divergence term)
        prior_score = (
            1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025)) * 0.50
        )
        
        # Combined (ELBO + prior)
        combined = elbo * 0.65 + prior_score * 0.35
        
        # TURN SIGNAL
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class NeuralNetworkInspired(BaseStrategy):
    """
    Neural network inspired: multi-layer transform with activations.
    Input layer -> Hidden layers with ReLU/tanh -> Output layer.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # INPUT LAYER (normalized features)
        x1 = (50 - df['rsi latest']) / 50  # Inverted RSI
        x2 = -df['osc latest'] / 100  # Inverted OSC
        x3 = -df['zscore latest'] / 3
        x4 = (50 - df['rsi weekly']) / 50
        x5 = -df['osc weekly'] / 100
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        x6 = ema_spread / 50
        x7 = (df['ma90 latest'] / df['ma200 latest'] - 1) * 5
        
        inputs = np.column_stack([x1, x2, x3, x4, x5, x6, x7])
        
        # HIDDEN LAYER 1 (7 -> 4) with ReLU
        W1 = np.array([
            [0.3, 0.4, 0.3, 0.2, 0.3, 0.2, 0.1],
            [0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.2],
            [0.1, 0.2, 0.2, 0.1, 0.2, 0.5, 0.3],
            [0.2, 0.2, 0.1, 0.2, 0.1, 0.4, 0.5]
        ])
        b1 = np.array([-0.3, -0.4, -0.2, -0.3])
        
        h1 = np.maximum(0, np.dot(inputs, W1.T) + b1)  # ReLU
        
        # HIDDEN LAYER 2 (4 -> 3) with tanh
        W2 = np.array([
            [0.5, 0.3, 0.4, 0.3],
            [0.3, 0.5, 0.3, 0.4],
            [0.2, 0.2, 0.5, 0.4]
        ])
        b2 = np.array([0.0, 0.0, -0.1])
        
        h2 = np.tanh(np.dot(h1, W2.T) + b2)
        
        # OUTPUT LAYER (3 -> 1) with sigmoid
        W3 = np.array([[0.4, 0.4, 0.3]])
        b3 = np.array([0.0])
        
        output = 1 / (1 + np.exp(-(np.dot(h2, W3.T) + b3)))
        output = output.flatten()
        
        # SKIP CONNECTION (residual)
        skip = (x1 + x2 + x3) / 3
        skip = (skip - skip.min()) / (skip.max() - skip.min() + 1e-6)
        
        final_output = output * 0.7 + skip * 0.3
        
        df['composite_score'] = final_output
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class GraphNeuralInspired(BaseStrategy):
    """
    Graph neural network inspired: stocks as nodes, indicator similarity as edges.
    Message passing to aggregate neighborhood information.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        n = len(df)
        
        # NODE FEATURES
        node_features = np.column_stack([
            1 / (1 + np.exp((df['rsi latest'] - 36) * 0.15)),
            1 / (1 + np.exp((df['osc latest'] + 45) * 0.032)),
            1 / (1 + np.exp((df['zscore latest'] + 1.5) * 0.72))
        ])
        
        # Initial node score
        node_score = np.mean(node_features, axis=1)
        
        # ADJACENCY (similarity-based, approximate without full matrix)
        # Use cross-sectional percentile similarity
        rsi_pct = df['rsi latest'].rank(pct=True).values
        osc_pct = df['osc latest'].rank(pct=True).values
        
        # MESSAGE PASSING (aggregate from similar nodes)
        # Approximate: boost if in same quintile as other oversold stocks
        bottom_quintile_mask = (rsi_pct < 0.25) & (osc_pct < 0.25)
        n_bottom = bottom_quintile_mask.sum()
        
        # Neighborhood score: average of bottom quintile
        if n_bottom > 0:
            neighborhood_score = node_score[bottom_quintile_mask].mean()
        else:
            neighborhood_score = node_score.mean()
        
        # AGGREGATION (node + neighborhood)
        # Nodes in bottom quintile get boosted by neighborhood
        aggregated = np.where(
            bottom_quintile_mask,
            node_score * 0.6 + neighborhood_score * 0.4,
            node_score * 0.8 + neighborhood_score * 0.2
        )
        
        # READOUT (attention over nodes)
        attention_weights = np.exp(aggregated * 2)
        attention_weights = attention_weights / attention_weights.sum()
        
        # Reweight by attention
        graph_score = aggregated * (1 + attention_weights * n * 0.1)
        
        # TURN SIGNAL
        ema_spread = df['9ema osc latest'] - df['21ema osc latest']
        turn_signal = 1 / (1 + np.exp(-ema_spread / 28))
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = graph_score * (0.4 + turn_signal * 0.8) * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)


class ContrastiveLearningStrategy(BaseStrategy):
    """
    Contrastive learning inspired: maximizes similarity to positive examples,
    minimizes similarity to negative examples.
    Positive = oversold with turn. Negative = overbought declining.
    """
    
    def generate_portfolio(self, df: pd.DataFrame, sip_amount: float = 100000.0) -> pd.DataFrame:
        required_columns = [
            'symbol', 'price', 'rsi latest', 'rsi weekly', 'osc latest', 'osc weekly',
            '9ema osc latest', '9ema osc weekly', '21ema osc latest', '21ema osc weekly',
            'zscore latest', 'zscore weekly', 'ma90 latest', 'ma200 latest',
            'ma90 weekly', 'ma200 weekly', 'dev20 latest', 'dev20 weekly',
            'ma20 latest', 'ma20 weekly'
        ]
        df = self._clean_data(df, required_columns)
        
        # EMBEDDING (normalized features)
        emb = np.column_stack([
            df['rsi latest'] / 100,
            (df['osc latest'] + 100) / 200,
            (df['zscore latest'] + 3) / 6,
            (df['9ema osc latest'] - df['21ema osc latest'] + 50) / 100
        ])
        
        # POSITIVE ANCHOR (ideal oversold with turn)
        pos_anchor = np.array([0.28, 0.18, 0.22, 0.62])
        
        # NEGATIVE ANCHOR (overbought declining)
        neg_anchor = np.array([0.72, 0.75, 0.70, 0.35])
        
        # SIMILARITY (cosine-like, using dot product)
        def similarity(x, anchor):
            dot = np.sum(x * anchor, axis=1)
            norm_x = np.sqrt(np.sum(x ** 2, axis=1))
            norm_a = np.sqrt(np.sum(anchor ** 2))
            return dot / (norm_x * norm_a + 1e-6)
        
        sim_pos = similarity(emb, pos_anchor)
        sim_neg = similarity(emb, neg_anchor)
        
        # CONTRASTIVE SCORE (InfoNCE-like)
        temperature = 0.5
        exp_pos = np.exp(sim_pos / temperature)
        exp_neg = np.exp(sim_neg / temperature)
        
        contrastive_score = exp_pos / (exp_pos + exp_neg + 1e-6)
        
        # WEEKLY CONFIRMATION
        weekly_score = (
            1 / (1 + np.exp((df['rsi weekly'] - 42) * 0.12)) * 0.50 +
            1 / (1 + np.exp((df['osc weekly'] + 40) * 0.025)) * 0.50
        )
        
        combined = contrastive_score * (0.6 + weekly_score * 0.5)
        
        # TREND
        ma_ratio = df['ma90 latest'] / (df['ma200 latest'] + 1e-6)
        trend_mult = np.tanh((ma_ratio - 0.95) * 7) * 0.3 + 1.0
        
        df['composite_score'] = combined * trend_mult
        df['composite_score'] = df['composite_score'] + 1e-6
        
        total_score = df['composite_score'].sum()
        df['weightage'] = df['composite_score'] / total_score if total_score > 0 else 1 / len(df)
        return self._allocate_portfolio(df, sip_amount)

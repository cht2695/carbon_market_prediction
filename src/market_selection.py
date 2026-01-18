"""
Market Selection for Touring
=============================

Goal: Identify the next best DMAs for an artist to play, balancing:
1. Expected demand (profit prediction)
2. Price sensitivity
3. Audience fit

Approach:
- Profit Model: XGBoost predicting log_revenue from streams, historical performance, market size
- Baseline Comparison: Simple prev_log_revenue to validate XGBoost improvement
- Cross-validation: Time-series CV to prevent overfitting

Key Features:
| Feature          | Correlation | Source         | Reasoning                        |
|------------------|-------------|----------------|----------------------------------|
| log_streams      | 0.77        | mstreams_week  | Current artist demand signal     |
| prev_log_revenue | 0.77        | ticket history | Historical performance in DMA    |
| log_followers    | 0.53        | Instagram      | Artist reach/fanbase size        |
| prev_avg_price   | 0.47        | ticket data    | Price point indicator            |
| log_dma_tickets  | —           | DMA aggregate  | Market size control              |

DMA Size Tiers (fixed thresholds):
| Tier   | Threshold        | Count | Examples                    |
|--------|------------------|-------|-----------------------------|
| Small  | < 5,000 tickets  | 96    | Rochester MN, Youngstown OH |
| Medium | 5,000 - 25,000   | 45    | Madison WI, Omaha NE        |
| Large  | 25,000 - 100,000 | 27    | Salt Lake City, St. Louis   |
| Major  | >= 100,000       | 26    | NYC, LA, Boston, Chicago    |
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_cleaning import run_cleaning_pipeline
from feature_exploration import (
    create_event_level_data,
    add_streaming_to_events,
    add_prev_event_features,
    create_dma_features,
    create_demographic_features,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, Dict]:
    """Load and prepare all data for market selection."""
    import io
    from contextlib import redirect_stdout

    print("Loading and cleaning data...")
    with redirect_stdout(io.StringIO()):
        cleaned = run_cleaning_pipeline(data_dir)

    print("Creating event-level dataset...")
    event_data = create_event_level_data(cleaned)
    event_data = add_streaming_to_events(event_data, cleaned)
    event_data = add_prev_event_features(event_data)

    print("Creating DMA and artist features...")
    dma_features = create_dma_features(event_data)
    artist_demo = create_demographic_features(cleaned['instagram'])

    # Merge features
    df = event_data.merge(
        dma_features[['dma_id', 'dma_name', 'log_dma_tickets', 'dma_size_tier',
                      'dma_total_tickets', 'dma_event_count']],
        on=['dma_id', 'dma_name'], how='left'
    )
    df = df.merge(
        artist_demo[['artist_id', 'num_followers', 'log_followers', 'engagement_rate']],
        on='artist_id', how='left'
    )

    return df, {'dma_features': dma_features, 'artist_demo': artist_demo, 'cleaned': cleaned}


def show_dma_tier_distribution(dma_features: pd.DataFrame):
    """Display DMA size tier distribution with examples."""
    print("\n" + "="*70)
    print("DMA SIZE TIER DISTRIBUTION")
    print("="*70)

    print("\nTier counts:")
    tier_counts = dma_features['dma_size_tier'].value_counts().reindex(['Small', 'Medium', 'Large', 'Major'])
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count}")

    print("\nThresholds:")
    print("  Small:  < 5,000 tickets")
    print("  Medium: 5,000 - 25,000 tickets")
    print("  Large:  25,000 - 100,000 tickets")
    print("  Major:  >= 100,000 tickets")

    print("\nExamples per tier:")
    for tier in ['Small', 'Medium', 'Large', 'Major']:
        tier_dmas = dma_features[dma_features['dma_size_tier'] == tier].nlargest(3, 'dma_total_tickets')
        print(f"\n  {tier}:")
        for _, row in tier_dmas.iterrows():
            print(f"    {row['dma_name']}: {row['dma_total_tickets']:,.0f} tickets, {row['dma_event_count']} events")


# =============================================================================
# PROFIT PREDICTION MODELS
# =============================================================================

class BaselineModel:
    """Baseline: predict using only prev_log_revenue."""

    def __init__(self):
        self.mean_revenue = None

    def fit(self, df: pd.DataFrame) -> 'BaselineModel':
        self.mean_revenue = df['log_revenue'].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = df['prev_log_revenue'].values.copy()
        preds = np.where(np.isnan(preds), self.mean_revenue, preds)
        return preds


class XGBoostProfitModel:
    """XGBoost model with time-series CV to prevent overfitting."""

    FEATURE_COLS = ['log_streams', 'prev_log_revenue', 'log_followers',
                    'log_dma_tickets', 'prev_avg_price']

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.cv_results = None

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'XGBoostProfitModel':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")

        # Filter to complete cases and sort by time
        train_df = df.dropna(subset=self.FEATURE_COLS + ['log_revenue'])
        train_df = train_df.sort_values('event_start')

        X = train_df[self.FEATURE_COLS].values
        y = train_df['log_revenue'].values
        X_scaled = self.scaler.fit_transform(X)

        # Conservative XGBoost params to prevent overfitting
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
        }

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_r2, cv_rmse = [], []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = xgb.XGBRegressor(**params)
            fold_model.fit(X_tr, y_tr, verbose=False)
            y_pred = fold_model.predict(X_val)

            cv_r2.append(r2_score(y_val, y_pred))
            cv_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        self.cv_results = {
            'r2_mean': np.mean(cv_r2), 'r2_std': np.std(cv_r2),
            'rmse_mean': np.mean(cv_rmse), 'rmse_std': np.std(cv_rmse)
        }

        # Fit final model on all data
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_scaled, y, verbose=False)

        # Training metrics for overfitting check
        y_train_pred = self.model.predict(X_scaled)
        self.train_r2 = r2_score(y, y_train_pred)
        self.train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
        self.n_train = len(train_df)

        if verbose:
            print(f"\nXGBoost trained on {self.n_train:,} events")
            print(f"  CV R²: {self.cv_results['r2_mean']:.4f} (±{self.cv_results['r2_std']:.4f})")
            print(f"  Train R²: {self.train_r2:.4f}")
            print(f"  Overfit gap: {self.train_r2 - self.cv_results['r2_mean']:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': self.FEATURE_COLS,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(df: pd.DataFrame, test_artists: List[str] = None):
    """Compare Baseline vs XGBoost and show XGBoost is better without overfitting."""

    print("\n" + "="*70)
    print("MODEL COMPARISON: BASELINE vs XGBOOST")
    print("="*70)

    # Filter to complete cases
    train_df = df.dropna(subset=XGBoostProfitModel.FEATURE_COLS + ['log_revenue'])
    train_df = train_df.sort_values('event_start')

    # Fit models
    print("\n--- BASELINE MODEL (prev_log_revenue only) ---")
    baseline = BaselineModel().fit(train_df)
    baseline_preds = baseline.predict(train_df)
    baseline_r2 = r2_score(train_df['log_revenue'], baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(train_df['log_revenue'], baseline_preds))
    print(f"  R²: {baseline_r2:.4f}")
    print(f"  RMSE: {baseline_rmse:.4f}")

    print("\n--- XGBOOST MODEL (all features) ---")
    xgb_model = XGBoostProfitModel().fit(train_df)
    xgb_preds = xgb_model.predict(train_df)

    print("\n--- FEATURE IMPORTANCE ---")
    importance = xgb_model.get_feature_importance()
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Overfitting check
    print("\n--- OVERFITTING CHECK ---")
    overfit_gap = xgb_model.train_r2 - xgb_model.cv_results['r2_mean']
    print(f"  Train R²: {xgb_model.train_r2:.4f}")
    print(f"  CV R²:    {xgb_model.cv_results['r2_mean']:.4f} (±{xgb_model.cv_results['r2_std']:.4f})")
    print(f"  Gap:      {overfit_gap:.4f} {'(OK - minimal overfitting)' if overfit_gap < 0.2 else '(WARNING - overfitting)'}")

    # Improvement summary
    print("\n--- IMPROVEMENT SUMMARY ---")
    improvement = xgb_model.cv_results['r2_mean'] - baseline_r2
    print(f"  Baseline R²: {baseline_r2:.4f}")
    print(f"  XGBoost CV R²: {xgb_model.cv_results['r2_mean']:.4f}")
    print(f"  Improvement: +{improvement:.4f} ({improvement/baseline_r2*100:.1f}% relative)")

    # Test on specific artists if provided
    if test_artists:
        print("\n" + "="*70)
        print("PER-ARTIST COMPARISON")
        print("="*70)

        for artist_name in test_artists:
            artist_df = train_df[train_df['artist_name'] == artist_name]
            if len(artist_df) == 0:
                print(f"\n{artist_name}: Not found or no complete data")
                continue

            actual = artist_df['log_revenue'].values
            actual_rev = np.expm1(actual)

            baseline_pred = baseline.predict(artist_df)
            baseline_rev = np.expm1(baseline_pred)
            baseline_mape = np.mean(np.abs(baseline_rev - actual_rev) / actual_rev) * 100

            xgb_pred = xgb_model.predict(artist_df)
            xgb_rev = np.expm1(xgb_pred)
            xgb_mape = np.mean(np.abs(xgb_rev - actual_rev) / actual_rev) * 100

            print(f"\n{artist_name} ({len(artist_df)} events):")
            print(f"  Baseline MAPE: {baseline_mape:.1f}%")
            print(f"  XGBoost MAPE:  {xgb_mape:.1f}%")
            print(f"  Improvement:   {baseline_mape - xgb_mape:+.1f}%")

            # Show top 5 DMAs
            artist_df = artist_df.copy()
            artist_df['actual_rev'] = actual_rev
            artist_df['baseline_rev'] = baseline_rev
            artist_df['xgb_rev'] = xgb_rev

            print(f"\n  {'DMA':<35} {'Actual':>12} {'Baseline':>12} {'XGBoost':>12}")
            print(f"  {'-'*75}")
            for _, row in artist_df.nlargest(5, 'actual_rev').iterrows():
                base_err = (row['baseline_rev'] - row['actual_rev']) / row['actual_rev'] * 100
                xgb_err = (row['xgb_rev'] - row['actual_rev']) / row['actual_rev'] * 100
                print(f"  {row['dma_name'][:35]:<35} ${row['actual_rev']:>10,.0f} "
                      f"${row['baseline_rev']:>10,.0f}({base_err:+.0f}%) "
                      f"${row['xgb_rev']:>10,.0f}({xgb_err:+.0f}%)")

    return baseline, xgb_model


# =============================================================================
# MARKET RANKER
# =============================================================================

class MarketSelector:
    """Rank DMAs for an artist based on predicted revenue."""

    def __init__(self, profit_model: XGBoostProfitModel, df: pd.DataFrame, dma_features: pd.DataFrame):
        self.model = profit_model
        self.df = df
        self.dma_features = dma_features

    def rank_dmas_for_artist(self, artist_name: str, top_n: int = 10) -> pd.DataFrame:
        """Rank all DMAs for a given artist."""
        artist_df = self.df[self.df['artist_name'] == artist_name]
        if len(artist_df) == 0:
            raise ValueError(f"Artist '{artist_name}' not found")

        artist_id = artist_df['artist_id'].iloc[0]

        # Get artist's average features
        avg_streams = artist_df['log_streams'].mean()
        avg_followers = artist_df['log_followers'].iloc[0]

        # Get previous performance by DMA
        prev_perf = artist_df.groupby('dma_id').agg({
            'log_revenue': 'last',
            'avg_price': 'last'
        }).rename(columns={'log_revenue': 'prev_log_revenue', 'avg_price': 'prev_avg_price'})

        # Score each DMA
        results = []
        for _, dma in self.dma_features.iterrows():
            dma_id = dma['dma_id']
            has_history = dma_id in prev_perf.index

            feature_row = pd.DataFrame([{
                'log_streams': avg_streams,
                'prev_log_revenue': prev_perf.loc[dma_id, 'prev_log_revenue'] if has_history else avg_streams,
                'log_followers': avg_followers,
                'log_dma_tickets': dma['log_dma_tickets'],
                'prev_avg_price': prev_perf.loc[dma_id, 'prev_avg_price'] if has_history else 100,
            }])

            pred_log_rev = self.model.predict(feature_row)[0]

            results.append({
                'dma_id': dma_id,
                'dma_name': dma['dma_name'],
                'dma_size_tier': dma['dma_size_tier'],
                'predicted_revenue': np.expm1(pred_log_rev),
                'has_history': has_history,
            })

        results_df = pd.DataFrame(results).sort_values('predicted_revenue', ascending=False)
        return results_df.head(top_n)


# =============================================================================
# MAIN
# =============================================================================

def main(data_dir: str = 'data'):
    """Run the complete market selection pipeline."""
    print("="*70)
    print("MARKET SELECTION FOR TOURING")
    print("="*70)

    # Prepare data
    df, extras = prepare_data(data_dir)
    print(f"\nDataset: {len(df):,} events, {df['artist_id'].nunique()} artists, {df['dma_id'].nunique()} DMAs")

    # Show DMA distribution
    show_dma_tier_distribution(extras['dma_features'])

    # Compare models
    test_artists = ['Beyoncé', 'Death Cab for Cutie']
    baseline, xgb_model = compare_models(df, test_artists)

    # Demo market selection
    print("\n" + "="*70)
    print("MARKET RECOMMENDATIONS")
    print("="*70)

    selector = MarketSelector(xgb_model, df, extras['dma_features'])

    for artist in test_artists:
        try:
            print(f"\n--- Top 10 DMAs for {artist} ---")
            recs = selector.rank_dmas_for_artist(artist, top_n=10)
            print(f"{'Rank':<5} {'DMA':<40} {'Tier':<8} {'Predicted Revenue':>18} {'History'}")
            print("-"*85)
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                hist = "★" if row['has_history'] else "○"
                print(f"{i:<5} {row['dma_name'][:40]:<40} {row['dma_size_tier']:<8} "
                      f"${row['predicted_revenue']:>16,.0f} {hist}")
        except Exception as e:
            print(f"Could not rank DMAs for {artist}: {e}")

    return df, xgb_model, selector


if __name__ == '__main__':
    df, model, selector = main()

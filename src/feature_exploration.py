"""
Feature Exploration for Market Selection Model
==============================================

This module documents the exploration process that led to feature selection
for the DMA market selection objective function.

Key Findings:
1. Aggregation must be at EVENT level (not weekly) - correlation jumps from 0 to 0.77
2. Profit features: log_streams, log_followers, avg_price, prev_revenue
3. Growth features: language_diversity (-), african_american_pct (+), engagement_rate (+)
4. Must stratify by artist size tier to see true demographic effects
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_cleaning import run_cleaning_pipeline


# =============================================================================
# 1. DATA LOADING AND EVENT AGGREGATION
# =============================================================================

def create_event_level_data(cleaned: Dict[str, pd.DataFrame], gap_weeks: int = 8) -> pd.DataFrame:
    """
    Aggregate ticket data to EVENT level.

    An "event" is defined as consecutive weeks of ticket activity.
    Gap > gap_weeks = new event.

    This is the KEY INSIGHT: weekly correlations are ~0,
    but event-level correlations are 0.77.
    """
    ticket = cleaned['ticket_dma_week'].copy()
    ticket['week_start_date'] = pd.to_datetime(ticket['week_start_date'])
    ticket = ticket.sort_values(['artist_id', 'dma_id', 'week_start_date'])

    # Assign event windows
    def assign_event_windows(group):
        group = group.sort_values('week_start_date').copy()
        group['days_since_last'] = group['week_start_date'].diff().dt.days
        group['new_event'] = (group['days_since_last'] > gap_weeks * 7) | (group['days_since_last'].isna())
        group['event_id'] = group['new_event'].cumsum()
        return group

    ticket = ticket.groupby(['artist_id', 'dma_id'], group_keys=False).apply(assign_event_windows)

    # Aggregate to event level
    event_agg = ticket.groupby(['artist_id', 'artist_name', 'dma_id', 'dma_name', 'genre', 'event_id']).agg({
        'st_num_tickets': 'sum',
        'st_order_value': 'sum',
        'week_start_date': ['min', 'max', 'count']
    }).reset_index()

    event_agg.columns = ['artist_id', 'artist_name', 'dma_id', 'dma_name', 'genre', 'event_id',
                         'total_tickets', 'total_revenue', 'event_start', 'event_end', 'num_weeks']

    event_agg['avg_price'] = event_agg['total_revenue'] / event_agg['total_tickets']
    event_agg['log_tickets'] = np.log1p(event_agg['total_tickets'])
    event_agg['log_revenue'] = np.log1p(event_agg['total_revenue'])

    return event_agg


def add_streaming_to_events(event_agg: pd.DataFrame, cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add streaming data aggregated to the same event time windows.

    For each event, sum streams during [event_start, event_end].
    """
    streams = cleaned['mstreams_week'].copy()
    streams['week_start_date'] = pd.to_datetime(streams['week_start_date'])

    # Aggregate streams by artist-week
    streams_agg = streams.groupby(['artist_id', 'week_start_date'])['number_of_streams'].sum().reset_index()

    # For each event, get total streams
    event_streams = []
    for _, event in event_agg.iterrows():
        mask = (streams_agg['artist_id'] == event['artist_id']) & \
               (streams_agg['week_start_date'] >= event['event_start']) & \
               (streams_agg['week_start_date'] <= event['event_end'])
        event_streams.append(streams_agg.loc[mask, 'number_of_streams'].sum())

    event_agg = event_agg.copy()
    event_agg['total_streams'] = event_streams
    event_agg['log_streams'] = np.log1p(event_agg['total_streams'])

    return event_agg


def add_prev_event_features(event_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from previous event for the same artist-DMA.

    prev_log_revenue is expected to be the best predictor of current revenue
    (same-category prediction).
    """
    event_agg = event_agg.sort_values(['artist_id', 'dma_id', 'event_start']).copy()

    # Group by artist-DMA and shift to get previous event
    event_agg['prev_log_revenue'] = event_agg.groupby(['artist_id', 'dma_id'])['log_revenue'].shift(1)
    event_agg['prev_log_tickets'] = event_agg.groupby(['artist_id', 'dma_id'])['log_tickets'].shift(1)
    event_agg['prev_log_streams'] = event_agg.groupby(['artist_id', 'dma_id'])['log_streams'].shift(1)
    event_agg['prev_avg_price'] = event_agg.groupby(['artist_id', 'dma_id'])['avg_price'].shift(1)

    return event_agg


# =============================================================================
# 2. CORRELATION VALIDATION: WHY EVENT LEVEL MATTERS
# =============================================================================

def validate_aggregation_scheme(cleaned: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Demonstrate that event-level aggregation reveals true correlations.

    Returns correlations at different aggregation levels.
    """
    # Weekly level (WRONG)
    ticket_weekly = cleaned['ticket_dma_week'].copy()
    streams_weekly = cleaned['mstreams_week'].copy()

    # Merge at weekly level
    ticket_weekly['week_start_date'] = pd.to_datetime(ticket_weekly['week_start_date'])
    streams_weekly['week_start_date'] = pd.to_datetime(streams_weekly['week_start_date'])

    streams_agg = streams_weekly.groupby(['artist_id', 'week_start_date'])['number_of_streams'].sum().reset_index()
    weekly_merged = ticket_weekly.merge(streams_agg, on=['artist_id', 'week_start_date'], how='inner')

    weekly_corr = weekly_merged['st_order_value'].corr(weekly_merged['number_of_streams'])
    weekly_log_corr = np.log1p(weekly_merged['st_order_value']).corr(np.log1p(weekly_merged['number_of_streams']))

    # Event level (CORRECT)
    event_agg = create_event_level_data(cleaned)
    event_agg = add_streaming_to_events(event_agg, cleaned)
    event_agg = event_agg[event_agg['total_streams'] > 0]

    event_corr = event_agg['total_revenue'].corr(event_agg['total_streams'])
    event_log_corr = event_agg['log_revenue'].corr(event_agg['log_streams'])

    return {
        'weekly_raw_corr': weekly_corr,
        'weekly_log_corr': weekly_log_corr,
        'event_raw_corr': event_corr,
        'event_log_corr': event_log_corr
    }


# =============================================================================
# 2.5 FEATURE CORRELATION CHECK
# =============================================================================

def check_feature_correlations(event_agg: pd.DataFrame) -> Dict[str, any]:
    """
    Check correlations between key features to identify multicollinearity.

    Key finding:
    - log_streams vs prev_log_revenue: r=0.58 (moderate, both useful)
    - log_streams vs log_revenue: r=0.75
    - prev_log_revenue vs log_revenue: r=0.77 (best single predictor)

    Both log_streams and prev_log_revenue can be used together since
    correlation is moderate (0.58), not high (>0.8).
    """
    # Filter to events with both features
    df = event_agg[
        event_agg['prev_log_revenue'].notna() &
        (event_agg['total_streams'] > 0)
    ].copy()

    # Key features for correlation matrix
    key_features = ['log_revenue', 'log_streams', 'prev_log_revenue',
                    'prev_log_streams', 'avg_price']

    corr_matrix = df[key_features].corr()

    # Specific correlation we care about
    streams_prev_rev_corr = df['log_streams'].corr(df['prev_log_revenue'])

    return {
        'correlation_matrix': corr_matrix,
        'log_streams_vs_prev_log_revenue': streams_prev_rev_corr,
        'n_events_with_both': len(df),
        'interpretation': (
            f"log_streams vs prev_log_revenue correlation: {streams_prev_rev_corr:.3f}\n"
            "This is moderate correlation - both features can be used together.\n"
            "They capture different information:\n"
            "  - log_streams: current artist popularity/demand signal\n"
            "  - prev_log_revenue: historical performance in THIS specific DMA"
        )
    }


# =============================================================================
# 3. PROFIT FEATURES ANALYSIS
# =============================================================================

def analyze_profit_features(event_agg: pd.DataFrame, instagram: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which features predict event-level revenue.

    Target: log_revenue

    Key findings:
    - log_streams: r=0.77 (strongest)
    - log_spotify_followers: r=0.61
    - avg_price: r=0.47
    - prev_log_revenue: ~0.7 (same-category best predictor)
    """
    # Merge Instagram demographics
    df = event_agg.merge(
        instagram[['artist_id', 'num_followers', 'engagement_rate', 'audience_credibility']],
        on='artist_id', how='left'
    )
    df['log_followers'] = np.log1p(df['num_followers'])

    # Calculate correlations with log_revenue
    target = 'log_revenue'
    features = ['log_streams', 'log_followers', 'avg_price', 'prev_log_revenue',
                'prev_log_streams', 'num_weeks', 'engagement_rate']

    results = []
    for feat in features:
        if feat in df.columns:
            valid = df[feat].notna() & df[target].notna()
            if valid.sum() > 100:
                corr = df.loc[valid, feat].corr(df.loc[valid, target])
                results.append({'feature': feat, 'correlation': corr, 'n': valid.sum()})

    return pd.DataFrame(results).sort_values('correlation', key=abs, ascending=False)


# =============================================================================
# 4. GROWTH FEATURES ANALYSIS (STRATIFIED)
# =============================================================================

def calculate_artist_growth(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate year-over-year stream growth at artist level.

    Formula: log(streams_recent_12w) - log(streams_same_period_last_year)
    """
    streams = cleaned['mstreams_week'].copy()
    streams['week_start_date'] = pd.to_datetime(streams['week_start_date'])

    streams_agg = streams.groupby(['artist_id', 'week_start_date'])['number_of_streams'].sum().reset_index()
    streams_agg = streams_agg.sort_values(['artist_id', 'week_start_date'])

    artist_growth = []
    for artist_id, group in streams_agg.groupby('artist_id'):
        group = group.sort_values('week_start_date')
        if len(group) >= 52:
            recent = np.log1p(group['number_of_streams'].tail(12)).mean()
            prior = np.log1p(group['number_of_streams'].iloc[-52:-40]).mean()
            growth = recent - prior
            artist_growth.append({
                'artist_id': artist_id,
                'stream_growth_yoy': growth,
                'recent_avg_streams': group['number_of_streams'].tail(12).mean(),
                'prior_avg_streams': group['number_of_streams'].iloc[-52:-40].mean()
            })

    return pd.DataFrame(artist_growth)


def create_demographic_features(instagram: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful demographic features from Instagram data.

    Note: Instagram demographic columns are PROPORTIONS (0-1), not counts.
    """
    df = instagram.copy()

    # Age composition
    df['gen_z_pct'] = df['ages_13_17'] + df['ages_18_24']
    df['millennial_pct'] = df['ages_25_34']
    df['older_pct'] = df['ages_45_64'] + df['ages_65_']

    # Racial/ethnic composition
    df['hispanic_pct'] = df['follower_hispanic']
    df['african_american_pct'] = df['follower_african_american']
    df['asian_pct'] = df['follower_asian']
    df['white_pct'] = df['follower_white']

    # Language
    df['spanish_pct'] = df['follower_lang_es']
    df['non_english_pct'] = 1 - df['follower_lang_en'].fillna(0)

    # Language diversity (Shannon entropy)
    lang_cols = ['follower_lang_en', 'follower_lang_es', 'follower_lang_pt', 'follower_lang_de',
                 'follower_lang_fr', 'follower_lang_zh', 'follower_lang_id', 'follower_lang_ru']

    def lang_entropy(row):
        props = [row[c] for c in lang_cols if c in row.index and pd.notna(row[c]) and row[c] > 0]
        if len(props) == 0:
            return 0
        return -sum(p * np.log(p) for p in props if p > 0)

    df['language_diversity'] = df.apply(lang_entropy, axis=1)

    # Gender
    df['female_pct'] = df['female_amount']

    # Engagement & authenticity
    df['engagement_rate_pct'] = df['engagement_rate']
    df['influencer_reach'] = df['notable_users_ratio']
    df['authentic_audience'] = df['audience_credibility']

    # Size
    df['log_followers'] = np.log1p(df['num_followers'])

    return df


def analyze_growth_stratified(growth_df: pd.DataFrame, demo_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze growth drivers STRATIFIED by artist size tier.

    Key insight: Must control for size because:
    - Bigger artists decline more (regression to mean): r=-0.19
    - Gen-Z correlates with size: r=+0.17
    - Raw correlations are confounded

    Stratified findings:
    - SMALL: older_pct positive, gen_z_pct negative (legacy rediscovery)
    - MEDIUM: african_american_pct positive (hip-hop growth), language_diversity negative
    - LARGE: engagement_rate positive, influencer_reach positive
    """
    analysis = growth_df.merge(demo_df, on='artist_id', how='inner')
    analysis['size_tier'] = pd.qcut(analysis['num_followers'], q=3, labels=['Small', 'Medium', 'Large'])

    features = ['gen_z_pct', 'millennial_pct', 'older_pct',
                'hispanic_pct', 'african_american_pct', 'asian_pct',
                'female_pct', 'non_english_pct', 'language_diversity',
                'engagement_rate_pct', 'influencer_reach', 'authentic_audience']

    results = {}
    for tier in ['Small', 'Medium', 'Large']:
        tier_data = analysis[analysis['size_tier'] == tier]
        tier_results = []

        for feat in features:
            if feat in tier_data.columns:
                corr = tier_data[feat].corr(tier_data['stream_growth_yoy'])
                tier_results.append({'feature': feat, 'correlation': corr})

        results[tier] = pd.DataFrame(tier_results).sort_values('correlation', key=abs, ascending=False)

    return results


# =============================================================================
# 5. DMA-LEVEL FEATURES
# =============================================================================

def create_dma_features(event_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Create DMA-specific features for market selection.

    Features:
    - dma_total_tickets: Total tickets sold in this DMA (proxy for market size)
    - dma_total_revenue: Total revenue in this DMA
    - dma_artist_count: Number of artists who played this DMA
    - dma_event_count: Number of events in this DMA
    - dma_avg_price: Average ticket price in this DMA
    - log_dma_tickets: Log-transformed total tickets (continuous size control)
    - dma_size_tier: Categorical tier based on meaningful ticket thresholds

    DMA Size Tier Thresholds (based on total_tickets):
    - Small: < 5,000 tickets (below median, ~96 DMAs)
    - Medium: 5,000 - 25,000 tickets (~45 DMAs)
    - Large: 25,000 - 100,000 tickets (~27 DMAs)
    - Major: >= 100,000 tickets (top ~26 DMAs - NYC, LA, Boston, etc.)
    """
    # DMA-level aggregates including event count
    dma_stats = event_agg.groupby(['dma_id', 'dma_name']).agg({
        'total_tickets': 'sum',
        'total_revenue': 'sum',
        'artist_id': 'nunique',
        'event_id': 'count',
        'avg_price': 'mean'
    }).reset_index()
    dma_stats.columns = ['dma_id', 'dma_name', 'dma_total_tickets', 'dma_total_revenue',
                         'dma_artist_count', 'dma_event_count', 'dma_avg_price']

    dma_stats['log_dma_tickets'] = np.log1p(dma_stats['dma_total_tickets'])
    dma_stats['log_dma_revenue'] = np.log1p(dma_stats['dma_total_revenue'])

    # DMA size tier based on meaningful ticket thresholds
    def assign_dma_tier(tickets):
        if tickets < 5000:
            return 'Small'
        elif tickets < 25000:
            return 'Medium'
        elif tickets < 100000:
            return 'Large'
        else:
            return 'Major'

    dma_stats['dma_size_tier'] = dma_stats['dma_total_tickets'].apply(assign_dma_tier)

    return dma_stats


def analyze_dma_sparsity(event_agg: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze DMA sparsity to determine appropriate control strategy.

    Key finding:
    - Many DMAs have very few events, making DMA fixed effects unreliable
    - 95 DMAs with < 10 events
    - 69 DMAs with < 5 events
    - Recommendation: Use DMA Size Tier (4 levels) instead of DMA fixed effects
    """
    dma_event_counts = event_agg.groupby('dma_id').size()

    sparsity_analysis = {
        'total_dmas': len(dma_event_counts),
        'dmas_lt_5_events': (dma_event_counts < 5).sum(),
        'dmas_lt_10_events': (dma_event_counts < 10).sum(),
        'dmas_lt_20_events': (dma_event_counts < 20).sum(),
        'min_events': dma_event_counts.min(),
        'max_events': dma_event_counts.max(),
        'median_events': dma_event_counts.median(),
        'event_count_distribution': dma_event_counts.describe(),
        'recommendation': (
            "Use DMA Size Tier (4 levels) instead of DMA fixed effects.\n"
            "Many DMAs have too few observations for reliable fixed effects."
        )
    }

    return sparsity_analysis


def create_artist_dma_features(event_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Create artist-DMA interaction features.

    Features:
    - artist_ticket_share_in_dma: What % of artist's tickets come from this DMA
    - artist_stream_share_in_dma: What % of artist's streams come from this DMA
    - untapped_potential: stream_share - ticket_share (gap = growth opportunity)
    """
    # Artist totals
    artist_totals = event_agg.groupby('artist_id').agg({
        'total_tickets': 'sum',
        'total_revenue': 'sum',
        'total_streams': 'sum'
    }).reset_index()
    artist_totals.columns = ['artist_id', 'artist_total_tickets', 'artist_total_revenue', 'artist_total_streams']

    # Artist-DMA totals
    artist_dma = event_agg.groupby(['artist_id', 'dma_id']).agg({
        'total_tickets': 'sum',
        'total_revenue': 'sum',
        'total_streams': 'sum'
    }).reset_index()

    # Merge and calculate shares
    artist_dma = artist_dma.merge(artist_totals, on='artist_id')
    artist_dma['ticket_share'] = artist_dma['total_tickets'] / artist_dma['artist_total_tickets']
    artist_dma['stream_share'] = artist_dma['total_streams'] / artist_dma['artist_total_streams'].replace(0, np.nan)
    artist_dma['untapped_potential'] = artist_dma['stream_share'] - artist_dma['ticket_share']

    return artist_dma[['artist_id', 'dma_id', 'ticket_share', 'stream_share', 'untapped_potential']]


# =============================================================================
# 6. MAIN EXPLORATION RUNNER
# =============================================================================

def run_full_exploration(data_dir: str = 'data', verbose: bool = True) -> Dict:
    """
    Run the complete feature exploration pipeline.

    Returns dictionary with all analysis results.
    """
    import io
    from contextlib import redirect_stdout

    # Load data
    if verbose:
        print("Loading and cleaning data...")
    with redirect_stdout(io.StringIO()):
        cleaned = run_cleaning_pipeline(data_dir)

    results = {}

    # 1. Validate aggregation scheme
    if verbose:
        print("\n" + "="*70)
        print("1. AGGREGATION SCHEME VALIDATION")
        print("="*70)

    agg_corrs = validate_aggregation_scheme(cleaned)
    results['aggregation_validation'] = agg_corrs

    if verbose:
        print(f"Weekly raw correlation: {agg_corrs['weekly_raw_corr']:.4f}")
        print(f"Weekly log correlation: {agg_corrs['weekly_log_corr']:.4f}")
        print(f"Event raw correlation: {agg_corrs['event_raw_corr']:.4f}")
        print(f"Event log correlation: {agg_corrs['event_log_corr']:.4f}")
        print("\n>>> Event-level aggregation reveals true correlation (0.77 vs 0.00)")

    # 2. Create event-level data
    if verbose:
        print("\n" + "="*70)
        print("2. CREATING EVENT-LEVEL DATASET")
        print("="*70)

    event_agg = create_event_level_data(cleaned)
    event_agg = add_streaming_to_events(event_agg, cleaned)
    event_agg = add_prev_event_features(event_agg)
    results['event_data'] = event_agg

    if verbose:
        print(f"Total events: {len(event_agg):,}")
        print(f"Artists: {event_agg['artist_id'].nunique()}")
        print(f"DMAs: {event_agg['dma_id'].nunique()}")

    # 3. Profit features analysis
    if verbose:
        print("\n" + "="*70)
        print("3. PROFIT FEATURES ANALYSIS")
        print("="*70)

    profit_corrs = analyze_profit_features(event_agg, cleaned['instagram'])
    results['profit_features'] = profit_corrs

    if verbose:
        print("Correlations with log_revenue:")
        print(profit_corrs.to_string(index=False))

    # 4. Growth features analysis (stratified)
    if verbose:
        print("\n" + "="*70)
        print("4. GROWTH FEATURES ANALYSIS (STRATIFIED BY SIZE)")
        print("="*70)

    growth_df = calculate_artist_growth(cleaned)
    demo_df = create_demographic_features(cleaned['instagram'])
    growth_stratified = analyze_growth_stratified(growth_df, demo_df)
    results['growth_features'] = growth_stratified

    if verbose:
        for tier, tier_results in growth_stratified.items():
            print(f"\n{tier.upper()} Artists - Top growth drivers:")
            print(tier_results.head(5).to_string(index=False))

    # 5. Feature correlation check
    if verbose:
        print("\n" + "="*70)
        print("5. FEATURE CORRELATION CHECK")
        print("="*70)

    corr_check = check_feature_correlations(event_agg)
    results['feature_correlations'] = corr_check

    if verbose:
        print(corr_check['interpretation'])
        print(f"\nCorrelation matrix (key features):")
        print(corr_check['correlation_matrix'].round(3).to_string())

    # 6. DMA features and sparsity analysis
    if verbose:
        print("\n" + "="*70)
        print("6. DMA FEATURES & SPARSITY ANALYSIS")
        print("="*70)

    dma_features = create_dma_features(event_agg)
    artist_dma_features = create_artist_dma_features(event_agg)
    dma_sparsity = analyze_dma_sparsity(event_agg)
    results['dma_features'] = dma_features
    results['artist_dma_features'] = artist_dma_features
    results['dma_sparsity'] = dma_sparsity

    if verbose:
        print(f"Total DMAs: {dma_sparsity['total_dmas']}")
        print(f"DMAs with < 5 events: {dma_sparsity['dmas_lt_5_events']}")
        print(f"DMAs with < 10 events: {dma_sparsity['dmas_lt_10_events']}")
        print(f"\nDMA size tier distribution:")
        print(dma_features['dma_size_tier'].value_counts().sort_index().to_string())
        print(f"\n>>> {dma_sparsity['recommendation']}")

    return results


if __name__ == '__main__':
    results = run_full_exploration(verbose=True)

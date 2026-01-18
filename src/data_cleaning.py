"""
Data Cleaning Pipeline for Artist Performance Dataset
Handles: deduplication, temporal alignment, missing values, and source reconciliation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def load_raw_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all raw CSV files into a dictionary of DataFrames."""
    data_dir = Path(data_dir)

    return {
        'instagram': pd.read_csv(data_dir / 'artist_instagram.csv'),
        'social_week': pd.read_csv(data_dir / 'artist_social_week.csv'),
        'mstreams_week': pd.read_csv(data_dir / 'artist_mstreams_week.csv'),
        'mstreams_dma_week': pd.read_csv(data_dir / 'artist_mstreams_dma_week.csv'),
        'l_ticket_week': pd.read_csv(data_dir / 'lsecondaryticket_artist_week.csv'),
        'l_ticket_dma_week': pd.read_csv(data_dir / 'lsecondaryticket_artist_dma_week.csv'),
        't_ticket_week': pd.read_csv(data_dir / 'tsecondaryticket_artist_week.csv'),
        't_ticket_dma_week': pd.read_csv(data_dir / 'tsecondaryticket_artist_dma_week.csv'),
    }


def clean_instagram(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Instagram data:
    1. Drop exact duplicate rows
    2. Remove problematic shared account (3975874939 mapped to multiple artists)
    3. Keep only highest-follower account per artist
    4. Drop columns with >90% missing values
    """
    df = df.copy()

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Instagram: dropped {n_before - len(df)} exact duplicate rows")

    # Remove problematic shared account
    problematic_account = 3975874939
    n_before = len(df)
    df = df[df['account_id'] != problematic_account]
    print(f"  Instagram: removed {n_before - len(df)} rows with shared account anomaly")

    # Keep only max-follower account per artist
    n_before = len(df)
    idx = df.groupby('artist_name')['num_followers'].idxmax()
    df = df.loc[idx].reset_index(drop=True)
    print(f"  Instagram: deduplicated to {len(df)} artists (removed {n_before - len(df)} fan accounts)")

    # Drop mostly-null columns
    cols_to_drop = ['engagements', 'avg_views', 'likes_not_from_followers']
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    print(f"  Instagram: dropped {len(existing_cols)} mostly-null columns")

    return df


def clean_streaming_dma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DMA-level streaming data:
    1. Drop rows with missing dma_id
    2. Convert dates to datetime
    3. Create unified week_id for joining
    """
    df = df.copy()

    # Drop missing DMA rows
    n_before = len(df)
    df = df.dropna(subset=['dma_id'])
    print(f"  Streaming DMA: dropped {n_before - len(df)} rows with missing dma_id")

    # Convert dates
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df['week_end_date'] = pd.to_datetime(df['week_end_date'])

    # Create week_id (ISO year-week format)
    df['week_id'] = df['week_start_date'].dt.strftime('%G-W%V')

    # Ensure dma_id is integer
    df['dma_id'] = df['dma_id'].astype(int)

    return df


def clean_ticket_dma(df_l: pd.DataFrame, df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and merge L and T ticket sources:
    1. Use L source as primary (better coverage)
    2. Filter to 'Primary Performer' to avoid duplicates from performance_type
    3. Join T source to get st_avg_price where available
    4. Align dates to Sunday-start weeks (shift back 1 day)
    5. Create unified week_id
    """
    df_l = df_l.copy()
    df_t = df_t.copy()

    # Filter to Primary Performer only (avoid duplicates from All/Secondary)
    n_before = len(df_l)
    df_l = df_l[df_l['performance_type'] == 'Primary Performer']
    print(f"  Ticket DMA: filtered to Primary Performer ({n_before:,} -> {len(df_l):,} rows)")

    # Convert dates
    df_l['start_date'] = pd.to_datetime(df_l['start_date'])
    df_l['end_date'] = pd.to_datetime(df_l['end_date'])
    df_t['start_date'] = pd.to_datetime(df_t['start_date'])
    df_t['end_date'] = pd.to_datetime(df_t['end_date'])

    # Shift L dates back 1 day to align Monday->Sunday week start
    df_l['week_start_date'] = df_l['start_date'] - pd.Timedelta(days=1)
    df_t['week_start_date'] = df_t['start_date'] - pd.Timedelta(days=1)

    # Create week_id
    df_l['week_id'] = df_l['week_start_date'].dt.strftime('%G-W%V')
    df_t['week_id'] = df_t['week_start_date'].dt.strftime('%G-W%V')

    # Merge T's avg_price into L
    t_prices = df_t[['artist_id', 'dma_id', 'week_id', 'st_avg_price']].drop_duplicates()

    df_merged = df_l.merge(
        t_prices,
        on=['artist_id', 'dma_id', 'week_id'],
        how='left'
    )

    n_with_price = df_merged['st_avg_price'].notna().sum()
    print(f"  Ticket DMA: merged with T avg_price ({n_with_price:,} matches)")

    # Calculate implied avg price where T price is missing
    df_merged['implied_avg_price'] = df_merged['st_order_value'] / df_merged['st_num_tickets']
    df_merged['avg_price'] = df_merged['st_avg_price'].fillna(df_merged['implied_avg_price'])

    return df_merged


def clean_social_week(df: pd.DataFrame) -> pd.DataFrame:
    """Clean social week data with datetime conversion and week_id."""
    df = df.copy()
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df['week_end_date'] = pd.to_datetime(df['week_end_date'])
    df['week_id'] = df['week_start_date'].dt.strftime('%G-W%V')
    return df


def clean_streaming_week(df: pd.DataFrame) -> pd.DataFrame:
    """Clean streaming week data with datetime conversion and week_id."""
    df = df.copy()
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df['week_end_date'] = pd.to_datetime(df['week_end_date'])
    df['week_id'] = df['week_start_date'].dt.strftime('%G-W%V')
    return df


def get_core_artists(data: Dict[str, pd.DataFrame]) -> set:
    """Get set of artist_ids present in all key tables."""
    ig_artists = set(data['instagram']['artist_id'].unique())
    stream_artists = set(data['mstreams_dma_week']['artist_id'].unique())
    ticket_artists = set(data['ticket_dma_week']['artist_id'].unique())

    core = ig_artists & stream_artists & ticket_artists
    print(f"  Core artists present in all tables: {len(core)}")
    return core


def run_cleaning_pipeline(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Run the full cleaning pipeline.

    Returns:
        Dictionary of cleaned DataFrames
    """
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)
    raw = load_raw_data(data_dir)

    print("\n" + "=" * 60)
    print("CLEANING PIPELINE")
    print("=" * 60)

    cleaned = {}

    # Instagram
    print("\n[1/5] Cleaning Instagram data...")
    cleaned['instagram'] = clean_instagram(raw['instagram'])

    # Streaming DMA
    print("\n[2/5] Cleaning streaming DMA data...")
    cleaned['mstreams_dma_week'] = clean_streaming_dma(raw['mstreams_dma_week'])

    # Ticket DMA (merge L and T)
    print("\n[3/5] Cleaning and merging ticket DMA data...")
    cleaned['ticket_dma_week'] = clean_ticket_dma(
        raw['l_ticket_dma_week'],
        raw['t_ticket_dma_week']
    )

    # Social week
    print("\n[4/5] Cleaning social week data...")
    cleaned['social_week'] = clean_social_week(raw['social_week'])

    # Streaming week (national)
    print("\n[5/5] Cleaning streaming week data...")
    cleaned['mstreams_week'] = clean_streaming_week(raw['mstreams_week'])

    # Identify core artists
    print("\n" + "=" * 60)
    print("IDENTIFYING CORE ARTISTS")
    print("=" * 60)
    cleaned['core_artist_ids'] = get_core_artists(cleaned)

    return cleaned


def create_analysis_dataset(
    cleaned: Dict[str, pd.DataFrame],
    filter_to_core: bool = True
) -> pd.DataFrame:
    """
    Create the main analysis dataset for DMA market selection.

    Joins:
    - Ticket DMA data (target: demand metrics)
    - Streaming DMA data (feature: streaming affinity)
    - Instagram demographics (feature: audience profile)

    Returns:
        DataFrame at artist-DMA-week grain with features and targets
    """
    ticket = cleaned['ticket_dma_week'].copy()
    streams = cleaned['mstreams_dma_week'].copy()
    instagram = cleaned['instagram'].copy()

    if filter_to_core:
        core_ids = cleaned['core_artist_ids']
        ticket = ticket[ticket['artist_id'].isin(core_ids)]
        streams = streams[streams['artist_id'].isin(core_ids)]
        instagram = instagram[instagram['artist_id'].isin(core_ids)]

    # Aggregate streaming by artist-dma-week (sum across platforms)
    streams_agg = streams.groupby(
        ['artist_id', 'dma_id', 'week_id']
    ).agg({
        'number_of_streams': 'sum',
        'week_start_date': 'first'
    }).reset_index()

    # Merge ticket with streaming
    df = ticket.merge(
        streams_agg,
        on=['artist_id', 'dma_id', 'week_id'],
        how='left',
        suffixes=('', '_stream')
    )

    # Merge with Instagram demographics (static per artist)
    ig_cols = [
        'artist_id', 'num_followers', 'avg_likes', 'avg_comments',
        'engagement_rate', 'male_amount', 'female_amount',
        'ages_13_17', 'ages_18_24', 'ages_25_34', 'ages_35_44', 'ages_45_64', 'ages_65_',
        'audience_credibility', 'notable_users_ratio'
    ]
    ig_subset = instagram[[c for c in ig_cols if c in instagram.columns]]

    df = df.merge(ig_subset, on='artist_id', how='left')

    print(f"Analysis dataset: {len(df):,} rows, {df['artist_id'].nunique()} artists, {df['dma_id'].nunique()} DMAs")

    return df


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'

    cleaned = run_cleaning_pipeline(data_dir)

    print("\n" + "=" * 60)
    print("CLEANED DATA SUMMARY")
    print("=" * 60)
    for name, df in cleaned.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {name}: {len(df):,} rows, {df.shape[1]} columns")
        elif isinstance(df, set):
            print(f"  {name}: {len(df)} items")

    # Create analysis dataset
    print("\n" + "=" * 60)
    print("CREATING ANALYSIS DATASET")
    print("=" * 60)
    analysis_df = create_analysis_dataset(cleaned)

    print("\nDone!")

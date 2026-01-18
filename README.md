# Market Selection for Touring - Solution

## Problem Framing

I chose **Question 3: Market selection for touring** - identifying the next best DMAs for an artist to play by predicting expected revenue. The goal is to rank markets balancing demand prediction and market characteristics.

## Data Used

- **ticket data** (lsecondaryticket_artist_dma_week): Aggregated to event-level (consecutive weeks of activity)
- **streaming data** (artist_mstreams_week): Artist demand signal
- **Instagram demographics**: Artist reach (followers)
- **DMA aggregates**: Market size controls

Key insight: Weekly correlations are ~0, but **event-level aggregation reveals r=0.77** between streams and revenue.

## Pipeline Design

```
python src/market_selection.py
```

The pipeline:
1. Loads and cleans data via `data_cleaning.py`
2. Creates event-level features via `feature_exploration.py`
3. Trains XGBoost model with time-series cross-validation
4. Compares against baseline (prev_log_revenue only)
5. Outputs market recommendations per artist

## Modeling Approach

**Target**: `log_revenue = log(1 + total_revenue)`

**Features** (by importance):
| Feature | Importance | Correlation |
|---------|------------|-------------|
| prev_log_revenue | 58% | 0.77 |
| log_streams | 29% | 0.77 |
| log_followers | 6% | 0.53 |
| prev_avg_price | 4% | 0.47 |
| log_dma_tickets | 3% | — |

**Model Comparison**:
| Metric | Baseline | XGBoost |
|--------|----------|---------|
| CV R² | 0.51 | **0.70** |
| Beyoncé MAPE | 58% | **26%** |
| Death Cab MAPE | 126% | **62%** |

**Overfitting check**: Train R²=0.88, CV R²=0.70, Gap=0.18. A bit of ovefitting, but model is still predictive.

## Limitations

- Death Cab still underestimated by 60-70% in top markets (loyal niche fanbase)
- First-time DMA visits lack historical signal. Taylor Swift is one example of someone who rarely repeat DMA
- Model regresses toward mean for extreme artists

## Next Steps

- Add genre-specific adjustments
- Incorporate DMA-level streaming data for untapped market detection
- Artist-tier interaction features for better handling of superstars vs. niche artists
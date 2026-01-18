# Take‑Home: Senior Applied Scientist

## Overview

You’ll spend 2–3 hours exploring the provided Artist Performance datasets (see `data/README.md`) and answer **one** of the three questions below. Focus on thoughtful scoping, rigorous methodology, and a maintainable, pipeline-like code structure rather than broad coverage. At the end of the timebox, submit your work as-is.

- **Timebox**: 2–3 hours, hard stop. Submit what you have, even if incomplete. Note what you would do next with more time.
- **Language**: Python only
- **Style**: Functions or class-based pipeline preferred over an exploratory notebook
- **Data Usage**: Do not try to use everything; select only the data relevant to your chosen question
- **AI usage**: Allowed. Be prepared to present and defend all work in a follow-up interview

## What we’re looking for

We are not evaluating you on model breadth, performance or leaderboard metrics. We are evaluating:

- Your coding quality, readibility and software craftsmanship
- Your critical thinking, problem framing, and judgment under time constraints
- How you assess data relevance, underlying assumptions, and risks
- How you reason about causality vs. correlation and temporal leakage
- How you structure a small, maintainable pipeline with clear separation of concerns

**Please:**

- Make and document explicit assumptions about the data, joins, time alignment, and leakage prevention
- Reduce scope and features to what’s feasible in 2–3 hours
- Prefer clarity and reasoning over complexity or overfitting
- Share what you’d do next with more time
- Select a sample of artists to make problem more tractable, explain decision.

## Datasets

Use the Artist Data Dictionary for detailed column definitions and scope. The main tables include:

- artist_instagram
- artist_social_week
- artist_mstreams_week
- lsecondaryticket_artist_week
- tsecondaryticket_artist_week
- tsecondaryticket_artist_dma_week
- lsecondaryticket_artist_dma_week
- artist_mstreams_dma_week

Important scope notes:

- Weekly granularity, last ~5 years
- Ticket sales primarily U.S.
- Streaming global and at DMA level
- Instagram audience demographics are followers, not artists

## Choose ONE primary question

Pick one question to focus on. You do not need to cover all three.

1) Forecasting demand for live events

- Goal: Predict near-term secondary ticket sales demand or revenue for an artist at DMA‑week level.
- Examples: st_num_tickets or st_order_value as the target; DMA features, streaming momentum, and social engagement as signals.

2) Drivers of streaming growth

- Goal: Quantify which factors correlate with short-term growth in Spotify streams for an artist.
- Examples: Use lagged social metrics, demographics, or event signals to explain changes in cleaned_cumulative_spotify_streams or number_of_streams.

3) Market selection for touring

- Goal: Identify the next best DMAs for an artist to play, balancing expected demand, price sensitivity, and audience fit.
- Examples: Rank DMAs using historical sales, price bands, and streaming audience concentration.

4) Classify artists into cohorts
- Goal: Classify artists into clear, data-driven cohorts. Explain the features chosen and why.
- Examples: Cohort artists by general popularity and momentum

## Bonus points (optional)

- Multi-variable impact measurements 
- Sensible handling of class/scale imbalance and outliers
- Lightweight hyperparameter search with clear stopping criteria

## What to submit

At the end of 2–3 hours, submit:

- Your code and pipeline
- A short Additional README that explains:
    - Problem framing and scope choices
    - Data used and why
    - Pipeline design and how to run it
    - Modeling and evaluation approach
    - Findings and limitations
    - Next steps if you had more time
- Optionally, a slim exploratory notebook for plots or quick checks
- Either:
    - Share a GitHub repository link, or
    - Compress the repo (zip) and send it

## Code requirements

- Python 3.9+ recommended
- Deterministic seeds where applicable
- Minimal dependencies; include requirements.txt or pyproject.toml
- If you use notebooks, keep them small and reproducible; the core logic should live in importable modules
- Clear instructions in the Additional README
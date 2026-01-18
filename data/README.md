# 🎵 Artist Data Dictionary

---

## 📘 Overview

This data dictionary describes the **Artist Performance Dataset**, which consolidates **social media, streaming, and live event metrics** **for a stratified sample 500 artists based on social media following rankings**.

The dataset covers the **past 5 years through the start of the current quarter**.

---

| Column | Description |
| --- | --- |
| `artist_id` | Unique identifier for each artist in the dataset. |
| `artist_name` | Name of the artist associated with the record. |
| `dma_id` | Unique identifier for the Designated Market Area (DMA) where the artist activity is measured. |
| `dma_name` | Name of the DMA region (e.g., “New York,” “Los Angeles”). |
| `platform_id` | Unique identifier for the platform (e.g., Spotify, YouTube, Instagram). |
| `platform_name` | Name of the digital or social platform where data is collected. |
| `start_date` | Start date of the reporting or data collection period. |
| `end_date` | End date of the reporting or data collection period.  |
| `week_start_date` | Start date of the corresponding week for the observation period. |
| `week_end_date` | End date of the corresponding week for the observation period.  |

---

## 🏁 Output Tables

---

### 🟧 `artist_instagram`

**Description:**

Instagram audience statistics and engagement metrics for top artists.

| Column | Description |
| --- | --- |
| `account_id` | Unique identifier for the artist’s Instagram account. |
| `num_followers` | Total number of followers on Instagram. |
| `avg_likes` | Average number of likes per post. |
| `avg_comments` | Average number of comments per post. |
| `avg_views` | Average number of video views per post. |
| `engagement_rate` | Engagement rate as a percentage of followers engaging with posts. |
| `male_amount` | Estimated number of male followers. |
| `female_amount` | Estimated number of female followers. |
| `ages_13_17` | Estimated number of followers aged 13–17. |
| `ages_18_24` | Estimated number of followers aged 18–24. |
| `ages_25_34` | Estimated number of followers aged 25–34. |
| `ages_35_44` | Estimated number of followers aged 35–44. |
| `ages_45_64` | Estimated number of followers aged 45–64. |
| `ages_65_` | Estimated number of followers aged 65 and older. |
| `follower_white` | Estimated number of followers identifying as White. |
| `follower_african_american` | Estimated number of followers identifying as African American. |
| `follower_asian` | Estimated number of followers identifying as Asian. |
| `follower_hispanic` | Estimated number of followers identifying as Hispanic. |
| `follower_lang_en` | Followers whose primary language is English. |
| `follower_lang_es` | Followers whose primary language is Spanish. |
| `follower_lang_pt` | Followers whose primary language is Portuguese. |
| `follower_lang_de` | Followers whose primary language is German. |
| `follower_lang_fr` | Followers whose primary language is French. |
| `follower_lang_zh` | Followers whose primary language is Chinese. |
| `follower_lang_id` | Followers whose primary language is Indonesian. |
| `follower_lang_ru` | Followers whose primary language is Russian. |
| `notable_users_ratio` | Ratio of verified or notable users following the account. |
| `audience_credibility` | Score representing follower authenticity. |
| `credibility_class` | Classification of follower credibility tier (e.g., high, medium, low). |
| `last_updated` | Timestamp of last update for this record. |
| `created_at` | Timestamp of record creation. |

---

### 🟧 `artist_social_week`

**Description:**

Aggregated social-media metrics by week.

| Column | Description |
| --- | --- |
| `max_following` | Maximum recorded social following during the week across all tracked platforms. |

---

### 🟧 `artist_mstreams_week`

**Description:**

Music streaming data aggregated by week.

| Column | Description |
| --- | --- |
| `number_of_streams` | Total number of mobile streams recorded during the week. |

---

### 🟧 `lsecondaryticket_artist_week`

**Description:**

Secondary ticket-sales data by week (L source).

| Column | Description |
| --- | --- |
| `genre_id` | Unique identifier for the artist’s genre. |
| `genre` | Name of the genre associated with the artist or event. |
| `st_event_count` | Number of ticketed events in the week. |
| `st_num_tickets` | Total number of tickets sold in the week. |
| `st_order_value` | Total order value (revenue) from secondary ticket sales. |

---

### 🟧 `tsecondaryticket_artist_week`

**Description:**

Secondary ticket-sales data by week (T source).

| Column | Description |
| --- | --- |
| `st_order_value` | Total order value (revenue) from secondary ticket sales. |
| `st_num_tickets` | Total number of tickets sold in the week. |
| `st_event_count` | Number of ticketed events in the week. |
| `st_avg_price` | Average price per ticket sold. |

---

### 🟧 `tsecondaryticket_artist_dma_week`

**Description:**

Secondary ticket-sales data by DMA & week (T source).

| Column | Description |
| --- | --- |
| `st_order_value` | Total order value (revenue) from secondary ticket sales. |
| `st_num_tickets` | Total number of tickets sold within the DMA. |
| `st_event_count` | Number of ticketed events within the DMA. |
| `st_avg_price` | Average ticket price for the DMA and week. |

---

### 🟧 `lsecondaryticket_artist_dma_week`

**Description:**

Secondary ticket-sales data by DMA & week (L source).

| Column | Description |
| --- | --- |
| `st_order_value` | Total order value (revenue) from secondary ticket sales. |
| `st_num_tickets` | Total number of tickets sold within the DMA. |
| `st_event_count` | Number of ticketed events within the DMA. |
| `st_avg_price` | Average ticket price for the DMA and week. |

---

### 🟧 `artist_mstreams_dma_week`

**Description:**

Music streaming data by DMA aggregated by week.

| Column | Description |
| --- | --- |
| `number_of_streams` | Total number of music streams recorded within the DMA for the week. |

## 🔍 Data Scope & Filters

- **Time Period:** Last 5 years through current quarter start
- **Artist Filter:** Stratified sample of 320 artists based on popularity
- **Geographic Scope:**
    - 🇺🇸 U.S. focus for ticket sales
    - 🌐 Worldwide and U.S. for streaming data
    - 🌎 Global Instagram audience demographics
- **Update Frequency:** Weekly
- **Data Sources:** Instagram, Spotify, secondary ticket markets, etc.

---

## 📝 Notes

- All `artist_instagram` demographics reflect **followers**, *not artists*.
- Reachability metrics represent potential advertising reach.
- Credibility metrics assess audience authenticity.
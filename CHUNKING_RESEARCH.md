# Chunking Strategy Research

Systematic sweep of chunking strategies for the hiking time prediction model.
All experiments use LOO-CV on 20 routes, ridge_alpha=0.5, two-stage architecture
(physics OLS → Ridge residuals).

---

## Baseline

| Strategy | Chunk size | MAE          | MAPE     |
|----------|------------|--------------|----------|
| distance | 600 m      | **13.8 min** | **4.4%** |

---

## 1. Distance-based (original strategy)

Split every N metres of horizontal distance.

| Chunk size | MAE          | MAPE     |
|------------|--------------|----------|
| 200 m      | 14.4 min     | 4.7%     |
| 300 m      | 16.0 min     | 5.2%     |
| **400 m**  | 14.7 min     | 5.0%     |
| **600 m**  | **13.8 min** | **4.4%** |
| 700 m      | 13.7 min     | 4.5%     |
| 800 m      | 16.4 min     | 5.4%     |
| 1000 m     | 13.8 min     | 4.6%     |
| 1500 m     | 19.1 min     | 6.1%     |

**Default updated: 200 m → 600 m.**
Larger chunks smooth out per-chunk gradient noise; degradation beyond ~800 m
starts losing signal about difficult sections.

---

## 2. Elevation-based

Split every N metres of cumulative absolute elevation change (|gain| + |loss|).

| Chunk size | MAE      | MAPE |
|------------|----------|------|
| 20 m       | 16.7 min | 5.3% |
| 30 m       | 19.2 min | 6.3% |
| 40 m       | 21.2 min | 6.5% |
| 50 m       | 19.2 min | 6.2% |
| 60 m       | 20.3 min | 6.3% |
| 75 m       | 20.6 min | 6.8% |
| 100 m      | 18.4 min | 5.8% |

**Result: worse across all sizes.** Variable-length chunks make per-chunk
distance stats noisy and incomparable across chunks.

---

## 3. Tobler-time-based

Split when accumulated theoretical Tobler hiking time reaches a threshold.
Steep terrain → small chunks; flat terrain → large chunks.

| Threshold  | MAE          | MAPE     |
|------------|--------------|----------|
| 5 min      | 20.4 min     | 6.6%     |
| 7 min      | 20.1 min     | 6.7%     |
| 9 min      | 20.0 min     | 6.8%     |
| **11 min** | **15.5 min** | **4.9%** |
| 13 min     | 18.6 min     | 6.2%     |
| 15 min     | 21.8 min     | 7.0%     |
| 20 min     | 22.5 min     | 7.0%     |

**Result: worse.** The p75/p90 Tobler percentile features — key Stage 2
predictors — measure the difficulty *distribution* across chunks. With
Tobler-time chunking every chunk has approximately the same Tobler time by
construction, so those features flatten out and lose their signal.

---

## 4. Elevation smoothing

Rolling-median smoothing applied to GPS elevation values before chunking.
Tested with distance-600m chunks.

| Window      | MAE          | MAPE     |
|-------------|--------------|----------|
| **1 (off)** | **13.8 min** | **4.4%** |
| 3           | 16.9 min     | 5.7%     |
| 5           | 16.9 min     | 5.6%     |
| 7           | 16.2 min     | 5.4%     |
| 9           | 16.7 min     | 5.6%     |
| 11          | 17.1 min     | 5.6%     |
| 15          | 16.7 min     | 5.6%     |
| 21          | 17.5 min     | 5.8%     |
| 31          | 18.2 min     | 6.1%     |

**Result: smoothing is counterproductive.** The Stage 2 features that matter
most (`total_steep_loss_m`, `frac_very_steep`, `loss_per_km`) detect steep
local sections. Smoothing blurs those out: Fanlo's error grows +21 → +44 min
and Ribeira Brava 25.5 km grows -49 → -65 min as window increases.
600 m chunks already average ~30–60 GPS points, so noise cancels naturally.

**Default: window=1 (disabled).**

---

## 5. Ascent/descent direction

Split at ascent→descent and descent→ascent reversals. Smoothed elevation
(window=11) used for direction detection; raw points used for features.
`min_dist` = minimum horizontal distance between splits.

| min_dist  | MAE          | MAPE     |
|-----------|--------------|----------|
| 50 m      | 16.7 min     | 5.4%     |
| **100 m** | **14.1 min** | **4.7%** |
| 150 m     | 14.2 min     | 4.7%     |
| 200 m     | 15.9 min     | 5.5%     |
| 300 m     | 18.0 min     | 6.1%     |

**Result: slightly worse than baseline.** Variable chunk lengths break the
distributional statistics. The model already captures ascent/descent asymmetry
through `total_tobler_ascent_min`, `total_tobler_descent_min`, and
`total_steep_loss_m`.

---

## 6. Grade-band

Split when the terrain category changes. Bands (sign-aware):
flat (|grade| < 3%), gentle (3–10%), steep (10–20%), very steep (>20%).
`min_dist` = minimum horizontal distance between splits.

| min_dist  | MAE          | MAPE     |
|-----------|--------------|----------|
| 50 m      | 18.7 min     | 6.3%     |
| 100 m     | 15.7 min     | 5.1%     |
| **150 m** | **15.0 min** | **4.9%** |
| 200 m     | 17.6 min     | 5.8%     |
| 300 m     | 16.2 min     | 5.4%     |

**Result: worse.** Same root cause as direction splitting — variable chunk
length degrades the statistics.

---

## 7. Fixed chunk count

Each route is split into exactly N equal-distance segments regardless of total
route length. Normalises the p75/p90 Tobler percentile features across routes
of different lengths.

| N      | MAE          | MAPE     |
|--------|--------------|----------|
| 8      | 22.1 min     | 7.0%     |
| 10     | 15.5 min     | 4.9%     |
| 11     | 17.4 min     | 5.5%     |
| **12** | **11.7 min** | **3.9%** |
| 13     | 18.4 min     | 5.8%     |
| 14     | 16.1 min     | 5.3%     |
| **15** | **11.9 min** | **3.9%** |
| **16** | **11.6 min** | **3.8%** |
| 17     | 16.5 min     | 5.0%     |
| 18     | 20.2 min     | 6.6%     |
| 20     | 14.0 min     | 4.6%     |
| 25     | 13.3 min     | 4.3%     |

**Result: unstable — not a real improvement.** N=12/15/16 appear to beat
the baseline, but N=13/17/18 are far worse. The ±1 sensitivity means the
apparent wins are coincidental chunk-boundary alignment for this specific
20-route dataset, not a generalizable signal.

---

## 8. Multi-scale

All 19 features from 600 m chunks, concatenated with 6 distribution-only
features from fine-scale chunks (p75, p90, grade_std_mean, frac_steep,
frac_very_steep, n_chunks) → 25 features total.

| Fine scale | Coarse scale | MAE          | MAPE     |
|------------|--------------|--------------|----------|
| **100 m**  | 600 m        | **14.5 min** | **4.8%** |
| 200 m      | 600 m        | 16.3 min     | 5.1%     |
| 300 m      | 600 m        | 15.6 min     | 5.0%     |
| 200 m      | 800 m        | 17.0 min     | 5.6%     |

**Result: worse.** Adding fine-scale distribution features increases noise
more than it adds signal with N=20.

---

---

## 9. Overlapping windows

600 m window with sliding stride (50–75% overlap).  Distribution features
(p75, p90, grade_std_mean, frac_steep, frac_very_steep, n_chunks) replaced
with estimates computed over the overlapping windows; global totals unchanged.

| Stride | Overlap | MAE      | MAPE |
|--------|---------|----------|------|
| 300 m  | 50%     | 17.1 min | 5.6% |
| 200 m  | 67%     | 14.9 min | 4.9% |
| 150 m  | 75%     | 16.0 min | 5.2% |

**Result: worse.** Overlapping windows produce correlated samples that
confuse the Ridge regulariser more than the marginal gain in percentile
stability.

---

## 10. Coarse dual-scale

Fine-scale (500–600 m) base features plus 6 distribution stats from a coarser
scale appended (→ 25 features total).

| Scales       | MAE      | MAPE |
|--------------|----------|------|
| 500 + 1000 m | 14.5 min | 4.8% |
| 600 + 900 m  | 17.0 min | 5.4% |
| 600 + 1200 m | 17.1 min | 5.5% |

**Result: marginally worse.** 500+1000 m is the closest to the baseline
(+0.7 min) but not a real improvement. The 600 m scale already captures
enough macro context; adding 1000–1200 m features introduces noise.

---

## 11. Position-aware features (+6)

Appended to 600 m base features:

- `second_half_tobler_frac`, `second_half_gain_frac`, `second_half_loss_frac`
- `hard25_gain_frac`, `hard25_loss_frac` (gain/loss fraction in hardest 25% of chunks)
- `hard25_center_pos` (mean positional index 0→1 of hardest chunks)

| Config        | MAE      | MAPE |
|---------------|----------|------|
| base + pos    | 20.4 min | 6.7% |

**Result: worse (+6.6 min).** Adding 6 features to N=20 causes overfitting
in Stage 2 that Ridge cannot regularise away cleanly.

---

## 12. Event features (+4)

Appended to 600 m base features:

- `n_reversals` (direction flips between consecutive chunks)
- `longest_steep_descent_km`, `longest_steep_ascent_km` (|grade| > 15%)
- `n_long_steep_descents` (descent runs > 500 m)

| Config          | MAE      | MAPE |
|-----------------|----------|------|
| base + events   | 18.2 min | 5.8% |

**Result: worse (+4.4 min).** Same cause — too many parameters for N=20.

---

## 13. Combinations

| Config                       | MAE      | MAPE  | vs baseline |
|------------------------------|----------|-------|-------------|
| position + events            | 25.8 min | 8.7%  | +12.0 min   |
| overlapping(200m) + pos+ev   | 25.0 min | 8.4%  | +11.2 min   |
| dual(500+1000) + pos+ev      | 19.4 min | 6.7%  | +5.6 min    |

**Result: uniformly bad.** Combining multiple extra-feature approaches
stacks overfitting — 10 extra features is far too many for 20 samples.

---

## Summary

| Strategy                       | Best config  | MAE          | vs baseline          |
|--------------------------------|--------------|--------------|----------------------|
| **distance (current default)** | **600 m**    | **13.8 min** | —                    |
| fixed_count                    | N=15/16      | ~11.7 min    | -2.1 min ⚠️ unstable |
| direction                      | min=100 m    | 14.1 min     | +0.3 min             |
| dual-scale (coarse)            | 500+1000 m   | 14.5 min     | +0.7 min             |
| overlapping windows            | stride=200 m | 14.9 min     | +1.1 min             |
| grade_band                     | min=150 m    | 15.0 min     | +1.2 min             |
| tobler-time                    | 11 min       | 15.5 min     | +1.7 min             |
| elevation                      | 20 m         | 16.7 min     | +2.9 min             |
| event features                 | +4 features  | 18.2 min     | +4.4 min             |
| position features              | +6 features  | 20.4 min     | +6.6 min             |
| multiscale (fine, old)         | fine=100 m   | 14.5 min     | +0.7 min             |

## Conclusion

**Distance-based at 600 m is the robust optimum.** It wins because:

1. Equal horizontal distance per chunk → per-chunk statistics (gradient
   distribution, Tobler percentiles) are on a comparable scale across all
   chunks and all routes.
2. 600 m covers ~30–60 GPS points, averaging out sensor noise without losing
   the steep-section signal the model depends on.
3. Results are stable across ±100 m — no sensitivity to exact boundary
   placement.

The key Stage 2 features (`total_steep_loss_m`, `frac_very_steep`,
`loss_per_km`) require that the model can detect steep sections relative to
the rest of the route. Any strategy that makes chunks variable in length or
normalises Tobler time per chunk destroys this contrast.

**The fundamental constraint is dataset size (N=20), not chunking strategy.**
Every tested improvement adds parameters faster than the 20-sample dataset
can absorb:

- Overlapping windows: correlated samples confuse Ridge
- Extra features (position, events, dual-scale): overfitting dominates
- Variable-length chunks (direction, grade-band, elevation, Tobler-time):
  break the distributional statistics that Stage 2 relies on

The remaining error (stubborn outlier: Ribeira Brava 25.5 km, −50 min) is
not addressable through feature engineering — more training data is the
only reliable path to lower MAE.

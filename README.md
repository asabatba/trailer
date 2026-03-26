# Hiking Time Predictor

Predicts route hiking time from GPX files.  
Designed for **small datasets (~20 recordings)** using a physics-anchored model.

```
GPX file
   │
   ▼ parse_gpx()
TrackPoints  [lat, lon, ele, timestamp]
   │
   ▼ chunk_track()
Chunks  [~200 m segments]
   │
   ▼ Chunk.compute()  (per chunk)
   ├─ dist_m, gain_m, loss_m
   ├─ mean_grade, max_grade, grade_std
   └─ tobler_min  ← Tobler's hiking function
   │
   ▼ aggregate_features()
Feature vector  (16 values)
   │
   ▼ HikingTimeModel.predict()
Duration (minutes)
```

---

## Why this architecture?

### The small-data problem

With N=20, standard ML models overfit immediately.  
A plain Ridge regressor has 16+ weights to fit — more parameters than data points.

### Solution: calibrated physics + residual correction

**Stage 1 — Physics calibration (Tobler)**  
Tobler's hiking function gives a theoretical speed for any slope:

```
speed (km/h) = 6 · exp(−3.5 · |gradient + 0.05|)
```

The `+0.05` bias shifts the optimum to ~2.86° downhill, matching empirical observations.  
We fit only **4 parameters** on top of Tobler:

```
ŷ = α·tobler_min + β·gain_m + γ·loss_m + δ
```

- `α` calibrates your personal pace vs theory (typically 0.6–1.2)
- `β` extra cost of climbing beyond Tobler (path difficulty, stops)
- `γ` extra cost of descending (knee stress, technical terrain)
- `δ` fixed overhead (start/end, short breaks)

This works well even with **5 samples**.

**Stage 2 — Residual correction (Ridge, N ≥ 12)**  
A Ridge regressor learns the residual from Stage 1 using terrain shape features: roughness, fraction of steep terrain, grade distribution.  Strongly regularised (α=10) to prevent overfitting.

### LOO-CV, not k-fold

With N=20, Leave-One-Out CV is the correct evaluation strategy.  
Every sample becomes the test set exactly once — no data is wasted.

---

## Chunk features

| Feature      | Description                         |
|--------------|-------------------------------------|
| `dist_m`     | Horizontal distance of chunk        |
| `gain_m`     | Elevation gained (metres)           |
| `loss_m`     | Elevation lost (abs metres)         |
| `mean_grade` | Mean rise/run ratio                 |
| `max_grade`  | Max rise/run in chunk               |
| `grade_std`  | Slope variability (roughness proxy) |
| `tobler_min` | Tobler time for this chunk          |
| `difficulty` | `gain·1.0 + loss·0.5 + dist·0.001`  |

## Aggregated features (model input)

| Feature                                     | Why it matters            |
|---------------------------------------------|---------------------------|
| `total_dist_km`                             | Distance baseline         |
| `total_gain_m`                              | Primary effort driver     |
| `total_loss_m`                              | Knee/pace cost            |
| `total_tobler_min`                          | **Best single predictor** |
| `mean_grade`, `max_grade`, `grade_std_mean` | Terrain shape             |
| `p75_tobler_min`, `p90_tobler_min`          | Bottleneck segments       |
| `frac_steep`, `frac_very_steep`             | Technical difficulty      |
| `gain_per_km`, `loss_per_km`                | Grade density             |
| `tobler_efficiency`                         | Pace proxy                |
| `n_chunks`                                  | Route resolution          |

---

## Setup

```bash
uv sync
```

## Usage

### Training (GPX files with timestamps)

```bash
# GPS-recorded tracks (timestamps extracted automatically)
trailer-train --gpx-dir ./my_hikes --output model.pkl

# Manual labels (filename_stem, actual_minutes)
trailer-train --gpx-dir ./my_hikes --labels labels.csv --output model.pkl
```

### Labels CSV format

```
hike_montserrat,187
hike_pedraforca,310
pyrenees_tour,485
```

### Prediction

```bash
trailer-predict model.pkl new_route.gpx
trailer-predict model.pkl --dir ./routes/ --verbose
```

### Inspect a GPX

```bash
trailer-train --gpx-dir ./my_hikes --describe
```

### Generate test data

```bash
trailer-generate-demo
trailer-train --gpx-dir demo_gpx --output model.pkl
```

### API server

```bash
uv run uvicorn trailer.server:app --reload
# or
trailer-server --reload
```

---

## Tuning

| Parameter      | Default | Effect                                                  |
|----------------|---------|---------------------------------------------------------|
| `--chunk-size` | 200 m   | Smaller → more granular features, slower. Try 100–500 m |
| `--alpha`      | 10.0    | Higher → more conservative residual correction          |

**Chunk size guidance:**

- 100 m — best for short technical routes (< 10 km)
- 200 m — good default balance  
- 500 m — better for very long routes (> 30 km), reduces noise

---

## Files

```
hike_predictor/
├── gpx_features.py      # parsing, chunking, feature extraction
├── model.py             # HikingTimeModel + LOO-CV
├── train.py             # training CLI
├── predict.py           # prediction CLI
└── generate_demo_gpx.py # synthetic test data generator
```

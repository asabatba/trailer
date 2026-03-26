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
Feature vector  (19 values)
   │
   ▼ HikingTimeModel.predict()
Duration (minutes)
```

---

## Why this architecture?

### The small-data problem

With N=20, standard ML models overfit immediately.  
A plain Ridge regressor has 16+ weights to fit — more parameters than data points.

### Solution: constrained physics model

**Physics-anchored monotone regression**  
Tobler's hiking function gives a theoretical speed for any slope:

```
speed (km/h) = 6 · exp(−3.5 · |gradient + 0.05|)
```

The `+0.05` bias shifts the optimum to ~2.86° downhill, matching empirical observations.  
The model learns only a small set of nonnegative penalty terms:

```
ŷ = α·total_tobler_min
  + β·total_steep_loss_m
  + γ·grade_std_mean
  + δ·frac_steep
  + ε·frac_very_steep
```

- `α` calibrates your personal pace vs Tobler
- `β` penalises steep descending terrain
- `γ` penalises rough / variable slope
- `δ`, `ε` penalise sustained steepness

All coefficients are constrained to be nonnegative and there is no intercept.
That means increasing any learned effort term cannot reduce predicted time.

The full aggregated feature vector is still computed, but the trained model only
uses the nonnegative, physics-aligned subset above.  This keeps the model
simple enough for small datasets while ruling out the obvious sign errors from
an unconstrained residual model.

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

All aggregated features are extracted for inspection and experimentation.
The default trained model uses this constrained subset:
`total_tobler_min`, `total_steep_loss_m`, `grade_std_mean`, `frac_steep`,
`frac_very_steep`.

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
| `--chunk-size` | 600 m   | Smaller → more granular features, slower. Try 100–500 m |
| `--alpha`      | 0.5     | Higher → more conservative constrained penalties        |

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

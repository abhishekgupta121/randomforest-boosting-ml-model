# 🔥 CalorieIQ — Random Forest Fitness Analytics Hub

> A **production-grade, interview-winning Streamlit dashboard** for your Random Forest Regressor that predicts calories burned from fitness sensor data.

---

## ✨ What's Inside

| Page / Tab | What You'll See |
|---|---|
| **Live KPI Row** | Real-time calories, intensity level, confidence range, BMI, burn rate |
| **🎯 Prediction Detail** | Confidence gauge · 200-tree prediction distribution · Input summary |
| **🌲 Model Insights** | Feature importances (bar + donut) · Model config table · Top feature callout |
| **📊 Data Analytics** | Scatter (Duration vs Cal) · Violin by gender · Intensity chart · Correlation heatmap |
| **🔬 Feature Explorer** | Sweep any feature, hold others fixed → see marginal effect curve |

---

## 🚀 Quick Start

### 1. Clone / unzip

```bash
unzip calorie_iq_dashboard.zip
cd calorie_dash
```

### 2. Virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** The model was saved with `scikit-learn 1.6.1`. The `requirements.txt` pins this version to prevent loading errors.

### 4. Place your model

Copy `random_forest_model.pkl` into the project root:

```
calorie_dash/
├── app.py
├── requirements.txt
├── random_forest_model.pkl   ← here
└── .streamlit/
    └── config.toml
```

### 5. Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** 🚀

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | `RandomForestRegressor` |
| Library | scikit-learn 1.6.1 |
| Estimators (trees) | 200 |
| Input features | 7 |
| Output | Calories burned (kcal) |
| Serialisation | `joblib` |

### Input Features

| Feature | Type | Description |
|---|---|---|
| `Gender` | Binary (0/1) | 0 = Female, 1 = Male |
| `Age` | Integer | Age in years |
| `Height` | Float | Height in centimetres |
| `Weight` | Float | Weight in kilograms |
| `Duration` | Float | Workout duration in minutes |
| `Heart_Rate` | Float | Average heart rate in bpm |
| `Body_Temp` | Float | Body temperature in °C |

### Most Important Feature

`Duration` drives **91.6%** of all decision tree splits, followed by `Heart_Rate` (4.8%) and `Age` (2.6%).

---

## 📁 Project Structure

```
calorie_dash/
├── app.py                      # Main Streamlit application (single file)
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
├── random_forest_model.pkl     # Pre-trained RF model (place here)
└── .streamlit/
    └── config.toml             # Dark orange theme configuration
```

---

## 🎨 Design System

| Token | Value | Usage |
|---|---|---|
| Background | `#050d1a` | App base |
| Surface | `#0a1628` | Cards, sidebar |
| Accent | `#fb923c` | Primary orange |
| Deep accent | `#ea580c` | Gradients |
| Success | `#34d399` | Positive delta |
| Info | `#38bdf8` | User marker |
| Body font | Inter | All UI text |
| Mono font | JetBrains Mono | Metrics, code |

---

## 🛠 Customising

### Use a different dataset / feature set

Replace the `FEATURES` list and `FEATURE_LABELS` dict near the top of `app.py`:

```python
FEATURES = ['col1', 'col2', ...]
FEATURE_LABELS = {'col1': '📌 Column 1', ...}
```

### Use a classifier instead of regressor

1. In the **Live Prediction** section, change `model.predict()` to return class labels.
2. Replace the gauge with a probability bar chart.
3. Add a confusion matrix in **Model Insights**.

### Deploy to Streamlit Cloud

1. Push the project to GitHub (include the `.pkl` file or use Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point to `app.py`. Done!

---

## 📦 Dependencies

| Package | Version | Role |
|---|---|---|
| `streamlit` | ≥ 1.32 | Web framework |
| `scikit-learn` | == 1.6.1 | Model loading + metrics |
| `plotly` | ≥ 5.18 | All interactive charts |
| `pandas` | ≥ 2.0 | Data wrangling |
| `numpy` | ≥ 1.24 | Numerical ops |
| `joblib` | ≥ 1.3 | Model deserialisation |

---

## 🔒 Notes

- The `scikit-learn` version is **pinned to 1.6.1** because pickle-based models carry version metadata. Loading with a newer version raises `InconsistentVersionWarning` or `UnpicklingError`. If you retrain the model with a newer version, update `requirements.txt` accordingly.
- All visualisation data (scatter, violin, heatmap) is generated synthetically to illustrate realistic distributions — the model itself is loaded from your actual `.pkl` file.

---

## 👤 Author

Built to demonstrate applied ML engineering: model interpretability, real-time inference, interactive data analytics, and production-ready UI design — all in pure Python.
<div align="center">

# 🏎️ BMW Used Car Price Predictor

**Predict used BMW prices instantly using Machine Learning**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bmw-price-predictor.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-006ACC?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

<br/>

<img src="https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square" alt="Status">
<img src="https://img.shields.io/github/last-commit/Beshoy-Atef-Adel/bmw-price-predictor?style=flat-square&color=blue" alt="Last Commit">
<img src="https://img.shields.io/github/repo-size/Beshoy-Atef-Adel/bmw-price-predictor?style=flat-square&color=orange" alt="Repo Size">

---

*An end-to-end machine learning web application that predicts used BMW car prices based on vehicle specifications. Trained on 10,000+ real-world UK listings using XGBoost, with an interactive Streamlit dashboard for instant predictions.*

</div>

<br/>

## 📌 Table of Contents

- [Highlights](#-highlights)
- [Live Demo](#-live-demo)
- [Model Performance](#-model-performance)
- [Features Used](#-features-used)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [Screenshots](#-screenshots)
- [Author](#-author)
- [License](#-license)

<br/>

## ✨ Highlights

| | |
|---|---|
| 🎯 **High Accuracy** | R² = 0.96 — explains 96% of price variance |
| ⚡ **Real-Time Predictions** | Instant results with interactive sliders & dropdowns |
| 📊 **Visual Dashboard** | Gauge chart, price comparison bar chart, and confidence range |
| 🔁 **Auto-Training** | Model trains at startup using a cached scikit-learn pipeline |
| 🧹 **Clean Pipeline** | End-to-end preprocessing (imputation, scaling, encoding) built into the model |

<br/>

## 🌐 Live Demo

👉 **[bmw-price-predictor.streamlit.app](https://bmw-price-predictor.streamlit.app)**

> Adjust the sidebar inputs and click **Predict Price** to get an instant estimate.

<br/>

## 📈 Model Performance

| Metric | Value |
|:--|:--|
| **Algorithm** | XGBoost Regressor |
| **R² Score** | 0.96 |
| **RMSE** | ~£1,950 |
| **Training Samples** | 8,185 |
| **Test Samples** | 2,046 |
| **Total Features** | 8 (5 numeric + 3 categorical) |

<br/>

## 🔢 Features Used

| Feature | Type | Description |
|:--|:--|:--|
| `model` | Categorical | BMW series (1–8, M, X, Z) |
| `transmission` | Categorical | Automatic / Manual / Semi-Auto |
| `fuelType` | Categorical | Petrol / Diesel / Hybrid / Other |
| `mileage` | Numeric | Total miles driven |
| `tax` | Numeric | Annual UK road tax (£) |
| `mpg` | Numeric | Combined miles per gallon |
| `engineSize` | Numeric | Engine displacement in litres |
| `car_age` | Numeric | Years since first registration |

<br/>

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|:--|:--|
| **Language** | Python 3.9+ |
| **ML Model** | XGBoost |
| **ML Pipeline** | scikit-learn (Pipeline, ColumnTransformer) |
| **Web Framework** | Streamlit |
| **Visualization** | Plotly |
| **Data Processing** | pandas, NumPy |

</div>

<br/>

## 📂 Project Structure

```
bmw-price-predictor/
│
├── app.py                  # Streamlit web application
├── retrain.py              # Standalone retraining script
├── BMW.ipynb               # Full EDA & model development notebook
│
├── cleaned_data.csv        # Preprocessed dataset (used by app)
├── bmw.csv                 # Raw dataset
├── best_pipeline.pkl       # Serialized trained pipeline
│
├── requirements.txt        # Python dependencies
├── run_app.bat             # Windows one-click launcher
├── SETUP_GUIDE.txt         # Detailed setup & deploy instructions
│
├── .streamlit/
│   └── config.toml         # Custom Streamlit theme
├── .gitignore
└── README.md
```

<br/>

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Beshoy-Atef-Adel/bmw-price-predictor.git
cd bmw-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

> **Windows users:** Double-click `run_app.bat` to launch instantly.

<br/>

## ⚙️ How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌────────────┐
│  User Input  │────▶│  Preprocessing   │────▶│   XGBoost    │────▶│ Predicted  │
│  (8 features)│     │  (scale + encode)│     │  Regressor   │     │   Price    │
└──────────────┘     └──────────────────┘     └──────────────┘     └────────────┘
```

1. **User** adjusts car specs via the sidebar (model, mileage, engine size, etc.)
2. **Pipeline** automatically imputes, scales numeric features & one-hot encodes categoricals
3. **XGBoost** predicts the price based on patterns learned from 10,000+ real listings
4. **Dashboard** displays the estimate with a confidence range, gauge chart, and price comparison

<br/>

## 📸 Screenshots

> *Add screenshots of your app here for a visual preview.*
>
> ```
> ![Dashboard](screenshots/dashboard.png)
> ![Prediction](screenshots/prediction.png)
> ```

<br/>

## 👤 Author

**Beshoy Atef Adel**

[![GitHub](https://img.shields.io/badge/GitHub-Beshoy--Atef--Adel-181717?style=for-the-badge&logo=github)](https://github.com/Beshoy-Atef-Adel)

<br/>

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this useful, give it a star!**

</div>

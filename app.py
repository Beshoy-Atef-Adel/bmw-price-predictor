"""
BMW Used Car Price Predictor — Streamlit App

Run locally:   streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="\U0001f3ce\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# TRAIN MODEL AT STARTUP (cached — trains only once)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    """Train the XGBoost pipeline fresh with current library versions."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_data.csv")
    df = pd.read_csv(csv_path)

    target = 'price'
    drop_cols = [c for c in ['price', 'year', 'price_per_mile'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]

    numeric_features = ['mileage', 'tax', 'mpg', 'engineSize', 'car_age']
    categorical_features = ['model', 'transmission', 'fuelType']
    for col in categorical_features:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            n_estimators=200, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return pipeline, r2, rmse


model, MODEL_R2, MODEL_RMSE = train_model()

# ═══════════════════════════════════════════════════════════
# CONSTANTS — from training data
# ═══════════════════════════════════════════════════════════
MODEL_CHOICES = [
    "1 series", "2 series", "3 series", "4 series", "5 series",
    "6 series", "7 series", "8 series", "m2", "m3", "m4", "m5",
    "m6", "x1", "x2", "x3", "x4", "x5", "x6", "z3", "z4",
]
TRANSMISSION_CHOICES = ["automatic", "manual", "semi-auto"]
FUEL_CHOICES = ["diesel", "hybrid", "other", "petrol"]

MODEL_NAME = "XGBoost Regressor"
MODEL_RMSE = int(round(MODEL_RMSE))

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.title("\U0001f3ce\ufe0f BMW Used Car Price Predictor")
st.markdown(
    "### Predict used BMW prices instantly with our trained Machine Learning model"
)
st.divider()

# ═══════════════════════════════════════════════════════════
# SIDEBAR — INPUT FEATURES
# ═══════════════════════════════════════════════════════════
st.sidebar.header("\U0001f4dd Input Features")
st.sidebar.markdown("Adjust the values below to describe the car.")

st.sidebar.subheader("\U0001f697 Vehicle Identity")

inp_model = st.sidebar.selectbox(
    "BMW Model",
    options=MODEL_CHOICES,
    index=MODEL_CHOICES.index("3 series"),
    help="Select the BMW model or series",
)
inp_transmission = st.sidebar.selectbox(
    "Transmission",
    options=TRANSMISSION_CHOICES,
    index=0,
    help="Gearbox type: automatic, manual, or semi-auto",
)
inp_fuel = st.sidebar.selectbox(
    "Fuel Type",
    options=FUEL_CHOICES,
    index=0,
    help="Engine fuel type",
)
inp_engine = st.sidebar.slider(
    "Engine Size (L)",
    min_value=1.0,
    max_value=6.6,
    value=2.0,
    step=0.1,
    help="Engine displacement in litres (most common: 2.0L)",
)

st.sidebar.subheader("\U0001f4c8 Usage & Specs")

inp_mileage = st.sidebar.number_input(
    "Mileage (miles)",
    min_value=0,
    max_value=250_000,
    value=25_000,
    step=1_000,
    help="Total miles driven",
)
inp_age = st.sidebar.slider(
    "Car Age (years)",
    min_value=1,
    max_value=30,
    value=8,
    step=1,
    help="Years since first registration",
)
inp_tax = st.sidebar.number_input(
    "Road Tax (\u00a3/year)",
    min_value=0,
    max_value=600,
    value=145,
    step=5,
    help="Annual UK road tax",
)
inp_mpg = st.sidebar.slider(
    "MPG (combined)",
    min_value=5.0,
    max_value=120.0,
    value=53.0,
    step=0.5,
    help="Miles per gallon — combined cycle",
)

st.sidebar.divider()
predict_clicked = st.sidebar.button(
    "\U0001f680 Predict Price",
    type="primary",
    use_container_width=True,
)

# ═══════════════════════════════════════════════════════════
# MAIN AREA — TWO COLUMNS
# ═══════════════════════════════════════════════════════════
col1, col2 = st.columns([2, 1])

# ── Right column: Model Info (always visible) ──
with col2:
    st.subheader("\U0001f4ca Model Info")
    st.info(f"**Algorithm:** {MODEL_NAME}")
    st.info(f"**R\u00b2 Score:** {MODEL_R2:.4f}")
    st.info(f"**RMSE:** \u00a3{MODEL_RMSE:,}")
    st.info("**Features:** 8 (5 numeric + 3 categorical)")
    st.info("**Training Data:** 10,231 BMW listings")

    st.subheader("\U0001f4cb Your Input Summary")
    summary_df = pd.DataFrame(
        {
            "Feature": [
                "Model", "Transmission", "Fuel Type", "Engine (L)",
                "Mileage", "Car Age", "Tax (\u00a3)", "MPG",
            ],
            "Value": [
                inp_model, inp_transmission, inp_fuel, inp_engine,
                f"{inp_mileage:,}", inp_age, inp_tax, inp_mpg,
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Left column: Prediction Result ──
with col1:
    st.subheader("\U0001f52e Prediction Result")

    if predict_clicked:
        try:
            # Build input DataFrame matching training schema
            input_df = pd.DataFrame(
                [
                    {
                        "model": inp_model,
                        "transmission": inp_transmission,
                        "fuelType": inp_fuel,
                        "mileage": float(inp_mileage),
                        "tax": float(inp_tax),
                        "mpg": float(inp_mpg),
                        "engineSize": float(inp_engine),
                        "car_age": int(inp_age),
                    }
                ]
            )

            # Run prediction through the full pipeline
            prediction = float(model.predict(input_df)[0])
            prediction = max(prediction, 0)

            # Confidence interval
            low = max(prediction - MODEL_RMSE, 0)
            high = prediction + MODEL_RMSE

            # ── Display result ──
            st.success(f"### \U0001f4b0 Estimated Price: \u00a3{prediction:,.0f}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Low Estimate", f"\u00a3{low:,.0f}")
            m2.metric("\u2b50 Best Estimate", f"\u00a3{prediction:,.0f}")
            m3.metric("High Estimate", f"\u00a3{high:,.0f}")

            st.caption(
                f"*Confidence range based on model RMSE (\u00b1\u00a3{MODEL_RMSE:,})*"
            )

            # ── Gauge chart ──
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    number={"prefix": "\u00a3", "valueformat": ",.0f"},
                    title={"text": "Predicted Price", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 60_000], "tickprefix": "\u00a3",
                                 "tickformat": ",.0f"},
                        "bar": {"color": "#1B69D3"},
                        "bgcolor": "#EEF2F6",
                        "steps": [
                            {"range": [0, 15_000], "color": "#A7D676"},
                            {"range": [15_000, 30_000], "color": "#44BBA4"},
                            {"range": [30_000, 45_000], "color": "#2E86AB"},
                            {"range": [45_000, 60_000], "color": "#0A2647"},
                        ],
                        "threshold": {
                            "line": {"color": "#E94F37", "width": 4},
                            "thickness": 0.8,
                            "value": prediction,
                        },
                    },
                )
            )
            fig.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=60, b=20),
                paper_bgcolor="#FAFBFC",
                font=dict(family="Arial", color="#1F2937"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Comparison context ──
            st.markdown("#### \U0001f4ca Price Context")
            context_data = pd.DataFrame(
                {
                    "Benchmark": [
                        "Your Prediction",
                        "Avg 1 Series",
                        "Avg 3 Series",
                        "Avg 5 Series",
                        "Avg X5",
                    ],
                    "Price (\u00a3)": [prediction, 14_600, 18_500, 22_000, 27_000],
                }
            )
            colors = ["#E94F37", "#2E86AB", "#2E86AB", "#2E86AB", "#2E86AB"]
            fig2 = go.Figure(
                go.Bar(
                    x=context_data["Benchmark"],
                    y=context_data["Price (\u00a3)"],
                    marker_color=colors,
                    text=[f"\u00a3{v:,.0f}" for v in context_data["Price (\u00a3)"]],
                    textposition="outside",
                    textfont=dict(size=13, color="#1F2937"),
                )
            )
            fig2.update_layout(
                yaxis_title="Price (\u00a3)",
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="#FAFBFC",
                plot_bgcolor="#FAFBFC",
                yaxis=dict(gridcolor="#E5E7EB"),
                font=dict(family="Arial", color="#1F2937"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"\u274c **Prediction failed:** {e}")

    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; color: #6B7280;">
                <p style="font-size: 48px; margin-bottom: 10px;">\U0001f449</p>
                <p style="font-size: 18px;">
                    Adjust the features in the <b>sidebar</b> and click
                    <b>Predict Price</b> to see the result.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Built with Streamlit \u00b7 "
    "Powered by scikit-learn + XGBoost"
)

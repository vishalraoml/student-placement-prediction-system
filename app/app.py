import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Performance & Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #0e1117;
        }
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .title-box {
            padding: 18px 22px;
            border-radius: 18px;
            background: linear-gradient(90deg, #1f2937, #111827);
            border: 1px solid #2d3748;
            margin-bottom: 18px;
        }
        .title-text {
            font-size: 34px;
            font-weight: 700;
            color: #ffffff;
        }
        .subtitle-text {
            font-size: 16px;
            color: #cbd5e1;
            margin-top: 6px;
        }
        .card {
            padding: 18px;
            border-radius: 16px;
            background: #111827;
            border: 1px solid #263244;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        .result-box {
            padding: 18px;
            border-radius: 16px;
            background: #0f172a;
            border: 1px solid #334155;
            margin-top: 15px;
        }
        .small-label {
            color: #94a3b8;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load data and artifacts
# -----------------------------
DATA_PATH = "dataset/student_placement_dataset.csv"
MODEL_PATH = "models/salary_model.pkl"
FEATURES_PATH = "models/model_features.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, features, df

model, model_features, raw_df = load_artifacts()

# -----------------------------
# Page Header
# -----------------------------
st.markdown(
    """
    <div class="title-box">
        <div class="title-text">🎓 Student Performance & Placement Prediction System</div>
        <div class="subtitle-text">Predict expected salary package using academic and skill-based student features.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("Student Input Panel")
st.sidebar.caption("Enter student details for prediction")

feature_cols = [c for c in raw_df.columns if c not in ["salary_package_lpa", "placement_status"]]

with st.sidebar.form("prediction_form"):
    input_values = {}

    for col in feature_cols:
        series = raw_df[col]

        # Categorical or low-cardinality numeric
        if series.dtype == "object" or series.nunique() <= 10:
            options = sorted(series.dropna().astype(str).unique().tolist())
            selected = st.selectbox(col.replace("_", " ").title(), options, key=col)
            input_values[col] = selected
        else:
            min_val = float(series.min())
            max_val = float(series.max())
            median_val = float(series.median())

            if pd.api.types.is_integer_dtype(series):
                value = st.number_input(
                    col.replace("_", " ").title(),
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(median_val),
                    key=col
                )
            else:
                value = st.number_input(
                    col.replace("_", " ").title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val,
                    step=0.1,
                    key=col
                )

            input_values[col] = value

    submitted = st.form_submit_button("Predict Salary")

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### Dataset Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{raw_df.shape[0]:,}")
    m2.metric("Columns", f"{raw_df.shape[1]}")
    m3.metric("Model Type", "Random Forest")
    m4.metric("Target", "Salary LPA")

    st.markdown("---")

    # Salary distribution
    st.markdown("### Salary Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(raw_df["salary_package_lpa"], bins=30, kde=True, ax=ax1)
    ax1.set_xlabel("Salary Package (LPA)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Salary Package")
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.markdown("### Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=model_features)
        top_features = importances.sort_values(ascending=False).head(12)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax2)
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        ax2.set_title("Top Feature Importance")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.info("Feature importance chart is not available for this model.")

    st.markdown("### Correlation Heatmap")
    numeric_df = raw_df.select_dtypes(include="number")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax3)
    ax3.set_title("Numeric Feature Correlation")
    st.pyplot(fig3, use_container_width=True)

# -----------------------------
# Prediction Section
# -----------------------------
if submitted:
    input_df = pd.DataFrame([input_values])

    # One-hot encode and align with model features
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    predicted_salary = float(model.predict(input_encoded)[0])

    # Gauge-style progress
    max_salary = 30.0
    progress_value = min(predicted_salary / max_salary, 1.0)

    st.markdown("### Prediction Result")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        st.metric("Predicted Salary", f"₹{predicted_salary:.2f} LPA")
        st.progress(progress_value)
        st.caption(f"Scaled against {max_salary:.0f} LPA for gauge display")

    with result_col2:
        if predicted_salary < 12:
            st.success("Prediction Category: Entry Level")
        elif predicted_salary < 18:
            st.info("Prediction Category: Mid Level")
        else:
            st.warning("Prediction Category: High Package")

        st.write("Model output suggests the expected compensation band for the entered student profile.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Input Summary")
    st.dataframe(input_df, use_container_width=True)
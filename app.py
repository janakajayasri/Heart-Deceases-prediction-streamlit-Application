"""
Heart Disease Prediction — Advanced Streamlit App
Based on: Heart_Disease_Prediction.ipynb
"""

import io
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .metric-card {
            background: linear-gradient(135deg, #1e3a5f, #0d6efd22);
            border: 1px solid #0d6efd55;
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
        }
        .metric-card h3 { margin: 0; font-size: 1.8rem; color: #58a6ff; }
        .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #8b949e; }

        .risk-high   { color: #ff4b4b; font-weight: 700; font-size: 1.4rem; }
        .risk-low    { color: #21c354; font-weight: 700; font-size: 1.4rem; }
        .section-hdr { border-left: 4px solid #0d6efd; padding-left: 10px; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
for key in ("models_trained", "best_model", "feature_cols"):
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pretrained_model():
    try:
        return joblib.load("heart_disease_model.pkl")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_pretrained_scaler():
    try:
        return joblib.load("scaler.pkl")
    except Exception:
        return None


PALETTE = {"LR": "#4e79a7", "RF": "#f28e2b", "XGB": "#e15759"}

# Reference categories dropped by pd.get_dummies(drop_first=True) for this dataset
CATEGORICAL_OPTIONS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"],
}


@st.cache_data(show_spinner=False)
def load_data(uploaded):
    return pd.read_csv(uploaded)


def preprocess(df):
    df = df.drop_duplicates()
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop("HeartDisease", axis=1)
    y = df_enc["HeartDisease"]
    return df_enc, X, y


@st.cache_resource(show_spinner=False)
def train_models(df_hash, _X, _y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    rf = RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42)
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=15,
        subsample=0.8, colsample_bytree=0.8, gamma=0.4,
        reg_alpha=0.5, reg_lambda=7, eval_metric="logloss", random_state=42,
    )

    models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        prob = m.predict_proba(X_te)[:, 1]
        results[name] = {
            "model": m,
            "acc": accuracy_score(y_te, pred),
            "auc": roc_auc_score(y_te, prob),
            "pred": pred,
            "prob": prob,
            "report": classification_report(y_te, pred, output_dict=True),
            "cm": confusion_matrix(y_te, pred),
            "fpr_tpr": roc_curve(y_te, prob),
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, m in models.items():
        scores = cross_val_score(m, _X, _y, cv=cv, scoring="accuracy")
        results[name]["cv_mean"] = scores.mean()
        results[name]["cv_std"] = scores.std()

    best_name = max(results, key=lambda n: results[n]["auc"])
    return results, X_te, y_te, X_tr, best_name


def build_input_row(inputs, feature_cols):
    """Build a single-row, one-hot-encoded DataFrame matching feature_cols,
    from raw patient inputs. No external scaling is applied — every model in
    this app (pretrained XGBoost, and the LR/RF/XGB trained in-app) was fit
    on unscaled/raw-encoded features, so scaling here would corrupt inputs."""
    row = {col: 0 for col in feature_cols}
    row["Age"] = inputs["Age"]
    row["RestingBP"] = inputs["RestingBP"]
    row["Cholesterol"] = inputs["Cholesterol"]
    row["FastingBS"] = inputs["FastingBS"]
    row["MaxHR"] = inputs["MaxHR"]
    row["Oldpeak"] = inputs["Oldpeak"]

    if inputs["Sex"] == "M" and "Sex_M" in row:
        row["Sex_M"] = 1

    cp_col = f"ChestPainType_{inputs['ChestPainType']}"
    if cp_col in row:
        row[cp_col] = 1

    ecg_col = f"RestingECG_{inputs['RestingECG']}"
    if ecg_col in row:
        row[ecg_col] = 1

    if inputs["ExerciseAngina"] == "Y" and "ExerciseAngina_Y" in row:
        row["ExerciseAngina_Y"] = 1

    slope_col = f"ST_Slope_{inputs['ST_Slope']}"
    if slope_col in row:
        row[slope_col] = 1

    return pd.DataFrame([row])[feature_cols]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
    st.title("🫀 Heart Disease Predictor")
    st.markdown("---")
    uploaded = st.file_uploader("Upload heart.csv", type=["csv"])
    st.markdown("---")
    use_pretrained = st.checkbox("🔄 Use Pre-trained Model", value=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Data Explorer", "🤖 Model Training", "🔬 Prediction", "📥 Export"],
    )
    st.markdown("---")
    st.caption("Built from Heart_Disease_Prediction.ipynb")

if uploaded is None:
    st.info("Upload heart.csv to get started.")
    st.stop()

df_raw = load_data(uploaded)
df_enc, X, y = preprocess(df_raw)

# ----------------------------------------------------------------------------
# Data Explorer
# ----------------------------------------------------------------------------
if page == "📊 Data Explorer":
    st.markdown("<h2 class='section-hdr'>Data Explorer</h2>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        (c1, c2, c3, c4),
        ("Rows", "Columns", "Positive Cases", "Missing Values"),
        (df_raw.shape[0], df_raw.shape[1], int(df_raw["HeartDisease"].sum()), int(df_raw.isnull().sum().sum())),
    ):
        col.markdown(f"<div class='metric-card'><h3>{value}</h3><p>{label}</p></div>", unsafe_allow_html=True)

    st.markdown("#### Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(df_raw.describe(), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df_raw, x="HeartDisease", ax=ax, palette=["#21c354", "#ff4b4b"])
        ax.set_xticklabels(["No Disease", "Disease"])
        st.pyplot(fig)

    with col_b:
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df_enc.corr(), cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

    st.markdown("#### Numeric Feature Distributions by Diagnosis")
    numeric_cols = [c for c in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"] if c in df_raw.columns]
    sel_col = st.selectbox("Feature", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(data=df_raw, x=sel_col, hue="HeartDisease", kde=True, ax=ax, palette=["#21c354", "#ff4b4b"])
    st.pyplot(fig)

# ----------------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------------
elif page == "🤖 Model Training":
    st.markdown("<h2 class='section-hdr'>Model Training</h2>", unsafe_allow_html=True)
    st.caption("Trains Logistic Regression, Random Forest, and XGBoost on the uploaded dataset.")

    if st.button("🚀 Train Models"):
        df_hash = int(pd.util.hash_pandas_object(df_enc).sum())
        with st.spinner("Training models..."):
            results, X_te, y_te, X_tr, best_name = train_models(df_hash, X, y)
        st.session_state.models_trained = results
        st.session_state.best_model = best_name
        st.session_state.feature_cols = list(X.columns)
        st.success(f"Training complete. Best model: **{best_name}**")

    
       

        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots()
        colors = ["#4e79a7", "#f28e2b", "#e15759"]
        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = res["fpr_tpr"]
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### Confusion Matrices")
        cols = st.columns(len(results))
        for col, (name, res) in zip(cols, results.items()):
            with col:
                st.caption(name)
                fig, ax = plt.subplots()
                sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                st.pyplot(fig)

        st.markdown("#### Feature Importance")
        tree_models = {n: r for n, r in results.items() if hasattr(r["model"], "feature_importances_")}
        if tree_models:
            sel = st.selectbox("Model", list(tree_models.keys()))
            imp = pd.DataFrame(
                {"Feature": X.columns, "Importance": tree_models[sel]["model"].feature_importances_}
            ).sort_values("Importance", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
            st.pyplot(fig)
    else:
        st.info("Click 'Train Models' to get started, or enable the pre-trained model in the sidebar.")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
elif page == "🔬 Prediction":
    st.markdown("<h2 class='section-hdr'>Patient Risk Prediction</h2>", unsafe_allow_html=True)

    if use_pretrained and st.session_state.models_trained is None:
        model = load_pretrained_model()
        if model is not None:
            cols = list(getattr(model, "feature_names_in_", X.columns))
            st.session_state.models_trained = {"Pre-trained XGBoost": {"model": model}}
            st.session_state.feature_cols = cols
            st.session_state.best_model = "Pre-trained XGBoost"
            st.success("Pre-trained model loaded!")
        else:
            st.error("Could not find heart_disease_model.pkl next to app.py.")

    if st.session_state.models_trained is None:
        st.warning("Train models on the 'Model Training' page, or enable the pre-trained model in the sidebar.")
        st.stop()

    scaler = load_pretrained_scaler()
    with st.expander("ℹ️ About scaler.pkl"):
        st.write(
            "A `scaler.pkl` was saved alongside the model, but it is **not applied** here. "
            "The saved XGBoost model (and the Random Forest / Logistic Regression trained in this "
            "app) were all fit on unscaled, one-hot-encoded features — Logistic Regression uses its "
            "own internal scaler inside its Pipeline. Applying the external scaler to raw inputs "
            "before prediction would shift the feature values onto the wrong scale and produce "
            f"incorrect results. Loaded scaler: `{type(scaler).__name__ if scaler is not None else 'not found'}`."
        )

    results = st.session_state.models_trained
    feature_cols = st.session_state.feature_cols
    model_names = list(results.keys())
    default_idx = model_names.index(st.session_state.best_model) if st.session_state.best_model in model_names else 0
    selected_name = st.selectbox("Model to use", model_names, index=default_idx)
    model = results[selected_name]["model"]

    st.markdown("#### Patient Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 100, 50)
        sex = st.selectbox("Sex", CATEGORICAL_OPTIONS["Sex"])
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        cholesterol = st.slider("Cholesterol (mg/dl)", 0, 600, 200)
    with c2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        chest_pain = st.selectbox("Chest Pain Type", CATEGORICAL_OPTIONS["ChestPainType"])
        resting_ecg = st.selectbox("Resting ECG", CATEGORICAL_OPTIONS["RestingECG"])
        max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    with c3:
        exercise_angina = st.selectbox("Exercise-Induced Angina", CATEGORICAL_OPTIONS["ExerciseAngina"])
        oldpeak = st.slider("Oldpeak (ST Depression)", -3.0, 7.0, 0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", CATEGORICAL_OPTIONS["ST_Slope"])

    inputs = {
        "Age": age, "Sex": sex, "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "ChestPainType": chest_pain, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope,
    }

    if st.button("Predict Risk", type="primary"):
        row = build_input_row(inputs, feature_cols)
        prob = model.predict_proba(row)[0, 1]
        pred = int(prob >= 0.5)

        col_res, col_row = st.columns([1, 2])
        with col_res:
            if pred == 1:
                st.markdown(f"<p class='risk-high'>⚠️ High Risk</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='risk-low'>✅ Low Risk</p>", unsafe_allow_html=True)
            st.metric("Predicted Probability of Heart Disease", f"{prob:.1%}")
        with col_row:
            st.caption("Model input (one-hot encoded)")
            st.dataframe(row, use_container_width=True)

# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
elif page == "📥 Export":
    st.markdown("<h2 class='section-hdr'>Export</h2>", unsafe_allow_html=True)

    if st.session_state.models_trained:
        model_names = list(st.session_state.models_trained.keys())
        sel = st.selectbox("Model to export", model_names)
        model_obj = st.session_state.models_trained[sel]["model"]

        buf = io.BytesIO()
        joblib.dump(model_obj, buf)
        buf.seek(0)
        st.download_button(
            "⬇️ Download Model (.pkl)",
            data=buf,
            file_name=f"{sel.replace(' ', '_').lower()}.pkl",
            mime="application/octet-stream",
        )

        if "report" in st.session_state.models_trained[sel]:
            report_df = pd.DataFrame(st.session_state.models_trained[sel]["report"]).T
            st.dataframe(report_df, use_container_width=True)
            csv_buf = io.StringIO()
            report_df.to_csv(csv_buf)
            st.download_button(
                "⬇️ Download Classification Report (.csv)",
                data=csv_buf.getvalue(),
                file_name=f"{sel.replace(' ', '_').lower()}_report.csv",
                mime="text/csv",
            )
    else:
        st.warning("No trained model in memory yet. Train a model, or enable the pre-trained model and run a prediction first.")

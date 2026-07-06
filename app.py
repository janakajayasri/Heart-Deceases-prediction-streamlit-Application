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

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────
for key in ("models_trained", "best_model", "feature_cols", "df_encoded"):
    if key not in st.session_state:
        st.session_state[key] = None

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
PALETTE = {"LR": "#4e79a7", "RF": "#f28e2b", "XGB": "#e15759"}

@st.cache_data(show_spinner=False)
def load_data(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def preprocess(df: pd.DataFrame):
    df = df.drop_duplicates()
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop("HeartDisease", axis=1)
    y = df_enc["HeartDisease"]
    return df_enc, X, y


@st.cache_resource(show_spinner=False)
def train_models(df_hash: str, _X, _y):
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


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60
    )
    st.title("🫀 Heart Disease\nPredictor")
    st.markdown("---")

    uploaded = st.file_uploader("Upload `heart.csv`", type=["csv"])
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Data Explorer", "🤖 Model Training", "🔬 Prediction", "📥 Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built from Heart_Disease_Prediction.ipynb")

# ──────────────────────────────────────────────
# Gate: need data
# ──────────────────────────────────────────────
if uploaded is None:
    st.markdown("## 👋 Welcome")
    st.info("Upload **heart.csv** in the sidebar to get started.", icon="📂")
    st.markdown(
        """
        **Expected columns:**  
        `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`,  
        `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`,  
        `Oldpeak`, `ST_Slope`, `HeartDisease`
        """
    )
    st.stop()

df_raw = load_data(uploaded)
df_enc, X, y = preprocess(df_raw)
df_hash = str(df_raw.shape) + str(df_raw.columns.tolist())

# ══════════════════════════════════════════════
# PAGE 1 — Data Explorer
# ══════════════════════════════════════════════
if page == "📊 Data Explorer":
    st.markdown("<h2 class='section-hdr'>Data Explorer</h2>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Total Records", len(df_raw)),
        (c2, "Features", df_raw.shape[1] - 1),
        (c3, "Heart Disease (%)", f"{df_raw['HeartDisease'].mean()*100:.1f}%"),
        (c4, "Missing Values", int(df_raw.isnull().sum().sum())),
    ]:
        col.markdown(
            f"<div class='metric-card'><h3>{val}</h3><p>{label}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📈 Distributions", "🔥 Correlations", "📦 Boxplots"])

    with tab1:

        buf = io.StringIO()
        df_raw.info(buf=buf)
        with st.expander("DataFrame .info()"):
            st.text(buf.getvalue())
        with st.expander("Descriptive Statistics"):
            st.dataframe(df_raw.describe().T.style.background_gradient(cmap="Blues"))
            

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots()
            sns.countplot(x="HeartDisease", data=df_raw, palette=["#21c354", "#ff4b4b"], ax=ax)
            ax.set_xticklabels(["No Disease", "Heart Disease"])
            ax.set_title("Target Distribution")
            st.pyplot(fig)
        with col_b:
            fig, ax = plt.subplots()
            sns.histplot(df_raw["Age"], bins=20, kde=True, color="#4e79a7", ax=ax)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

        num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        chosen = st.selectbox("Select numeric column", num_cols)
        fig, ax = plt.subplots(figsize=(7,5))
        sns.histplot(
                    data=df_raw,
                    x=chosen,
                    bins=20,
                    kde=True,
                    color="#4e79a7",
                    edgecolor="black",
                    ax=ax
                )
        ax.set_title(f"{chosen} Distribution")
        ax.set_xlabel(chosen)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_enc.corr(numeric_only=True), annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap (encoded)")
        st.pyplot(fig)

    with tab4:
        num_cols_bp = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        fig, axes = plt.subplots(1, len(num_cols_bp), figsize=(16, 4))
        for ax, col in zip(axes, num_cols_bp):
            sns.boxplot(y=df_raw[col], ax=ax, color="#4e79a7")
            ax.set_title(col, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════
# PAGE 2 — Model Training
# ══════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("<h2 class='section-hdr'>Model Training & Evaluation</h2>", unsafe_allow_html=True)

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training Logistic Regression, Random Forest & XGBoost…"):
            results, X_te, y_te, X_tr, best_name = train_models(df_hash, X, y)
        st.session_state.models_trained = results
        st.session_state.best_model = best_name
        st.session_state.feature_cols = list(X.columns)
        st.session_state.df_encoded = df_enc
        st.success(f"✅ Training complete! Best model: **{best_name}**")

    if st.session_state.models_trained is None:
        st.info("Click **Train All Models** to begin.", icon="⏳")
        st.stop()

    results, X_te, y_te, X_tr, best_name = train_models(df_hash, X, y)
    model_names = list(results.keys())

    # ── Accuracy / AUC summary ──
    st.markdown("### Model Comparison")
    cols = st.columns(3)
    for col, name in zip(cols, model_names):
        r = results[name]
        tag = " 🏆" if name == best_name else ""
        col.markdown(
            f"<div class='metric-card'>"
            f"<h3>{r['acc']*100:.1f}%</h3>"
            f"<p>{name}{tag}<br>AUC: {r['auc']:.3f} | CV: {r['cv_mean']:.3f}±{r['cv_std']:.3f}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    tab_roc, tab_cm, tab_cr, tab_fi, tab_cv = st.tabs(
        ["📈 ROC Curves", "🔲 Confusion Matrix", "📝 Classification Report",
         "⭐ Feature Importance", "🔁 Cross-Validation"]
    )

    with tab_roc:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = list(PALETTE.values())
        for (name, r), color in zip(results.items(), colors):
            fpr, tpr, _ = r["fpr_tpr"]
            ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", color=color, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with tab_cm:
        chosen_m = st.selectbox("Model", model_names)
        cm = results[chosen_m]["cm"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        ax.set_title(f"Confusion Matrix — {chosen_m}")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        st.pyplot(fig)

    with tab_cr:
        chosen_m2 = st.selectbox("Model ", model_names, key="cr")
        rep = results[chosen_m2]["report"]
        rep_df = pd.DataFrame(rep).T.drop(columns=["support"], errors="ignore")
        st.dataframe(rep_df.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.3f}"),
                     use_container_width=True)

    with tab_fi:
        fi_models = [n for n in model_names if hasattr(results[n]["model"], "feature_importances_")]
        if fi_models:
            chosen_fi = st.selectbox("Model  ", fi_models)
            m = results[chosen_fi]["model"]
            imp = pd.DataFrame({"Feature": X.columns, "Importance": m.feature_importances_})
            imp = imp.sort_values("Importance", ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(data=imp, x="Importance", y="Feature", palette="Blues_r", ax=ax)
            ax.set_title(f"Top 15 Feature Importances — {chosen_fi}")
            st.pyplot(fig)
        else:
            st.info("Feature importance is not available for Logistic Regression.")

    with tab_cv:
        cv_data = {
            n: {"Mean": results[n]["cv_mean"], "Std": results[n]["cv_std"]}
            for n in model_names
        }
        cv_df = pd.DataFrame(cv_data).T
        st.dataframe(cv_df.style.background_gradient(cmap="Blues").format("{:.4f}"),
                     use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(model_names, cv_df["Mean"],
               yerr=cv_df["Std"], capsize=6,
               color=list(PALETTE.values()), alpha=0.85)
        ax.set_ylim(0.7, 1.0)
        ax.set_ylabel("CV Accuracy")
        ax.set_title("5-Fold Cross-Validation Accuracy")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

# ══════════════════════════════════════════════
# PAGE 3 — Prediction
# ══════════════════════════════════════════════
elif page == "🔬 Prediction":
    st.markdown("<h2 class='section-hdr'>Patient Risk Prediction</h2>", unsafe_allow_html=True)

    if st.session_state.models_trained is None:
        st.warning("Please train models first on the **Model Training** page.", icon="⚠️")
        st.stop()

    results = st.session_state.models_trained
    feature_cols = st.session_state.feature_cols
    best_name = st.session_state.best_model

    st.markdown("#### Enter Patient Data")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 20, 100, 55)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.slider("Resting BP (mm Hg)", 80, 200, 130)
    with c2:
        cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    with c3:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        model_choice = st.selectbox("Predict with model", list(results.keys()),
                                    index=list(results.keys()).index(best_name))

    if st.button("🫀 Predict Risk", type="primary", use_container_width=True):
        # Build raw row
        row = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope, "HeartDisease": 0,
        }
        inp_df = pd.DataFrame([row])
        inp_enc = pd.get_dummies(inp_df, drop_first=True).drop("HeartDisease", axis=1)

        # Align columns
        for col in feature_cols:
            if col not in inp_enc.columns:
                inp_enc[col] = 0
        inp_enc = inp_enc[feature_cols]

        model = results[model_choice]["model"]
        pred = model.predict(inp_enc)[0]
        prob = model.predict_proba(inp_enc)[0][1]

        st.markdown("---")
        left, right = st.columns([1, 2])
        with left:
            risk_class = "risk-high" if pred == 1 else "risk-low"
            label = "⚠️ HIGH RISK" if pred == 1 else "✅ LOW RISK"
            st.markdown(
                f"<div class='metric-card'>"
                f"<h3 class='{risk_class}'>{label}</h3>"
                f"<p>Probability: <b>{prob*100:.1f}%</b><br>Model: {model_choice}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with right:
            fig, ax = plt.subplots(figsize=(5, 1.5))
            ax.barh(["Heart Disease Risk"], [prob], color="#ff4b4b" if pred else "#21c354", height=0.4)
            ax.barh(["Heart Disease Risk"], [1 - prob], left=[prob], color="#eee", height=0.4)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1)
            ax.set_xlabel("Probability")
            ax.set_title("Risk Gauge")
            ax.spines[["top", "right", "left"]].set_visible(False)
            st.pyplot(fig)

        if pred == 1:
            st.error("This patient shows signs of elevated heart disease risk. "
                     "Clinical evaluation is strongly advised.")
        else:
            st.success("Low predicted risk of heart disease based on provided features.")

# ══════════════════════════════════════════════
# PAGE 4 — Export
# ══════════════════════════════════════════════
elif page == "📥 Export":
    st.markdown("<h2 class='section-hdr'>Export Models & Reports</h2>", unsafe_allow_html=True)

    if st.session_state.models_trained is None:
        st.warning("Train models first.", icon="⚠️")
        st.stop()

    results = st.session_state.models_trained
    best_name = st.session_state.best_model

    st.markdown("### Download Trained Models")
    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.items()):
        buf = io.BytesIO()
        joblib.dump(r["model"], buf)
        tag = " 🏆" if name == best_name else ""
        col.download_button(
            label=f"⬇ {name}{tag}",
            data=buf.getvalue(),
            file_name=f"{'_'.join(name.lower().split())}_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Download Metrics Report (CSV)")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(r["acc"], 4),
            "AUC": round(r["auc"], 4),
            "CV Mean": round(r["cv_mean"], 4),
            "CV Std": round(r["cv_std"], 4),
            "Best": name == best_name,
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)
    st.download_button(
        "⬇ Download Metrics CSV",
        metrics_df.to_csv(index=False),
        file_name="model_metrics.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Download Processed Dataset")
    df_enc_out = st.session_state.df_encoded
    if df_enc_out is not None:
        st.download_button(
            "⬇ Download Encoded Dataset CSV",
            df_enc_out.to_csv(index=False),
            file_name="heart_encoded.csv",
            mime="text/csv",
        )
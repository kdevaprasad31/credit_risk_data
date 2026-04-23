# app.py - Fraud Detection System | ABA Assignment
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0a0f1e; color: #e0e6f0; }
    .stApp { background-color: #0a0f1e; }

    .header-box {
        background: linear-gradient(135deg, #0d1b3e 0%, #1a2f5a 100%);
        border: 1px solid #1e3a6e;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 32px rgba(0,100,255,0.08);
    }
    .header-box h1 {
        font-family: 'Space Mono', monospace;
        color: #4fc3f7;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -1px;
    }
    .header-box p {
        color: #7a9cc4;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    .metric-card {
        background: #0d1b3e;
        border: 1px solid #1e3a6e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,80,200,0.08);
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        color: #4fc3f7;
        font-weight: 700;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #7a9cc4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .risk-high   { color: #ff5252; font-weight: 700; }
    .risk-medium { color: #ffb300; font-weight: 700; }
    .risk-low    { color: #69f0ae; font-weight: 700; }

    .section-title {
        font-family: 'Space Mono', monospace;
        color: #4fc3f7;
        font-size: 1.1rem;
        border-left: 3px solid #4fc3f7;
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d1b3e;
        border-right: 1px solid #1e3a6e;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1565c0, #0288d1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1976d2, #039be5);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(79,195,247,0.25);
    }
    .stDataFrame { border-radius: 8px; }
    label, .stSelectbox label, .stSlider label { color: #7a9cc4 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛡️ Fraud Detection System</h1>
    <p>Built for ABA Project &nbsp;|&nbsp; Credit Card Fraud Analysis &nbsp;|&nbsp; Random Forest + Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    st.markdown("---")
    st.markdown("### 🔧 Model Settings")
    n_estimators = st.slider("RF Trees", 50, 300, 200, 50)
    test_size    = st.slider("Test Size %", 10, 40, 20, 5)
    threshold    = st.slider("Fraud Threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    run_btn = st.button("🚀 Train & Evaluate")

# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────
def risk_level(p):
    if p < 0.3:   return "🟢 Low Risk"
    elif p < 0.7: return "🟡 Medium Risk"
    else:         return "🔴 High Risk"

@st.cache_data
def load_and_train(data_bytes, n_est, ts):
    import io
    df = pd.read_csv(io.BytesIO(data_bytes))

    fraud     = df[df['Class'] == 1]
    non_fraud = df[df['Class'] == 0].sample(len(fraud), random_state=42)
    df_bal    = pd.concat([fraud, non_fraud])

    scaler = StandardScaler()
    df_bal = df_bal.copy()
    df_bal['Amount'] = scaler.fit_transform(df_bal[['Amount']])
    df_bal['Time']   = scaler.fit_transform(df_bal[['Time']])

    X = df_bal.drop('Class', axis=1)
    y = df_bal['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts/100, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:,1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=n_est,
                                class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:,1]

    return (X, X_test, y_test,
            y_pred_lr, y_prob_lr,
            y_pred_rf, y_prob_rf,
            rf, lr, df)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
# Load data on upload
if uploaded_file is None:
    st.info("👈 Upload your **creditcard.csv** in the sidebar to begin.")
    st.stop()

raw_bytes = uploaded_file.read()

if True:   # auto-run on upload
    with st.spinner("Training models... please wait"):
        (X, X_test, y_test,
         y_pred_lr, y_prob_lr,
         y_pred_rf, y_prob_rf,
         rf, lr, df) = load_and_train(raw_bytes, n_estimators, test_size)

    # ── TABS ──────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "🤖 Model Results", "📈 Charts", "🔍 Predict"
    ])

    # ── TAB 1: OVERVIEW ───────────────────────
    with tab1:
        st.markdown('<div class="section-title">Dataset Overview</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        total  = len(df)
        fraud  = df['Class'].sum()
        normal = total - fraud

        for col, val, lbl in zip(
            [c1,c2,c3,c4],
            [total, fraud, normal, f"{fraud/total*100:.2f}%"],
            ["Total Records","Fraud Cases","Normal Cases","Fraud Rate"]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Sample Data</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    # ── TAB 2: MODEL RESULTS ──────────────────
    with tab2:
        st.markdown('<div class="section-title">Random Forest Performance</div>',
                    unsafe_allow_html=True)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        acc_lr = accuracy_score(y_test, y_pred_lr)

        m1, m2 = st.columns(2)
        m1.markdown(f"""<div class="metric-card">
            <div class="value">{acc_rf*100:.1f}%</div>
            <div class="label">Random Forest Accuracy</div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card">
            <div class="value">{acc_lr*100:.1f}%</div>
            <div class="label">Logistic Regression Accuracy</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Classification Report — Random Forest</div>',
                    unsafe_allow_html=True)
        report = classification_report(y_test, y_pred_rf, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2),
                     use_container_width=True)

        st.markdown('<div class="section-title">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots(figsize=(4,3))
        fig.patch.set_facecolor('#0d1b3e')
        ax.set_facecolor('#0d1b3e')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=ax, linewidths=0.5,
                    annot_kws={"color":"white","size":14})
        ax.set_xlabel("Predicted", color='#7a9cc4')
        ax.set_ylabel("Actual",    color='#7a9cc4')
        ax.tick_params(colors='#7a9cc4')
        st.pyplot(fig, use_container_width=False)

    # ── TAB 3: CHARTS ─────────────────────────
    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">ROC Curve</div>',
                        unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor('#0d1b3e')
            ax.set_facecolor('#0d1b3e')
            ax.plot(fpr, tpr, color='#4fc3f7',
                    lw=2, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0,1],[0,1],'--', color='#546e7a', lw=1)
            ax.set_xlabel("False Positive Rate", color='#7a9cc4')
            ax.set_ylabel("True Positive Rate",  color='#7a9cc4')
            ax.set_title("ROC Curve", color='#4fc3f7')
            ax.legend(facecolor='#0d1b3e', labelcolor='white')
            ax.tick_params(colors='#7a9cc4')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e3a6e')
            st.pyplot(fig)

        with c2:
            st.markdown('<div class="section-title">Top Feature Importances</div>',
                        unsafe_allow_html=True)
            importance = pd.Series(rf.feature_importances_,
                                   index=X.columns).sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor('#0d1b3e')
            ax.set_facecolor('#0d1b3e')
            importance.plot(kind='barh', ax=ax, color='#4fc3f7')
            ax.set_xlabel("Importance", color='#7a9cc4')
            ax.set_title("Top 10 Features", color='#4fc3f7')
            ax.tick_params(colors='#7a9cc4')
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e3a6e')
            st.pyplot(fig)

        st.markdown('<div class="section-title">Fraud Risk Score Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,3))
        fig.patch.set_facecolor('#0d1b3e')
        ax.set_facecolor('#0d1b3e')
        ax.hist(y_prob_rf[y_test==0], bins=40, alpha=0.7,
                color='#69f0ae', label='Normal')
        ax.hist(y_prob_rf[y_test==1], bins=40, alpha=0.7,
                color='#ff5252', label='Fraud')
        ax.set_xlabel("Fraud Risk Score", color='#7a9cc4')
        ax.set_ylabel("Count", color='#7a9cc4')
        ax.legend(facecolor='#0d1b3e', labelcolor='white')
        ax.tick_params(colors='#7a9cc4')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a6e')
        st.pyplot(fig)

    # ── TAB 4: PREDICT ────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Predict on Test Set</div>',
                    unsafe_allow_html=True)

        results = pd.DataFrame({
            'Actual':          y_test.values,
            'Predicted':       y_pred_rf,
            'Fraud Risk Score': np.round(y_prob_rf, 4)
        })
        results['Risk Level'] = results['Fraud Risk Score'].apply(risk_level)
        results['Result'] = results.apply(
            lambda r: '✅ Correct' if r['Actual'] == r['Predicted'] else '❌ Wrong', axis=1)

        # Filter
        col1, col2 = st.columns(2)
        filter_risk   = col1.selectbox("Filter by Risk Level",
                                       ["All","🔴 High Risk","🟡 Medium Risk","🟢 Low Risk"])
        filter_result = col2.selectbox("Filter by Result",
                                       ["All","✅ Correct","❌ Wrong"])

        filtered = results.copy()
        if filter_risk   != "All": filtered = filtered[filtered['Risk Level'] == filter_risk]
        if filter_result != "All": filtered = filtered[filtered['Result']     == filter_result]

        st.dataframe(filtered.head(100), use_container_width=True)

        st.markdown(f"**Showing {len(filtered)} records**")
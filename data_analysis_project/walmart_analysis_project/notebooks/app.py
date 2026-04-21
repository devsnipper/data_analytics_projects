import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    .stApp {
        background: linear-gradient(135deg, #0071ce 0%, #004f93 100%);
        min-height: 100vh;
    }

    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }

    .header-card {
        background: white;
        border-radius: 16px;
        padding: 2rem 2rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }

    .result-good {
        background: linear-gradient(135deg, #00b96b, #00875a);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .result-bad {
        background: linear-gradient(135deg, #ff4d4f, #cf1322);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .metric-box {
        background: #f0f7ff;
        border: 1px solid #cce4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0071ce;
    }

    .badge-accuracy {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    div[data-testid="stNumberInput"] label {
        font-weight: 600;
        color: #374151;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0071ce, #004f93);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.9;
    }

    .prob-bar-container {
        background: #e5e7eb;
        border-radius: 999px;
        height: 12px;
        margin: 0.5rem 0;
        overflow: hidden;
    }

    .footer-note {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 0.78rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_data.joblib")
    return joblib.load(model_path)

model_data = load_model()
pipeline   = model_data["pipeline"]
accuracy   = model_data["accuracy"]

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div style="font-size:2.5rem; margin-bottom:0.4rem">🛒</div>
    <h1 style="margin:0; font-size:1.6rem; color:#0071ce; font-weight:700;">
        Walmart Sales Predictor
    </h1>
    <p style="margin:0.4rem 0 0.6rem; color:#6b7280; font-size:0.95rem;">
        Predict whether a product will be a <strong>High Seller</strong> based on quantity & average price.
    </p>
    <span class="badge-accuracy">✅ Model Accuracy: {:.1f}%</span>
</div>
""".format(accuracy * 100), unsafe_allow_html=True)

# ── Input card ──────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📦 Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    quantity = st.number_input(
        "🔢 Quantity Sold",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Number of units sold in the order"
    )

with col2:
    avg_price = st.number_input(
        "💵 Average Price (USD)",
        min_value=1.0,
        max_value=5000.0,
        value=49.99,
        step=0.01,
        format="%.2f",
        help="Average selling price per unit"
    )

# Estimated sales
est_sales = quantity * avg_price
st.markdown(f"""
<div style="display:flex; gap:1rem; margin-top:1rem;">
    <div class="metric-box" style="flex:1">
        <div class="metric-label">Estimated Sales</div>
        <div class="metric-value">${est_sales:,.2f}</div>
    </div>
    <div class="metric-box" style="flex:1">
        <div class="metric-label">Units</div>
        <div class="metric-value">{quantity}</div>
    </div>
    <div class="metric-box" style="flex:1">
        <div class="metric-label">Unit Price</div>
        <div class="metric-value">${avg_price:,.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Sales Performance"):
    X_input   = np.array([[quantity, avg_price]])
    prediction = pipeline.predict(X_input)[0]
    probability = pipeline.predict_proba(X_input)[0]

    prob_high = probability[1] * 100
    prob_low  = probability[0] * 100

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-good">
            🚀 HIGH SELLER<br>
            <span style="font-size:1rem; font-weight:400; opacity:0.9;">
                This product is predicted to sell well
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-bad">
            ⚠️ LOW SELLER<br>
            <span style="font-size:1rem; font-weight:400; opacity:0.9;">
                This product may underperform — consider adjusting pricing or quantity
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Confidence Breakdown**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style="margin-bottom:0.5rem">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:600; color:#00875a;">
                <span>✅ High Seller</span><span>{prob_high:.1f}%</span>
            </div>
            <div class="prob-bar-container">
                <div style="width:{prob_high}%; height:100%; background:#00b96b; border-radius:999px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="margin-bottom:0.5rem">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:600; color:#cf1322;">
                <span>❌ Low Seller</span><span>{prob_low:.1f}%</span>
            </div>
            <div class="prob-bar-container">
                <div style="width:{prob_low}%; height:100%; background:#ff4d4f; border-radius:999px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Insight
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        if prob_high >= 80:
            insight = "🟢 **Strong signal.** High confidence this product will perform above the median sales threshold."
        else:
            insight = "🟡 **Moderate signal.** The model leans towards a good seller, but monitor performance closely."
    else:
        if prob_low >= 80:
            insight = "🔴 **Weak signal.** High confidence this product will underperform. Consider reducing price or stocking fewer units."
        else:
            insight = "🟠 **Borderline.** The product is close to the threshold. Small changes in price or quantity could flip the outcome."

    st.info(insight)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Batch section ───────────────────────────────────────────────────────────────
with st.expander("📋 Try Multiple Scenarios at Once"):
    st.markdown("Quick scenarios to explore:")

    scenarios = [
        {"name": "Budget Item", "qty": 10, "price": 12.99},
        {"name": "Mid-Range Product", "qty": 5, "price": 89.99},
        {"name": "Premium Item", "qty": 2, "price": 349.99},
        {"name": "High Volume, Low Cost", "qty": 14, "price": 6.50},
        {"name": "Luxury Product", "qty": 1, "price": 999.00},
    ]

    results = []
    for s in scenarios:
        X_s = np.array([[s["qty"], s["price"]]])
        pred = pipeline.predict(X_s)[0]
        prob = pipeline.predict_proba(X_s)[0][1] * 100
        results.append({
            "Product Scenario": s["name"],
            "Qty": s["qty"],
            "Avg Price ($)": f"{s['price']:.2f}",
            "Prediction": "✅ High Seller" if pred == 1 else "❌ Low Seller",
            "Confidence (%)": f"{prob:.1f}%"
        })

    import pandas as pd
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    Built with ❤️ using Logistic Regression · Trained on Walmart Retail Data (2011–2014)<br>
    Threshold = Median Sales · Model Accuracy: 92.5%
</div>
""", unsafe_allow_html=True)
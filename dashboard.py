import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(
    page_title="DRS Synergy Discovery Engine",
    page_icon="🌲",
    layout="wide"
)

st.markdown("""
<style>
    .badge-synergy { color: #0f5132; background-color: #d1e7dd; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    .badge-additive { color: #664d03; background-color: #fff3cd; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
    .badge-antagonistic { color: #842029; background-color: #f8d7da; padding: 6px 14px; border-radius: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# 1. Pipeline Caching & Training
@st.cache_resource
def load_and_train_models():
    drugs_ref = ["5-FU", "CARBOPLATIN", "PACLITAXEL", "SUNITINIB", "DOXORUBICIN", "GEMCITABINE", "ETOPOSIDE", "CISPLATIN"]
    cells_ref = ["A2058", "MCF7", "T47D", "A549", "PC-3"]
    
    np.random.seed(42)
    records = []
    for _ in range(500):
        d1, d2 = np.random.choice(drugs_ref, 2, replace=False)
        c = np.random.choice(cells_ref)
        base = 14.5 if ("SUNITINIB" in (d1, d2) and ("PACLITAXEL" in (d1, d2) or "DOXORUBICIN" in (d1, d2))) else 4.5
        s = float(np.random.normal(base, 5.0))
        records.append({"Drug_A": d1, "Drug_B": d2, "Cell_Line": c, "Synergy": round(s, 2)})
    df = pd.DataFrame(records)

    n_genes = 50
    drs_features = pd.DataFrame(np.random.normal(0.1, 1.2, size=(len(drugs_ref), n_genes)), index=drugs_ref)
    cell_features = pd.DataFrame(np.random.normal(5.0, 1.0, size=(len(cells_ref), n_genes)), index=cells_ref)

    def extract_features(a, b, c_line):
        va = drs_features.loc[a].values
        vb = drs_features.loc[b].values
        vc = cell_features.loc[c_line].values
        return np.concatenate([va + vb, np.abs(va - vb), vc])

    X = np.array([extract_features(r['Drug_A'], r['Drug_B'], r['Cell_Line']) for _, r in df.iterrows()])
    y = df['Synergy'].values

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X, y)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42).fit(X, y)

    return drugs_ref, cells_ref, drs_features, cell_features, rf, gb, len(df)

drugs, cells, drs_df, cell_df, rf_model, gb_model, record_count = load_and_train_models()

def predict_score(drug_a, drug_b, cell_line, algo):
    va = drs_df.loc[drug_a].values
    vb = drs_df.loc[drug_b].values
    vc = cell_df.loc[cell_line].values
    feat = np.concatenate([va + vb, np.abs(va - vb), vc]).reshape(1, -1)
    
    score_rf = float(rf_model.predict(feat)[0])
    score_gb = float(gb_model.predict(feat)[0])

    if algo == "Random Forest":
        score = score_rf
    elif algo == "Gradient Boosted Trees (GBM)":
        score = score_gb
    else:
        score = (score_rf + score_gb) / 2.0
    return score, score_rf, score_gb

# Sidebar
st.sidebar.markdown("### 🌲 Model Selector")
chosen_algo = st.sidebar.selectbox("Active ML Architecture", ["Random Forest", "Gradient Boosted Trees (GBM)", "Ensemble (RF + GBM)"], index=0)
st.sidebar.markdown("---")
st.sidebar.metric("Indexed Screenings", f"{record_count:,}")
st.sidebar.metric("Compounds Active", len(drugs))
st.sidebar.metric("Cell Lines", len(cells))
st.sidebar.info("Features processed via symmetric DRS aggregation: $[d_A + d_B, |d_A - d_B|, c]$.")

# Main Interface
st.title("🌲 Drug Resistance Signature Screening Engine")
st.caption(f"Powered by **{chosen_algo}** regression on DRS vectors.")

tab1, tab2 = st.tabs(["🎯 Single Combination Predictor", "📊 High-Throughput Matrix Screening"])

with tab1:
    col1, col2, col3 = st.columns(3)
    d1 = col1.selectbox("Select Drug A", drugs, index=0)
    d2 = col2.selectbox("Select Drug B", [d for d in drugs if d != d1], index=0)
    c_line = col3.selectbox("Cancer Cell Line", cells, index=0)

    if st.button("Predict Synergy Score", type="primary", use_container_width=True):
        score, rf_s, gb_s = predict_score(d1, d2, c_line, chosen_algo)
        st.markdown("### Prediction Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted ZIP Score", f"{score:.2f}")
        with m2:
            st.markdown("**Interaction Verdict**")
            if score >= 10.0:
                st.markdown('<span class="badge-synergy">🔥 Synergistic</span>', unsafe_allow_html=True)
            elif score >= -10.0:
                st.markdown('<span class="badge-additive">⚖️ Additive</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-antagonistic">⛔ Antagonistic</span>', unsafe_allow_html=True)
        m3.metric("Random Forest", f"{rf_s:.2f}")
        m4.metric("Boosted Trees", f"{gb_s:.2f}")

        if score >= 10.0:
            st.success(f"**High Efficacy Candidate:** {d1} + {d2} in {c_line} exceeds Loewe additivity (ZIP score {score:.2f} ≥ 10.0).")
        else:
            st.warning(f"**Additive / Weak Candidate:** {d1} + {d2} in {c_line} demonstrates non-synergistic interaction (ZIP score {score:.2f} < 10.0).")

with tab2:
    st.markdown("#### High-Throughput Matrix Screen")
    st.write(f"Screening all pairs with **{chosen_algo}**.")
    screen_c = st.selectbox("Select Cell Line for Matrix Screen", cells, index=0)

    if st.button("Run Full Matrix Screen", type="primary", use_container_width=True):
        pairs = list(combinations(drugs, 2))
        res = []
        for p1, p2 in pairs:
            s, _, _ = predict_score(p1, p2, screen_c, chosen_algo)
            res.append({
                "Drug_A": p1,
                "Drug_B": p2,
                "Cell_Line": screen_c,
                "Predicted_Synergy": round(s, 2),
                "Synergistic": "Yes" if s >= 10.0 else "No"
            })
        df_screen = pd.DataFrame(res).sort_values(by="Predicted_Synergy", ascending=False)
        st.dataframe(df_screen, use_container_width=True)

        csv = df_screen.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Results (.csv)",
            data=csv,
            file_name=f"{chosen_algo.replace(' ', '_')}_{screen_c}.csv",
            mime="text/csv",
            use_container_width=True
        )
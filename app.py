import os
import re
import time
import pandas as pd
import streamlit as st

from pycaret.classification import load_model as load_cls, predict_model as predict_cls
from pycaret.clustering import load_model as load_clu, predict_model as predict_clu

# Optional GenAI prescriptions (app works without it)
try:
    from genai_prescriptions import generate_prescription
except Exception:
    generate_prescription = None

st.set_page_config(page_title="Cognitive SOAR: Prediction → Attribution", layout="wide")

# ---- Paths and schema (must match training) ----
CLS_PATH = "models/phishing_url_detector"
CLU_PATH = "models/threat_actor_profiler"
FEATURE_PLOT = "models/feature_importance.png"

FEATURES = [
    "having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
    "double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State",
    "URL_of_Anchor","Links_in_tags","SFH","Abnormal_URL","has_political_keyword",
]

@st.cache_resource
def load_assets():
    cls = load_cls(CLS_PATH) if os.path.exists(CLS_PATH + ".pkl") else None
    clu = load_clu(CLU_PATH) if os.path.exists(CLU_PATH + ".pkl") else None
    plot = FEATURE_PLOT if os.path.exists(FEATURE_PLOT) else None
    return cls, clu, plot

model_cls, model_clu, feature_plot = load_assets()

st.title("Cognitive SOAR — From Prediction to Attribution")

if not model_cls:
    st.error("Classifier not found. Try `make clean && make up` to retrain.")
    st.stop()

# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.header("URL Feature Input")
    url_length = st.select_slider("URL Length", options=["Short","Normal","Long"], value="Long")
    ssl_state  = st.select_slider("SSL Status", options=["Trusted","Suspicious","None"], value="Suspicious")
    sub_domain = st.select_slider("Sub-domain Count", options=["None","One","Many"], value="One")

    prefix_suffix = st.checkbox("Has Prefix/Suffix", True)
    has_ip        = st.checkbox("Uses IP Address", False)
    short_service = st.checkbox("Is Shortened", False)
    at_symbol     = st.checkbox("Has '@' Symbol", False)
    abnormal_url  = st.checkbox("Abnormal URL", True)
    pol_keyword   = st.checkbox("Political Keyword Present", False)

    provider = st.selectbox("Prescriptions Provider", ["Local fallback", "OpenAI", "Gemini"])
    submitted = st.button("Analyze", type="primary", use_container_width=True)

if not submitted:
    st.info("Set features in the sidebar and click **Analyze**.")
    st.stop()

# --------------- Map inputs → model features ---------------
input_row = {
    "having_IP_Address": 1 if has_ip else -1,
    "URL_Length": -1 if url_length == "Short" else (0 if url_length == "Normal" else 1),
    "Shortining_Service": 1 if short_service else -1,
    "having_At_Symbol": 1 if at_symbol else -1,
    "double_slash_redirecting": -1,          # neutral default (not exposed in UI)
    "Prefix_Suffix": 1 if prefix_suffix else -1,
    "having_Sub_Domain": -1 if sub_domain == "None" else (0 if sub_domain == "One" else 1),
    "SSLfinal_State": -1 if ssl_state == "None" else (0 if ssl_state == "Suspicious" else 1),
    "URL_of_Anchor": 0, "Links_in_tags": 0, "SFH": 0,  # neutral defaults
    "Abnormal_URL": 1 if abnormal_url else -1,
    "has_political_keyword": 1 if pol_keyword else 0,
}
input_df = pd.DataFrame([input_row], columns=FEATURES)

# ------------------- Runbook status -------------------
with st.status("Executing SOAR playbook...", expanded=True) as status:
    # Step 1: Classification
    st.write("Step 1: Predictive Analysis (Classification)")
    time.sleep(0.1)
    try:
        pred = predict_cls(model_cls, data=input_df)
    except Exception as e:
        st.error(f"Classification failed: {e}")
        st.stop()

    label_col = "prediction_label" if "prediction_label" in pred.columns else pred.columns[-1]
    label_raw = pred[label_col].iloc[0]
    try:
        label = int(label_raw)
    except Exception:
        # PyCaret shouldn't do this for classification, but be safe:
        label = 1 if str(label_raw).strip().lower() in {"1","true","malicious"} else -1

    score = float(pred["prediction_score"].iloc[0]) if "prediction_score" in pred.columns else None
    verdict = "MALICIOUS" if label == 1 else "BENIGN"
    st.write(f"Step 2: Verdict → **{verdict}**")

    # Step 2b: Clustering (Attribution) if malicious
    cluster_id = None
    cluster_name = None
    if label == 1 and model_clu is not None:
        st.write("Step 3: Threat Attribution (Clustering)")
        try:
            clu_out = predict_clu(model_clu, data=input_df)
            # Column can be "Cluster" (common) or sometimes "Label"
            col = "Cluster" if "Cluster" in clu_out.columns else ("Label" if "Label" in clu_out.columns else clu_out.columns[-1])
            raw = clu_out[col].iloc[0]
            # Robust parse: handle 2 or "Cluster 2"
            try:
                cluster_id = int(raw)
            except Exception:
                m = re.search(r"(\d+)", str(raw))
                cluster_id = int(m.group(1)) if m else 0
            mapping = {0: "Organized Cybercrime", 1: "State-Sponsored", 2: "Hacktivist"}
            cluster_name = mapping.get(cluster_id, f"Cluster {cluster_id}")
        except Exception as e:
            st.warning(f"Attribution unavailable: {e}")
            cluster_id, cluster_name = None, None

    status.update(label="Playbook Executed", state="complete")

# ------------------------ Tabs ------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Analysis Summary", "Visual Insights", "Prescriptive Plan", "Threat Attribution"]
)

with tab1:
    st.subheader("Verdict")
    st.metric("URL Verdict", verdict)
    if score is not None:
        st.caption(f"Model score: **{score:.3f}** (higher means more likely malicious)")
    st.subheader("Input Features")
    st.json(input_row)

with tab2:
    st.subheader("Feature Importance (Classifier)")
    if feature_plot and os.path.exists(feature_plot):
        st.image(feature_plot, caption="Feature importance saved during training")
    else:
        st.info("Feature importance plot not found. Re-run training to generate it.")

with tab3:
    st.subheader("Prescriptive Response Plan")
    if label != 1:
        st.info("Prescriptions appear only for MALICIOUS verdicts.")
    else:
        if generate_prescription is None:
            st.warning("GenAI module not found; showing local fallback plan.")
            plan = None
        else:
            prov = {"Local fallback": "local", "OpenAI": "OpenAI", "Gemini": "Gemini"}[provider]
            plan = generate_prescription(prov, input_row)

        if plan:
            st.write("**Summary:**", plan.get("summary", ""))
            st.write("**Risk Level:**", plan.get("risk_level", ""))
            st.write("**Recommended Actions:**")
            for a in plan.get("recommended_actions", []):
                st.write(f"• {a}")
            st.write("**Communication Draft:**")
            st.code(plan.get("communication_draft", ""), language="markdown")
        else:
            st.info("Using built-in fallback plan.")
            st.write("• Block URL • Quarantine emails • Hunt IOCs • Notify users • Open ticket")

with tab4:
    st.subheader("Threat Attribution")
    if label == 1 and cluster_name:
        st.metric("Predicted Actor", cluster_name)
        st.caption(f"(Cluster ID: {cluster_id})")
        st.markdown("""
**Profile Summaries**
- **State-Sponsored** — higher sophistication, valid SSL, subtle deception (prefix/suffix).
- **Organized Cybercrime** — high-volume/noisy: shorteners, IP usage, abnormal structures.
- **Hacktivist** — opportunistic mix; often signals with political terms.
        """)
    elif label == 1 and model_clu is None:
        st.warning("Clustering model not found; attribution disabled. Retrain to enable.")
    else:
        st.info("Attribution runs only for MALICIOUS verdicts.")


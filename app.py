import os
import time
from typing import Dict, Tuple, Optional

import pandas as pd
import streamlit as st
from pycaret.classification import load_model as load_cls
from pycaret.classification import predict_model as predict_cls
from pycaret.clustering import load_model as load_clu
from pycaret.clustering import predict_model as predict_clu

try:
    from genai_prescriptions import generate_prescription
except Exception:  # module is optional
    generate_prescription = None  # type: ignore


st.set_page_config(page_title="Cognitive SOAR", layout="wide")


@st.cache_resource
def load_models() -> Tuple[object, Optional[object]]:
    cls_path = "models/phishing_url_detector.pkl"
    clu_path = "models/threat_actor_profiler.pkl"

    model_cls = load_cls(cls_path)
    try:
        model_clu = load_clu(clu_path)
    except Exception:
        model_clu = None
    return model_cls, model_clu


def sidebar_inputs() -> Dict[str, int]:
    st.sidebar.header("URL Features")

    def tri(label: str) -> int:
        val = st.sidebar.selectbox(label, ["-1", "0", "1"], key=label)
        return int(val)

    def biny(label: str) -> int:
        return 1 if st.sidebar.checkbox(label, value=False, key=label) else -1

    def cat(label: str, options: Dict[str, int]) -> int:
        choice = st.sidebar.selectbox(label, list(options.keys()), key=label)
        return options[choice]

    url_len = cat("URL Length", {"Short": -1, "Normal": 0, "Long": 1})
    ssl = cat("SSL Status", {"None": -1, "Suspicious": 0, "Trusted": 1})
    subd = cat("Sub-domain", {"None": -1, "One": 0, "Many": 1})

    feats = {
        "having_IP_Address": biny("Uses IP Address"),
        "URL_Length": url_len,
        "Shortining_Service": biny("Is Shortened"),
        "having_At_Symbol": biny("Has '@' Symbol"),
        "double_slash_redirecting": biny("Double Slash Redirecting"),
        "Prefix_Suffix": biny("Has Prefix/Suffix (-)"),
        "having_Sub_Domain": subd,
        "SSLfinal_State": ssl,
        "URL_of_Anchor": tri("URL of Anchor (-1/0/1)"),
        "Links_in_tags": tri("Links in Tags (-1/0/1)"),
        "SFH": tri("SFH (-1/0/1)"),
        "Abnormal_URL": biny("Abnormal URL"),
        "has_political_keyword": biny("Political Keyword Present"),
    }
    return feats


def get_cluster_id(df: pd.DataFrame) -> int:
    # PyCaret clustering sometimes returns "Cluster 2" as a string column.
    col = next((c for c in df.columns if c.lower().startswith("cluster")), "Cluster")
    raw = df[col].iloc[0]
    if isinstance(raw, str):
        parts = [p for p in raw.split() if p.isdigit()]
        return int(parts[-1]) if parts else 0
    return int(raw)


def actor_mapping(cid: int) -> str:
    mapping = {0: "Organized Cybercrime", 1: "State-Sponsored", 2: "Hacktivist"}
    return mapping.get(cid, f"Cluster {cid}")


def actor_blurb(name: str) -> str:
    if name == "Organized Cybercrime":
        return (
            "High-volume, noisy campaigns. Frequent shorteners, IP-in-URL, and abnormal structures."
        )
    if name == "State-Sponsored":
        return "Subtle, well-crafted operations. Valid SSL with deceptive structure."
    if name == "Hacktivist":
        return "Opportunistic activity with political signaling and mixed tactics."
    return "Unspecified profile."


def render_visual_insights():
    path = "models/feature_importance.png"
    if os.path.exists(path):
        st.image(path, caption="Feature importance from training", use_container_width=True)
    else:
        st.info("Feature importance plot not found. Re-run training to regenerate.")


def render_prescription(provider: str, payload: Dict[str, int]):
    if generate_prescription is None or provider == "Local fallback":
        st.warning("GenAI module not found or provider set to fallback; showing local plan.")
        provider = "local"

    plan = (
        generate_prescription(provider, payload)
        if generate_prescription is not None
        else {
            "summary": "Local fallback plan.",
            "risk_level": "Medium",
            "recommended_actions": [
                "Block URL at gateway.",
                "Hunt for IOCs in SIEM for last 7 days.",
                "Notify potentially impacted users.",
            ],
            "communication_draft": (
                "A suspicious URL was detected and blocked while we investigate."
            ),
        }
    )

    st.subheader("Prescriptive plan")
    st.write(f"**Risk:** {plan.get('risk_level','Unknown')}")
    st.write(plan.get("summary", ""))
    st.write("**Recommended actions:**")
    for item in plan.get("recommended_actions", []):
        st.write(f"- {item}")
    st.write("**Communication draft:**")
    st.code(plan.get("communication_draft", ""), language="markdown")


def main():
    st.title("Cognitive SOAR — From Prediction to Attribution")
    model_cls, model_clu = load_models()

    with st.sidebar:
        provider = st.selectbox(
            "Prescriptions Provider", ["Local fallback", "OpenAI", "Gemini"]
        )
        st.divider()
        features = sidebar_inputs()
        analyze = st.button("Analyze", type="primary")

    tabs = st.tabs(
        ["Analysis Summary", "Visual Insights", "Prescriptive Plan", "Threat Attribution"]
    )
    t_summary, t_visual, t_plan, t_attr = tabs

    if analyze:
        input_df = pd.DataFrame([features])

        with st.status("Executing SOAR playbook...", expanded=True) as status:
            st.write("Step 1: Predictive Analysis (Classification)")
            time.sleep(0.1)
            pred = predict_cls(model_cls, data=input_df)
            label = int(pred["prediction_label"].iloc[0])
            score = float(pred.get("prediction_score", pd.Series([0.0])).iloc[0])

            st.write("Step 2: Verdict")
            verdict = "MALICIOUS" if label == 1 else "BENIGN"
            st.write(f"Verdict → **{verdict}**")
            time.sleep(0.1)

            cluster_name = None
            cluster_id = None
            if label == 1 and model_clu is not None:
                st.write("Step 3: Threat Attribution (Clustering)")
                clu_out = predict_clu(model_clu, data=input_df)
                cluster_id = get_cluster_id(clu_out)
                cluster_name = actor_mapping(cluster_id)

            status.update(label="Playbook Executed", state="complete")

        with t_summary:
            st.subheader("Verdict")
            st.metric("URL Verdict", verdict, delta=None)
            st.write(f"Model score: {score:.3f} (higher means more likely malicious)")

            st.subheader("Input Features")
            st.json(features, expanded=False)

        with t_visual:
            render_visual_insights()

        with t_plan:
            render_prescription(provider, features)

        with t_attr:
            if verdict == "MALICIOUS" and cluster_name is not None:
                st.subheader("Predicted Threat Actor")
                st.write(f"**{cluster_name}** (Cluster {cluster_id})")
                st.caption(actor_blurb(cluster_name))
            else:
                st.info("Attribution is only shown for MALICIOUS verdicts.")
    else:
        st.info("Set the features on the left and click **Analyze**.")


if __name__ == "__main__":
    main()

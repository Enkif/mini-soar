import os
import numpy as np
import pandas as pd

from pycaret.classification import (
    setup as cls_setup,
    compare_models,
    finalize_model as cls_finalize,
    save_model as cls_save,
    plot_model,
)
from pycaret.clustering import (
    setup as clu_setup,
    create_model as clu_create,
    save_model as clu_save,
    predict_model as clu_predict,  # used at inference
)

FEATURES = [
    "having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
    "double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State",
    "URL_of_Anchor","Links_in_tags","SFH","Abnormal_URL","has_political_keyword",
]

def generate_synthetic_data(num_samples=900, seed=42):
    rng = np.random.default_rng(seed)
    n_each = num_samples // 3

    # State-Sponsored
    state = pd.DataFrame({
        "having_IP_Address": rng.choice([1,-1], n_each, p=[0.2,0.8]),
        "URL_Length":        rng.choice([1,0,-1], n_each, p=[0.4,0.5,0.1]),
        "Shortining_Service":rng.choice([1,-1], n_each, p=[0.1,0.9]),
        "having_At_Symbol":  rng.choice([1,-1], n_each, p=[0.2,0.8]),
        "double_slash_redirecting": rng.choice([1,-1], n_each, p=[0.2,0.8]),
        "Prefix_Suffix":     rng.choice([1,-1], n_each, p=[0.8,0.2]),
        "having_Sub_Domain": rng.choice([1,0,-1], n_each, p=[0.5,0.4,0.1]),
        "SSLfinal_State":    rng.choice([1,0,-1], n_each, p=[0.8,0.15,0.05]),
        "URL_of_Anchor":     rng.choice([1,0,-1], n_each, p=[0.3,0.6,0.1]),
        "Links_in_tags":     rng.choice([1,0,-1], n_each, p=[0.3,0.6,0.1]),
        "SFH":               rng.choice([1,0,-1], n_each, p=[0.3,0.6,0.1]),
        "Abnormal_URL":      rng.choice([1,-1], n_each, p=[0.3,0.7]),
        "has_political_keyword": np.zeros(n_each, dtype=int),
    })
    state["_profile"] = "State-Sponsored"

    # Organized Cybercrime
    crime = pd.DataFrame({
        "having_IP_Address": rng.choice([1,-1], n_each, p=[0.8,0.2]),
        "URL_Length":        rng.choice([1,0,-1], n_each, p=[0.6,0.3,0.1]),
        "Shortining_Service":rng.choice([1,-1], n_each, p=[0.8,0.2]),
        "having_At_Symbol":  rng.choice([1,-1], n_each, p=[0.6,0.4]),
        "double_slash_redirecting": rng.choice([1,-1], n_each, p=[0.6,0.4]),
        "Prefix_Suffix":     rng.choice([1,-1], n_each, p=[0.6,0.4]),
        "having_Sub_Domain": rng.choice([1,0,-1], n_each, p=[0.6,0.3,0.1]),
        "SSLfinal_State":    rng.choice([1,0,-1], n_each, p=[0.1,0.2,0.7]),
        "URL_of_Anchor":     rng.choice([1,0,-1], n_each, p=[0.6,0.3,0.1]),
        "Links_in_tags":     rng.choice([1,0,-1], n_each, p=[0.6,0.3,0.1]),
        "SFH":               rng.choice([1,0,-1], n_each, p=[0.6,0.3,0.1]),
        "Abnormal_URL":      rng.choice([1,-1], n_each, p=[0.8,0.2]),
        "has_political_keyword": np.zeros(n_each, dtype=int),
    })
    crime["_profile"] = "Organized Cybercrime"

    # Hacktivist
    hactiv = pd.DataFrame({
        "having_IP_Address": rng.choice([1,-1], n_each, p=[0.4,0.6]),
        "URL_Length":        rng.choice([1,0,-1], n_each, p=[0.5,0.3,0.2]),
        "Shortining_Service":rng.choice([1,-1], n_each, p=[0.3,0.7]),
        "having_At_Symbol":  rng.choice([1,-1], n_each, p=[0.5,0.5]),
        "double_slash_redirecting": rng.choice([1,-1], n_each, p=[0.4,0.6]),
        "Prefix_Suffix":     rng.choice([1,-1], n_each, p=[0.5,0.5]),
        "having_Sub_Domain": rng.choice([1,0,-1], n_each, p=[0.4,0.4,0.2]),
        "SSLfinal_State":    rng.choice([1,0,-1], n_each, p=[0.3,0.4,0.3]),
        "URL_of_Anchor":     rng.choice([1,0,-1], n_each, p=[0.4,0.4,0.2]),
        "Links_in_tags":     rng.choice([1,0,-1], n_each, p=[0.4,0.4,0.2]),
        "SFH":               rng.choice([1,0,-1], n_each, p=[0.4,0.4,0.2]),
        "Abnormal_URL":      rng.choice([1,-1], n_each, p=[0.5,0.5]),
        "has_political_keyword": rng.choice([0,1], n_each, p=[0.2,0.8]),
    })
    hactiv["_profile"] = "Hacktivist"

    malicious = pd.concat([state, crime, hactiv], ignore_index=True)
    malicious["label"] = 1

    # Benign
    n_benign = num_samples
    benign = pd.DataFrame({
        "having_IP_Address": rng.choice([1,-1], n_benign, p=[0.05,0.95]),
        "URL_Length":        rng.choice([1,0,-1], n_benign, p=[0.2,0.6,0.2]),
        "Shortining_Service":rng.choice([1,-1], n_benign, p=[0.05,0.95]),
        "having_At_Symbol":  rng.choice([1,-1], n_benign, p=[0.1,0.9]),
        "double_slash_redirecting": rng.choice([1,-1], n_benign, p=[0.1,0.9]),
        "Prefix_Suffix":     rng.choice([1,-1], n_benign, p=[0.1,0.9]),
        "having_Sub_Domain": rng.choice([1,0,-1], n_benign, p=[0.2,0.6,0.2]),
        "SSLfinal_State":    rng.choice([1,0,-1], n_benign, p=[0.9,0.09,0.01]),
        "URL_of_Anchor":     rng.choice([1,0,-1], n_benign, p=[0.2,0.6,0.2]),
        "Links_in_tags":     rng.choice([1,0,-1], n_benign, p=[0.2,0.6,0.2]),
        "SFH":               rng.choice([1,0,-1], n_benign, p=[0.2,0.6,0.2]),
        "Abnormal_URL":      rng.choice([1,-1], n_benign, p=[0.05,0.95]),
        "has_political_keyword": np.zeros(n_benign, dtype=int),
        "_profile": "Benign",
    })
    benign["label"] = -1

    df = pd.concat([malicious, benign], ignore_index=True)
    return df[FEATURES + ["label", "_profile"]]

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model_path_cls = "models/phishing_url_detector"
    model_path_clu = "models/threat_actor_profiler"
    plot_path = "models/feature_importance.png"

    # skip retrain if both exist
    if os.path.exists(model_path_cls + ".pkl") and os.path.exists(model_path_clu + ".pkl"):
        print("Models already exist. Skipping training.")
        return

    print("Generating data…")
    data = generate_synthetic_data()
    data.to_csv("data/phishing_synthetic.csv", index=False)

    # ---- Classification ----
    print("Classification setup…")
    # IMPORTANT: ignore '_profile' so prediction input doesn't need it
    cls_setup(data, target="label", ignore_features=['_profile'], session_id=42, verbose=False)
    best = compare_models(n_select=1)
    final_cls = cls_finalize(best)

    print("Saving classifier & feature importance…")
    plot_model(final_cls, plot="feature", save=True)
    for candidate in ["Feature Importance.png", "Feature Importance.PNG"]:
        if os.path.exists(candidate):
            os.replace(candidate, plot_path)
            break
    cls_save(final_cls, model_path_cls)

    # ---- Clustering (KMeans, k=3) ----
    print("Clustering setup…")
    features_only = data.drop(columns=["label", "_profile"])
    clu_setup(features_only, session_id=42, verbose=False)
    model = clu_create("kmeans", num_clusters=3)
    clu_save(model, model_path_clu)
    print("Saved clustering model.")

if __name__ == "__main__":
    train()

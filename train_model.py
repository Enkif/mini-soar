import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from pycaret.classification import (
    compare_models,
    finalize_model,
    load_model,
    predict_model,
    save_model,
    setup,
    plot_model,
)
from pycaret.clustering import create_model as clu_create
from pycaret.clustering import save_model as clu_save
from pycaret.clustering import setup as clu_setup


RNG = np.random.default_rng(42)


def _tri(p_neg: float, p_zero: float, p_pos: float) -> int:
    return RNG.choice([-1, 0, 1], p=[p_neg, p_zero, p_pos]).item()  # type: ignore[call-overload]


def synth_profile_state(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "having_IP_Address": RNG.choice([-1, 1], size=n, p=[0.95, 0.05]),
            "URL_Length": RNG.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2]),
            "Shortining_Service": RNG.choice([-1, 1], size=n, p=[0.95, 0.05]),
            "having_At_Symbol": RNG.choice([-1, 1], size=n, p=[0.9, 0.1]),
            "double_slash_redirecting": RNG.choice([-1, 1], size=n, p=[0.9, 0.1]),
            "Prefix_Suffix": RNG.choice([-1, 1], size=n, p=[0.2, 0.8]),
            "having_Sub_Domain": RNG.choice([-1, 0, 1], size=n, p=[0.4, 0.4, 0.2]),
            "SSLfinal_State": RNG.choice([-1, 0, 1], size=n, p=[0.05, 0.05, 0.9]),
            "URL_of_Anchor": [ _tri(0.6, 0.2, 0.2) for _ in range(n) ],
            "Links_in_tags": [ _tri(0.6, 0.2, 0.2) for _ in range(n) ],
            "SFH": [ _tri(0.6, 0.2, 0.2) for _ in range(n) ],
            "Abnormal_URL": RNG.choice([-1, 1], size=n, p=[0.8, 0.2]),
            "has_political_keyword": RNG.choice([-1, 1], size=n, p=[0.95, 0.05]),
            "_profile": ["state"] * n,
            "label": [1] * n,
        }
    )


def synth_profile_crime(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "having_IP_Address": RNG.choice([-1, 1], size=n, p=[0.2, 0.8]),
            "URL_Length": RNG.choice([-1, 0, 1], size=n, p=[0.1, 0.3, 0.6]),
            "Shortining_Service": RNG.choice([-1, 1], size=n, p=[0.2, 0.8]),
            "having_At_Symbol": RNG.choice([-1, 1], size=n, p=[0.4, 0.6]),
            "double_slash_redirecting": RNG.choice([-1, 1], size=n, p=[0.4, 0.6]),
            "Prefix_Suffix": RNG.choice([-1, 1], size=n, p=[0.3, 0.7]),
            "having_Sub_Domain": RNG.choice([-1, 0, 1], size=n, p=[0.2, 0.3, 0.5]),
            "SSLfinal_State": RNG.choice([-1, 0, 1], size=n, p=[0.6, 0.2, 0.2]),
            "URL_of_Anchor": [ _tri(0.2, 0.2, 0.6) for _ in range(n) ],
            "Links_in_tags": [ _tri(0.2, 0.2, 0.6) for _ in range(n) ],
            "SFH": [ _tri(0.2, 0.2, 0.6) for _ in range(n) ],
            "Abnormal_URL": RNG.choice([-1, 1], size=n, p=[0.2, 0.8]),
            "has_political_keyword": RNG.choice([-1, 1], size=n, p=[0.9, 0.1]),
            "_profile": ["crime"] * n,
            "label": [1] * n,
        }
    )


def synth_profile_hacktivist(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "having_IP_Address": RNG.choice([-1, 1], size=n, p=[0.6, 0.4]),
            "URL_Length": RNG.choice([-1, 0, 1], size=n, p=[0.4, 0.4, 0.2]),
            "Shortining_Service": RNG.choice([-1, 1], size=n, p=[0.6, 0.4]),
            "having_At_Symbol": RNG.choice([-1, 1], size=n, p=[0.7, 0.3]),
            "double_slash_redirecting": RNG.choice([-1, 1], size=n, p=[0.7, 0.3]),
            "Prefix_Suffix": RNG.choice([-1, 1], size=n, p=[0.6, 0.4]),
            "having_Sub_Domain": RNG.choice([-1, 0, 1], size=n, p=[0.5, 0.4, 0.1]),
            "SSLfinal_State": RNG.choice([-1, 0, 1], size=n, p=[0.5, 0.3, 0.2]),
            "URL_of_Anchor": [ _tri(0.4, 0.2, 0.4) for _ in range(n) ],
            "Links_in_tags": [ _tri(0.4, 0.2, 0.4) for _ in range(n) ],
            "SFH": [ _tri(0.4, 0.2, 0.4) for _ in range(n) ],
            "Abnormal_URL": RNG.choice([-1, 1], size=n, p=[0.5, 0.5]),
            "has_political_keyword": RNG.choice([-1, 1], size=n, p=[0.2, 0.8]),
            "_profile": ["hacktivist"] * n,
            "label": [1] * n,
        }
    )


def synth_benign(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "having_IP_Address": RNG.choice([-1, 1], size=n, p=[0.98, 0.02]),
            "URL_Length": RNG.choice([-1, 0, 1], size=n, p=[0.4, 0.5, 0.1]),
            "Shortining_Service": RNG.choice([-1, 1], size=n, p=[0.98, 0.02]),
            "having_At_Symbol": RNG.choice([-1, 1], size=n, p=[0.98, 0.02]),
            "double_slash_redirecting": RNG.choice([-1, 1], size=n, p=[0.98, 0.02]),
            "Prefix_Suffix": RNG.choice([-1, 1], size=n, p=[0.95, 0.05]),
            "having_Sub_Domain": RNG.choice([-1, 0, 1], size=n, p=[0.2, 0.7, 0.1]),
            "SSLfinal_State": RNG.choice([-1, 0, 1], size=n, p=[0.05, 0.05, 0.9]),
            "URL_of_Anchor": [ _tri(0.7, 0.2, 0.1) for _ in range(n) ],
            "Links_in_tags": [ _tri(0.7, 0.2, 0.1) for _ in range(n) ],
            "SFH": [ _tri(0.7, 0.2, 0.1) for _ in range(n) ],
            "Abnormal_URL": RNG.choice([-1, 1], size=n, p=[0.97, 0.03]),
            "has_political_keyword": RNG.choice([-1, 1], size=n, p=[0.99, 0.01]),
            "_profile": ["benign"] * n,
            "label": [0] * n,
        }
    )


def generate_data(n_per_class: int = 300) -> pd.DataFrame:
    parts = [
        synth_profile_state(n_per_class),
        synth_profile_crime(n_per_class),
        synth_profile_hacktivist(n_per_class),
        synth_benign(n_per_class * 3),
    ]
    df = pd.concat(parts, ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def train_models() -> Tuple[str, str]:
    ensure_dirs()
    print("Generating data…")
    df = generate_data()
    df.to_csv("data/phishing_synthetic.csv", index=False)

    print("Classification setup…")
    _ = setup(
        data=df,
        target="label",
        ignore_features=["_profile"],
        verbose=False,
        session_id=42,
    )
    best = compare_models(n_select=1, include=["rf", "et", "gbc"])
    final = finalize_model(best)
    save_model(final, "models/phishing_url_detector")

    # Optional: save feature importance image
    try:
        plot_model(final, plot="feature", save=True)
        # PyCaret saves as 'Feature Importance.png' in cwd
        src = Path("Feature Importance.png")
        if src.exists():
            src.rename("models/feature_importance.png")
    except Exception:
        pass

    print("Clustering setup…")
    features_only = df.drop(columns=["label", "_profile"])
    _ = clu_setup(data=features_only, verbose=False, session_id=42)
    kmeans = clu_create("kmeans", num_clusters=3)
    clu_save(kmeans, "models/threat_actor_profiler")

    return "models/phishing_url_detector.pkl", "models/threat_actor_profiler.pkl"


if __name__ == "__main__":
    if not (Path("models/phishing_url_detector.pkl").exists() and
            Path("models/threat_actor_profiler.pkl").exists()):
        train_models()
    else:
        print("Models already exist. Skipping training.")

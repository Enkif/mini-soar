#Cognitive SOAR — From Prediction to Attribution#

Mini‑SOAR upgraded: binary phishing detection → enriched triage with threat attribution using unsupervised learning. The app classifies URL feature vectors as BENIGN/MALICIOUS and, when malicious, maps to a likely threat‑actor profile (State‑Sponsored, Organized Cybercrime, Hacktivist) via K‑Means clustering (k=3).

Why this matters

SOCs don’t just need “block/allow.” They need context—the who/why/how—to route incidents faster, pick the right playbooks, and communicate clearly.

Features

Classification: PyCaret auto‑ML pipeline trains a phishing detector from synthetic data.

Attribution: Separate PyCaret clustering workflow (K‑Means, k=3) on features‑only dataset.

Streamlit UI: Beginner‑friendly sidebar inputs, status steps, tabs for insights, prescriptions, and attribution.

GenAI Prescriptions (optional): Plug in OpenAI or Gemini keys to generate a structured response plan (falls back locally if keys are absent).

Dockerized: One command to build, train, and run via Docker Compose.

CI Linting: GitHub Actions workflow for flake8.

Architecture (Dual‑Model)

[Streamlit app]
   ├─ Classifier: models/phishing_url_detector.pkl  (PyCaret classification)
   └─ Clusterer: models/threat_actor_profiler.pkl   (PyCaret clustering – KMeans, k=3)

Flow: Input features → Classify → if MALICIOUS → Cluster → Map(Cluster→Actor Profile)

Synthetic profiles

State‑Sponsored: high sophistication (valid SSL), subtle deception (prefix/suffix)

Organized Cybercrime: noisy, shorteners + IPs + abnormal URLs

Hacktivist: opportunistic mix, political‑keyword signal



# Build, train, run
make up
   
Open: http://localhost:8501



#Configuration

Secrets (recommended)

Created .streamlit/secrets.toml

OPENAI_API_KEY = "..."
GEMINI_API_KEY = "..."

Default UI port is 8501. 

#Repository Layout

mini-soar/
├─ app.py                    # Streamlit UI (predict → attribute → prescribe)
├─ train_model.py            # Data synth + classification + clustering training
├─ genai_prescriptions.py    # Optional GenAI plan generator (safe fallbacks)
├─ Dockerfile                # Python 3.11 slim + deps
├─ docker-compose.yml        # Train then launch Streamlit
├─ Makefile                  # up/down/clean/logs shortcuts
├─ requirements.txt
├─ models/                   # Saved models (created at runtime)
├─ data/                     # Synthetic CSV (created at runtime)
├─ .github/workflows/lint.yml# CI linting
├─ README.md
├─ INSTALL.md
└─ TESTING.md




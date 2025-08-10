#Goal

Turn the mini‑SOAR baseline into a dual‑model app (prediction + attribution) and run it locally in WSL with Docker Desktop.
#Configure Secrets
.streamlit/secrets.toml
OPENAI_API_KEY = "..."
GEMINI_API_KEY = "..."

#Build, Train, Run
make up
make logs
Open http://localhost:8501.
make clean && make up

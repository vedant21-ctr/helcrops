---
title: 🌿 HaveCrops Analytics
emoji: 🚜
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Project: Crop Yield Prediction & Agentic Farming Advisor

## From Predictive Analytics to Intelligent Intervention

This repository implements a **crop-driven agricultural analytics system** that predicts yield (Milestone 1) and extends into an **agentic farming advisor** with LangGraph, RAG, and an LLM (Milestone 2).

- **Milestone 1:** Classical machine learning on historical agricultural, soil, and weather data to predict yield and identify drivers of productivity.
- **Milestone 2:** An agent-based flow that reasons about farm conditions, retrieves agronomic best practices (RAG), and produces structured recommendations and PDF reports.

---

### Constraints & Requirements

- **Team Size:** 3–4 Students  
- **API Budget:** Free tier (open-source models / free APIs)  
- **Framework:** LangGraph  
- **Hosting:** Hugging Face Spaces, Streamlit Cloud, or Render  

---

### Technical Stack

| Layer | Technology |
| :--- | :--- |
| **Prediction Engine** | Scikit-Learn (Linear Regression, Decision Tree CART) |
| **Statistical Analysis** | Pandas, NumPy, SciPy |
| **Visualizations** | Seaborn, Matplotlib, Plotly |
| **Intelligence Layer** | LangGraph, FAISS + sentence-transformers (RAG), Groq (Llama 3.3) |
| **Interface** | Streamlit |

---

### Running the App

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Set a **Groq API key** (used by the AI advisory pipeline):
   ```bash
   export GROQ_API_KEY="your-key"
   ```
   Or use a `.env` file with `GROQ_API_KEY=...` (see `python-dotenv` usage in `app.py`).
3. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

Open the **AI Advisory** tab after entering the dashboard; the agent requires `GROQ_API_KEY` in the environment.

---

### Milestones & Deliverables

**Milestone 1 (mid-sem):** ML yield prediction, UI, model metrics (MAE, RMSE, R², etc.).

**Milestone 2 (end-sem):** Deployed app, LangGraph workflow (see `agent/graph.py`, `agent/nodes.py`), RAG over `rag/documents.json`, structured reports and optional PDF export (`utils/pdf_generator.py`).

---

### Evaluation (reference)

| Phase | Focus |
| :--- | :--- |
| **Mid-Sem** | Interface, feature engineering, regression accuracy, code quality |
| **End-Sem** | Agent reasoning, vector retrieval, deployment, UX |

---

> **Note:** Localhost-only demos may not meet course hosting requirements; deploy to HF Spaces, Streamlit Cloud, or Render.

**Academic context:** Advanced Machine Learning in Agriculture — whole-team project.

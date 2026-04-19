---
title: 🌿 HaveCrops Analytics
emoji: 🚜
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Project: Crop Yield Prediction & Agentic Farming Advisor
## From Predictive Analytics to Intelligent## 📖 Milestone 1 & 2 Combined Architecture
This repository implements both a classical Machine Learning yield predictor (Milestone 1) and a fully Autonomous Agentic AI system (Milestone 2) using LangGraph and HuggingFace LLMs.s the design and implementation of a **Crop-driven agricultural analytics system** that predicts crop yield and evolves into an agentic farming advisor.

- **Milestone 1:** Classical machine learning techniques applied to historical agricultural, soil, and weather data to predict yield and identify key drivers of productivity.
- **Milestone 2:** Extension into an agent-based application that autonomously reasons about farm conditions, retrieves agronomic best practices (RAG), and plans intervention strategies.

---

### Constraints & Requirements
- **Team Size:** 3-4 Students
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** LangGraph (Recommended)
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Cloud, or Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **ML Models (M1)** | Linear Regression, Decision Trees, Scikit-Learn |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit or Gradio |
| **LLMs (M2)** | Open-source models or Free-tier APIs |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Yield Prediction (Mid-Sem)
**Objective:** Identify yield potential using historical farm data focus on classical ML pipelines *without LLMs*.

**Key Deliverables:**
- Problem understanding & Business context.
- System architecture diagram.
- Working local application with UI (Streamlit/Gradio).
- Model performance evaluation report (MAE, RMSE, R2, etc.).

#### Milestone 2: Agentic Farm Advisor (End-Sem)
**Objective:**### 🧠 Milestone 2: Agentic Farm Advisory (New)
The new `🤖 AI Advisor` tab extends the application into an Autonomous Agent utilizing:
1. **LangGraph Pipeline**: Orchestrates workflows between inputs, models, and reasoning nodes.
2. **FAISS RAG System**: Embeds domain knowledge using sentence-transformers, preventing the LLM from relying on unverified internet data.
3. **Zephyr-7b-beta LLM**: A HuggingFace real-time text-generation engine that synthesizes the predictions, queries, and RAG contexts into actionable strategies.
4. **Anti-Hallucination Prompts**: Built with explicit LangChain `PromptTemplate`s heavily constraining the AI to *only rely upon the provided RAG documents*, returning "Insufficient data" appropriately.
5. **PDF Reporter**: Condenses the AI Advisory output into a dynamically sized PDF using `reportlab`.

## 🚀 Running the App
1. Load virtual environment:
   `source .venv/bin/activate`
2. Run installation (includes Milestone 1 & 2 dependencies):
   `pip install -r requirements.txt`
3. Launch Streamlit:
   `streamlit run app.py`

*Note for AI Advisory Feature:* 
You must provide a free-tier **HuggingFace Access Token** in the dashboard UI when prompting to generate the report.gentic strategist that reasons about yield conditions and retrieves best practices to generate structured recommendations.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Agent workflow documentation (States & Nodes).
- Structured recommendation report generation.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering, UI Usability, Evaluation Metrics. |
| **End-Sem** | 30% | Reasoning quality, RAG & State management implementation, Output clarity, Deployment success. |

> [!WARNING]
> Localhost-only demonstrations will **not** be accepted for final submission. Project must be hosted.

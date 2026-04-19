---
title: 🌿 HaveCrops Analytics
emoji: 🚜
colorFrom: green
colorTo: blue
sdk: streamlit
pinned: false
---

# 🌾 HaveCrops Analytics: AI Advisory System 

![Premium UI Dashboard](https://img.shields.io/badge/UI-Premium_Dark_Mode-0B0F19?style=for-the-badge&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-3B82F6?style=for-the-badge&logoColor=white)
![Deployed](https://img.shields.io/badge/Status-Deployed_Online-22C55E?style=for-the-badge&logo=streamlit)

**An End-Semester Capstone Project: From Predictive Analytics to Intelligent Intervention**

HaveCrops Analytics is an advanced, production-grade agricultural engine. It transitions the agricultural ecosystem from traditional machine learning yield prediction (Milestone 1) into a fully automated, **Agentic Farming Advisor** operating on LangGraph, RAG (Retrieval-Augmented Generation), and LLM pipelines (Milestone 2).

---

## 🏗️ System Architecture & Milestones

This repository completes the full lifecycle of the agronomic advisor architecture:

### Milestone 1: Predictive Analytics
* **Classical Machine Learning:** Predicts exact crop yields based on soil, rainfall, and fertilizer geometries using Scikit-Learn pipelines (Linear Regression, Decision Tree CART).
* **Continuous Visual Dashboard:** Plots interactive multi-variable analytics connecting predictive outcomes natively with historical distributions via Plotly.
* **Geochemical Risk Metrics:** Flags real-time thresholds tracking yield probabilities mapped through exact data logic.

### Milestone 2: LLM Agentic Advisory (Current)
* **LangGraph Node Routing:** Reasons dynamically regarding farm environments.
* **RAG (Retrieval-Augmented Generation):** Synthesizes vector embeddings of agronomic best-practices directly into the local context.
* **Formatted Generative Reports:** Securely exports programmatic step-by-step PDF solutions outlining explicitly modeled action plans dynamically bound to the user's soil health predictions!

---

## 🎨 Premium Dark-Mode User Interface
The entire frontend has been reconstructed into a **SaaS-Level Dark Dashboard**, natively configuring:
- `Overview Matrix:` 4-Node card grids explicitly guiding user interaction workflows.
- `Visual Analytics Engine:` Interactive Plotly distributions dynamically updating on continuous threshold sliders.
- `System Architecture Diagrams:` Native HTML visualizations of the complex AI inference pipeline embedded securely inside the application.
- `Dynamic Risk Scaling:` Gauges inversely bound to mathematical dataset metrics.

---

## ⚙️ Technical Stack

| Layer | Technology |
| :--- | :--- |
| **Prediction Engine** | Scikit-Learn |
| **Statistical Analysis** | Pandas, NumPy, SciPy |
| **Interactive Visualizations** | Plotly (px, go) |
| **Intelligence Layer** | LangGraph, FAISS + `sentence-transformers` (RAG), Groq (Llama 3.3) |
| **Interface & DOM** | Streamlit (Custom Theme Architecture) |

---

## 🚀 Running the App Locally

To clone the deployment natively and test the code workflows:

1. **Create a virtual environment and assemble dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Establish the Groq Network Key:**
   To securely generate the AI agentic protocols, create a `.env` file right inside the root folder, and paste your LLM key:
   ```env
   GROQ_API_KEY="your-groq-key-here"
   ```
   *Note: This file is fully protected by `.gitignore` and securely prevents your private key from leaking online.*

3. **Launch the Engine:**
   ```bash
   streamlit run app.py
   ```

*(For live cloud deployments like Vercel or Streamlit Cloud, inject your API keys directly into the web "Environment Variables" / "Secrets" dashboard instead of utilizing `.env`!)*

---

## 🎓 Academic Deliverables & Evaluation

This platform aligns fully as an End-Semester submission for **Advanced Machine Learning in Agriculture**. 

* **Track**: BTech CSE | AI & ML  
* **Scale**: End-to-End ML Pipeline + LLM Agent Workflow Implementation.  

### 👥 Team Members
| Name | PRN | Role |
| :--- | :--- | :--- |
| **Vedant Satbhai** | `2401010500` | Team Leader |
| **Pratik Agade** | `2401010346` | Report Maker & UI Checker |
| **Raghvendra Singh** | `2401010367` | Research Analytics & UI/UX |


Evaluation parameters heavily target System Contextual Routing, Vector Retrieval Stability, Interface Precision, and Deployment Operations UX constraints mapped extensively throughout this repository.

> **Status:** Submission Ready ✅

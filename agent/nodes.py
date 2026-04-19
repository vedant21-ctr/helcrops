import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from agent.state import AgState
from rag.vector_db import get_retriever

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY", "")
    )

def input_node(state: AgState) -> AgState:
    return state

def prediction_node(state: AgState) -> AgState:
    # Prediction has already been populated from Milestone 1 app.py state insertion
    return state

def analysis_node(state: AgState) -> AgState:
    try:
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["input_data", "prediction", "risk"],
            template=(
                "Analyze crop yield prediction.\n"
                "Explain key factors.\n"
                "Do NOT hallucinate.\n"
                "Use only input data.\n"
                "If unsure, say insufficient data.\n\n"
                "Input: {input_data}\nPrediction: {prediction}\nRisk: {risk}\n\nAnalysis:"
            )
        )
        res = llm.invoke(prompt.format(input_data=state["crop_data"], prediction=state["prediction"], risk=state["risk_level"]))
        return {"analysis": res.content if hasattr(res, 'content') else str(res)}
    except Exception as e:
        return {"analysis": f"Error communicating with LLM: {str(e)}"}

def rag_node(state: AgState) -> AgState:
    retriever = get_retriever()
    if retriever:
        query = f"Best practices for {state['crop_data']['crop']} in {state['crop_data']['soil']} soil."
        docs = retriever.invoke(query)
    else:
        docs = []
    return {"retrieved_docs": docs}

def planning_node(state: AgState) -> AgState:
    try:
        llm = get_llm()
        docs = [d.page_content for d in state.get('retrieved_docs', [])]
        context = "\n".join(docs) if docs else "No specific documents found."
        
        prompt = PromptTemplate(
            input_variables=["context"],
            template=(
                "You are an agriculture expert.\n\n"
                "Rules:\n"
                "- Use retrieved RAG documents only\n"
                "- Do NOT invent information\n"
                "- Give practical steps\n"
                "- Do not fabricate agricultural advice\n\n"
                "Documents:\n{context}\n\nStrategy:"
            )
        )
        res = llm.invoke(prompt.format(context=context))
        return {"recommendations": res.content if hasattr(res, 'content') else str(res)}
    except Exception as e:
        return {"recommendations": f"Strategy unavailable due to LLM error: {str(e)}"}

def report_node(state: AgState) -> AgState:
    try:
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["analysis", "strategy", "docs", "crop", "pred", "risk"],
            template=(
                "Generate structured report:\n\n"
                "Sections:\n"
                "- Crop Summary: {crop}\n"
                "- Yield Prediction: {pred}\n"
                "- Risk Analysis: {risk}\n"
                "- Action Plan: {strategy}\n"
                "- Sources: {docs}\n"
                "- Disclaimer: This is an AI advice summary.\n\n"
                "Keep it clear and professional.\n\nReport:"
            )
        )
        sources = [d.metadata.get("topic", "N/A") for d in state.get("retrieved_docs", [])]
        
        res = llm.invoke(prompt.format(
            analysis=state.get("analysis", ""),
            strategy=state.get("recommendations", ""),
            docs=", ".join(sources) if sources else "None",
            crop=state['crop_data']['crop'],
            pred=state['prediction'],
            risk=state['risk_level']
        ))
        return {"report": res.content if hasattr(res, 'content') else str(res)}
    except Exception as e:
        return {"report": f"Report generation failed: Make sure GROQ_API_KEY is actively exported into your terminal environment.\\nError: {str(e)}"}

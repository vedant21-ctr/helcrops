from langgraph.graph import StateGraph, END
from agent.state import AgState
from agent.nodes import input_node, prediction_node, rag_node, analysis_node, planning_node, report_node

def build_graph():
    workflow = StateGraph(AgState)
    
    workflow.add_node("Input", input_node)
    workflow.add_node("Prediction", prediction_node)
    workflow.add_node("RAG", rag_node)
    workflow.add_node("Analysis", analysis_node)
    workflow.add_node("Planning", planning_node)
    workflow.add_node("Report", report_node)
    
    # Workflow steps
    workflow.add_edge("Input", "Prediction")
    workflow.add_edge("Prediction", "RAG")
    workflow.add_edge("RAG", "Analysis")
    workflow.add_edge("Analysis", "Planning")
    workflow.add_edge("Planning", "Report")
    workflow.add_edge("Report", END)
    
    workflow.set_entry_point("Input")
    return workflow.compile()

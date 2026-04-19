from typing import TypedDict, List, Dict, Any

class AgState(TypedDict, total=False):
    # Inputs
    crop_data: Dict[str, Any]
    query: str
    
    # Milestone 1 ML outputs passed in
    prediction: float
    risk_level: str
    
    # Internal agent state
    retrieved_docs: List[Any]
    analysis: str
    recommendations: str
    
    # Output
    report: str

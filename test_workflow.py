import os
from dotenv import load_dotenv
load_dotenv()

from agent.graph import build_graph

print("GROQ API KEY LOADED:", os.environ.get("GROQ_API_KEY", "MISSING")[:10] + "...")

graph = build_graph()
try:
    res = graph.invoke({
        "crop_data": {
            "crop": "Cotton", "soil": "Loamy", "rainfall": 500,
            "temperature": 25, "ph": 6.5, "fertilizer": 100
        },
        "prediction": 4.5,
        "risk_level": "Medium",
        "query": "Give me best practices."
    })
    print("SUCCESS!")
    print(res["report"])
except Exception as e:
    print("WORKFLOW FAILED WITH EXCEPTION:")
    import traceback
    traceback.print_exc()

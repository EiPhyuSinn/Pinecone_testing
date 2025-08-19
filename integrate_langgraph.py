import os
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Dict, Optional, List
load_dotenv()

class GraphState(TypedDict):
    query: str
    matches: List[dict]
    namespace: str
    recommendation: Optional[str] 


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "eps-testing"
index = pc.Index(index_name)

def retrieve_movies(state):
    query = state["query"]
    namespace = state.get("namespace", "happy-ending")  
    query_embed = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "passage", "truncate": "END"}
    )[0].values
    
    results = index.query(
        vector=query_embed,
        top_k=3,
        include_metadata=True,
        namespace=namespace
    )
    
    return {
        "matches": results["matches"],
        "query": query,
        "namespace": namespace
    }

def generate_recommendation(state):
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)
    
    matches_info = []
    for match in state["matches"]:
        meta = match["metadata"]
        matches_info.append(
            f"Title: {meta['title']} ({meta['year']})\n"
            f"Summary: {meta['summary']}\n"
            f"Box Office: ${meta['box_office']:,}\n"
            f"Relevance Score: {match['score']:.2f}"
        )
    
    context = "\n\n".join(matches_info)
    
    response = llm.invoke(
        f"Based on these movie matches for '{state['query']}' "
        f"(namespace: {state['namespace']}):\n\n{context}\n\n"
        "Please analyze these recommendations and suggest which one might be "
        "most interesting to watch, considering the user's query."
    )
    return {"recommendation": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_movies)
workflow.add_node("generate", generate_recommendation)
workflow.add_edge("retrieve", "generate")
workflow.set_entry_point("retrieve")

app = workflow.compile()

result = app.invoke({
    "query": "sci-fi battle",
    "namespace": "happy-ending"
})

print("\nFinal Recommendation:")
print(result["recommendation"])
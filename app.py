import os
import streamlit as st
from pinecone import Pinecone
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Dict, Optional, List

load_dotenv()

class GraphState(TypedDict):
    query: str
    matches: List[dict]
    recommendation: Optional[str] 

# ---- Setup Pinecone ----
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "movies-walkthrough"
index = pc.Index(index_name)

# ---- Retrieval Function ----
def retrieve_movies(state):
    query = state["query"]
    query_embed = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "passage", "truncate": "END"}
    )[0].values
    
    results = index.query(
        vector=query_embed,
        top_k=3,
        include_metadata=True
    )
    
    return {
        "matches": results["matches"],
        "query": query,
    }

# ---- Recommendation Function ----
def generate_recommendation(state):
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.7)
    
    matches_info = []
    for match in state["matches"]:
        meta = match["metadata"]
        matches_info.append(
            f"**Title:** {meta['title']} ({meta['year']})\n"
            f"**Summary:** {meta['summary']}\n"
            f"**Relevance Score:** {match['score']:.2f}"
        )
    
    context = "\n\n".join(matches_info)
    
    response = llm.invoke(
        f"Based on these movie matches for '{state['query']}'\n\n{context}\n\n"
        "Please suggest the best movie recommendation."
    )
    return {"recommendation": response.content}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_movies)
workflow.add_node("generate", generate_recommendation)
workflow.add_edge("retrieve", "generate")
workflow.set_entry_point("retrieve")
app = workflow.compile()

st.title("🎬 Movie Recommendation System")
st.write("Ask me in natural language, e.g. *recommend me a kid movie with animals*")

query = st.text_input("Enter your query:")

if st.button("Get Recommendation") and query.strip():
    with st.spinner("Finding the best recommendation..."):
        result = app.invoke({"query": query})
        
        st.subheader("✅ Final Recommendation")
        st.write(result["recommendation"])
        
        st.subheader("🎯 Top Matches from Pinecone")
        for match in result["matches"]:
            meta = match["metadata"]
            st.markdown(
                f"**{meta['title']} ({meta['year']})**\n\n"
                f"- Summary: {meta['summary']}\n"
                f"- Relevance Score: {match['score']:.2f}"
            )

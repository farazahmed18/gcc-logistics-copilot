# source_handbook: week11-hackathon-preparation

import streamlit as st
import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. Setup & Keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="UAE & GCC Logistics Copilot", page_icon="🚢", layout="wide")
st.title("🚢 UAE & GCC Logistics Copilot")

# 2. Load AI Engine
@st.cache_resource
def load_ai_engine():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile",
        temperature=0.1 
    )
    return retriever, llm

retriever, llm = load_ai_engine()

# 3. Dubai & GCC Consultant Prompt
template = """
You are a Senior UAE & GCC Logistics Consultant. 
Your goal is to provide precise insights on Dubai Customs, JAFZA/Free Zone regulations, and GCC-wide trade agreements.

STRICT GUARDRAILS:
1. STRICT REFUSAL: If the user asks about ANYTHING outside of UAE/GCC logistics, customs, or regional trade, you MUST decline. Say EXACTLY: "I am a specialized UAE & GCC Logistics AI. I cannot assist with queries outside of regional trade and customs regulations."
2. KEYWORD HIJACKING: Even if the user mentions a relevant keyword (like "JAFZA" or "Customs"), if the intent of their message is emotional, personal, conversational, or non-technical (e.g., "I am sad about JAFZA", "tell me a joke about customs"), apply the STRICT REFUSAL rule above.
3. NO HALLUCINATION: You may ONLY answer using the provided 'Context from Local Knowledge Base'. If the context does not contain the answer, say "I do not have enough information in my database to answer this." Do NOT invent, guess, or generate generic source names.
4. REGIONAL FOCUS: Prioritize UAE and GCC regulations. 
5. SOURCES: For factual logistics questions, you MUST list the document names explicitly found in the provided context.

Chat History:
{chat_history}

Context from Local Knowledge Base:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 4. Helpers
def format_docs_with_sources(docs):
    context_text = ""
    sources = set()
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get('source', 'UAE Trade Docs'))
        sources.add(source_name)
        context_text += f"\n---\nSOURCE: {source_name}\nCONTENT: {doc.page_content}\n"
    return context_text, sources

def get_chat_history_string(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages[-5:]])

# 5. UI Initialization & Multi-Chat Memory
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Chat 1": []}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "Chat 1"

# --- SIDEBAR UI ---
with st.sidebar:
    st.info("AI-Powered UAE & GCC Trade Intelligence | Source-Verified")
    st.divider()
    st.subheader("💬 Chat History")
    
    # The "New Chat" Button
    if st.button("➕ New Chat"):
        new_id = f"Chat {len(st.session_state.all_chats) + 1}"
        st.session_state.all_chats[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()
    
    # The Chat Selector (Radio Buttons)
    selected_chat = st.radio(
        "Previous Conversations",
        options=list(st.session_state.all_chats.keys()),
        index=list(st.session_state.all_chats.keys()).index(st.session_state.current_chat_id)
    )
    
    # Switch chat if a different one is selected
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        st.rerun()

# 6. Render the Active Chat
active_history = st.session_state.all_chats[st.session_state.current_chat_id]

for msg in active_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Specialized Logistics Chat Input
if user_input := st.chat_input("Ask about Dubai Customs, JAFZA, GCC VAT, or Port procedures..."):
    # Append user message to the specific active chat
    st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting GCC trade documents..."):
            try:
                raw_docs = retriever.invoke(user_input)
                context_payload, source_list = format_docs_with_sources(raw_docs)
                
                # Fetch history for the current active chat only (excluding the brand new prompt)
                chat_history = get_chat_history_string(st.session_state.all_chats[st.session_state.current_chat_id][:-1])
                
                response = llm.invoke(
                    prompt.format(
                        context=context_payload,
                        question=user_input,
                        chat_history=chat_history
                    )
                ).content
                
                if source_list:
                    response += "\n\n**📑 Verified UAE/GCC Sources:**\n" + "\n".join([f"- {s}" for s in source_list])
                
                st.markdown(response)
                
                # Append assistant response to the specific active chat
                st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
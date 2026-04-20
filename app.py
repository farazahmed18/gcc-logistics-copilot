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
st.sidebar.info("AI-Powered UAE & GCC Trade Intelligence | Source-Verified")

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

STRICT RULES:
1. REGIONAL FOCUS: Prioritize UAE and GCC regulations. If the answer involves Dubai Customs or GCC Customs Union, cite those specific standards.
2. SOURCES: You MUST list the document names (e.g., JAFZA Guidebook, Dubai Customs Guide) used to generate the answer.
3. PERSONALITY: Be professional and authoritative on Middle Eastern trade corridors.

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

# 5. UI Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Specialized Logistics Chat Input
if user_input := st.chat_input("Ask about Dubai Customs, JAFZA, GCC VAT, or Port procedures..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting GCC trade documents..."):
            try:
                raw_docs = retriever.invoke(user_input)
                context_payload, source_list = format_docs_with_sources(raw_docs)
                chat_history = get_chat_history_string(st.session_state.messages[:-1])
                
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
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
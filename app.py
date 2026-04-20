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

# --- LANGSMITH OBSERVABILITY SETUP ---
# LangChain looks for these specific OS environment variables to enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGCHAIN_PROJECT"] = "Logistics_Copilot_Hackathon"

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
You are a Senior UAE & GCC Logistics Consultant. You have access to a specialized 'Local Knowledge Base' containing official trade regulations and customs documents.

### INFORMATION HIERARCHY:
1. PRIMARY SOURCE: If the 'Context from Local Knowledge Base' contains information relevant to the question, you MUST use it as your primary source. Mention specifically what the documents say.
2. SECONDARY SOURCE: If the context is empty or insufficient, but the question is about UAE/GCC logistics, use your internal professional knowledge to provide an answer. Clearly state that this is "General Logistics Guidance."
3. OUT OF SCOPE: If the question is NOT about UAE/GCC trade, customs, or logistics, you MUST decline. Say exactly: "I am a specialized UAE & GCC Logistics AI. I cannot assist with queries outside of regional trade and customs regulations."

### SOURCE CITATION RULES:
- ONLY list document names under 'Verified UAE/GCC Sources' if they actually appear in the 'Context' section below. 
- If you are answering from your own knowledge because the context was empty, do NOT list any sources.

### CONTEXT FROM LOCAL KNOWLEDGE BASE:
{context}

### CHAT HISTORY:
{chat_history}

### USER QUESTION:
{question}

### CONSULTANT RESPONSE:
"""

prompt_template = ChatPromptTemplate.from_template(template)

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
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {"Chat 1": "New Chat"}
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
        st.session_state.chat_titles[new_id] = "New Chat"
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()
    
    # The Chat Selector (Radio Buttons)
    selected_chat = st.radio(
        "Previous Conversations",
        options=list(st.session_state.all_chats.keys()),
        format_func=lambda x: st.session_state.chat_titles.get(x, x),
        index=list(st.session_state.all_chats.keys()).index(st.session_state.current_chat_id)
    )
    
    # Switch chat if a different one is selected
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        st.rerun()

# 6. Render the Active Chat History
active_history = st.session_state.all_chats[st.session_state.current_chat_id]

for msg in active_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Specialized Logistics Chat Input
if user_input := st.chat_input("Ask about Dubai Customs, JAFZA, GCC VAT, or Port procedures..."):
    
    # --- GENERATE SMART TITLE (On First Message Only) ---
    if len(st.session_state.all_chats[st.session_state.current_chat_id]) == 0:
        try:
            title_prompt = f"Summarize this logistics query into 3 words maximum for a sidebar menu title. Do not use quotes or periods. Query: {user_input}"
            smart_title = llm.invoke(title_prompt).content.strip('". ')
            st.session_state.chat_titles[st.session_state.current_chat_id] = smart_title
        except:
            pass 

    # Append user message to the specific active chat
    st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting GCC trade documents..."):
            try:
                # 1. Get the docs
                raw_docs = retriever.invoke(user_input)
                context_payload, source_list = format_docs_with_sources(raw_docs)
                
                # 2. Get history
                chat_history_str = get_chat_history_string(st.session_state.all_chats[st.session_state.current_chat_id][:-1])
                
                # 3. Always call the LLM, but let the PROMPT handle the refusal
                response = llm.invoke(
                    prompt_template.format(
                        context=context_payload,
                        question=user_input,
                        chat_history=chat_history_str
                    )
                ).content
                
                # 4. Only show "Verified Sources" if they actually exist
                if source_list and "I cannot assist" not in response:
                    response += "\n\n**📑 Verified UAE/GCC Sources:**\n" + "\n".join([f"- {s}" for s in source_list])
                
                st.markdown(response)
                st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
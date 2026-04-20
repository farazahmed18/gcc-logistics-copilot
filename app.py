import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. SETUP & OBSERVABILITY
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LangSmith Observability Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGCHAIN_PROJECT"] = "Logistics_Copilot_Hackathon"

# ==========================================
# 2. PAGE CONFIG & CUSTOM CSS
# ==========================================
# Layout centered for a much cleaner, tighter chat interface
st.set_page_config(page_title="UAE & GCC Logistics Copilot", page_icon="🚢", layout="centered", initial_sidebar_state="expanded")

# Inject Custom CSS for an Enterprise feel
st.markdown("""
    <style>
    /* Pull the main content up and remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 🎯 Safely hide ONLY the Deploy button */
    .stAppDeployButton {display: none !important;}
    
    /* Style the main title */
    .stMarkdown h1 {
        color: #0f172a;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Give chat bubbles a clean, bordered box look */
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚢 UAE & GCC Logistics Copilot")

# ==========================================
# 3. AI ENGINE LOAD
# ==========================================
@st.cache_resource
def load_ai_engine():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile",
        temperature=0.1 
    )
    return retriever, llm

retriever, llm = load_ai_engine()

# ==========================================
# 4. CONSULTANT PROMPT TEMPLATE
# ==========================================
template = """
You are a Senior UAE & GCC Logistics Consultant. 

### SCOPE & MISSION:
- You are an expert in Dubai Customs, JAFZA/Free Zones, GCC Trade Agreements, and REGIONAL TAX LAWS (VAT/Excise).
- If a user asks about specific Articles, Laws, or Penalties related to UAE/GCC trade, this is IN-SCOPE.
- Your goal is to be helpful and professional, providing guidance based on the documents or your internal expertise.

### REFUSAL PROTOCOL:
- ONLY refuse if the question is 100% unrelated to business/trade (e.g., "how to make tea", "tell me a joke", "I'm feeling sad"). 
- For those cases, say: "I am a specialized UAE & GCC Logistics AI. I cannot assist with queries outside of regional trade and customs regulations."

### INFORMATION SOURCES:
1. PRIMARY: Use the 'Context from Local Knowledge Base' if it contains the answer.
2. SECONDARY: If the context is empty, use your internal knowledge to provide 'General Logistics Guidance'. Do NOT say "match not found"—just provide the answer professionally.

Context from Local Knowledge Base:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

prompt_template = ChatPromptTemplate.from_template(template)

# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================
def format_docs_with_sources(docs):
    context_text = ""
    sources = set()
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get('source', 'UAE Trade Docs'))
        sources.add(source_name)
        context_text += f"\n---\nSOURCE: {source_name}\nCONTENT: {doc.page_content}\n"
    return context_text, list(sources)

def get_chat_history_string(messages):
    return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages[-5:]])

# ==========================================
# 6. UI SESSION STATE & SIDEBAR
# ==========================================
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Chat 1": []}
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {"Chat 1": "New Chat"}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "Chat 1"

with st.sidebar:
    st.title("🚢 GCC Intelligence")
    st.info("💡 **Pro-Tip:** Ask about JAFZA regulations, import steps, or VAT penalties for source-verified results.")
    
    # System Status Indicators
    st.success("Knowledge Base: Online")
    st.success("Tracing: LangSmith Active")
    st.divider()
    
    # --- MOVED METRICS HERE ---
    with st.expander("⚙️ System Architecture"):
        st.metric("🧠 Core LLM", "Llama 3.3 (70B)")
        st.metric("⚡ Orchestration", "LangChain & Groq")
        st.metric("📚 Retrieval Engine", "ChromaDB Local")
    st.divider()
    
    st.subheader("Conversations")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = f"Chat {len(st.session_state.all_chats) + 1}"
        st.session_state.all_chats[new_id] = []
        st.session_state.chat_titles[new_id] = "New Chat"
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()
    
    # Chat History Selector
    selected_chat = st.radio(
        "History",
        options=list(st.session_state.all_chats.keys()),
        format_func=lambda x: st.session_state.chat_titles.get(x, x),
        index=list(st.session_state.all_chats.keys()).index(st.session_state.current_chat_id),
        label_visibility="collapsed"
    )
    
    # Switch Chat Logic
    if selected_chat != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat
        st.rerun()

# ==========================================
# 7. RENDER ACTIVE CHAT & PROCESS INPUT
# ==========================================
active_history = st.session_state.all_chats[st.session_state.current_chat_id]

for msg in active_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about Dubai Customs, JAFZA, GCC VAT, or Port procedures..."):
    
    # Generate Smart Title (On First Message Only)
    if len(st.session_state.all_chats[st.session_state.current_chat_id]) == 0:
        try:
            title_prompt = f"Summarize this logistics query into 3 words maximum for a sidebar menu title. Do not use quotes or periods. Query: {user_input}"
            smart_title = llm.invoke(title_prompt).content.strip('". ')
            st.session_state.chat_titles[st.session_state.current_chat_id] = smart_title
        except:
            pass 

    # Append user message
    st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consulting GCC trade documents..."):
            try:
                # 1. Retrieve Docs
                raw_docs = retriever.invoke(user_input)
                context_payload, source_list = format_docs_with_sources(raw_docs)
                
                # 2. Extract History
                chat_history_str = get_chat_history_string(st.session_state.all_chats[st.session_state.current_chat_id][:-1])
                
                # 3. Generate LLM Response
                response = llm.invoke(
                    prompt_template.format(
                        context=context_payload,
                        question=user_input,
                        chat_history=chat_history_str
                    )
                ).content
                
                # 4. Render Response and UI Expander for Sources
                if source_list and "I cannot assist" not in response and "General Logistics Guidance" not in response:
                    st.markdown(response)
                    with st.expander("🔍 View Verified UAE/GCC Sources"):
                        for source in source_list:
                            st.markdown(f"- 📄 `{source}`")
                    
                    # Ensure memory gets both the text and the source notation
                    memory_response = response + "\n\n*(Sources Verified in Knowledge Base)*"
                    st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "assistant", "content": memory_response})
                else:
                    st.markdown(response)
                    st.session_state.all_chats[st.session_state.current_chat_id].append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
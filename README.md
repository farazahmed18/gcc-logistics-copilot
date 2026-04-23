<div align="center">
  <h1>🚢 UAE & GCC Logistics Copilot</h1>
  <h3><i>Empowering Trade with AI-Driven Intelligence</i></h3>
  <br>
  <a href="https://gcc-logisticsai-faraz.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit" alt="Live Demo">
  </a>
</div>

<hr>

<h2> Project Overview</h2>
<p><b>Problem:</b> Navigating UAE and GCC logistics involves manually parsing hundreds of pages of complex, region-specific customs and free zone manuals, often leading to costly compliance errors.</p>
<p><b>Value Proposition:</b> The UAE & GCC Logistics Copilot instantly retrieves and synthesizes exact fees, steps, and regulations, saving hours of research and preventing unexpected supply chain costs.</p>
<p><b>AI Architecture:</b> This project utilizes an advanced <b>Retrieval-Augmented Generation (RAG)</b> pipeline. It leverages <b>Groq's Llama 3.3</b> and <b>ChromaDB</b> to extract, reason over, and accurately cite specific local trade documents while maintaining conversational memory.</p>

<hr>

<h2> Key Features</h2>
<ul>
  <li><b>Dynamic Knowledge Suggestions:</b> The app utilizes the LLM to read the vector database on load, generating randomized, context-aware prompt suggestions.</li>
  <li><b>Multi-Chat Session Memory:</b> Maintains distinct chat histories, allowing for deep, conversational follow-ups on complex customs procedures.</li>
  <li><b>Enterprise Guardrails:</b> Implements a strict "Refusal Protocol" prompt architecture to defend against prompt injection and restrict answers strictly to regional trade laws.</li>
  <li><b>Theme-Aware UI:</b> Fully responsive frontend engineered with custom CSS to support both Light and Dark mode seamlessly.</li>
</ul>

<hr>

<h2> Tech Stack</h2>
<ul>
  <li><b>Core LLM:</b> Llama 3.3 (70B) via Groq for ultra-low latency execution.</li>
  <li><b>Vector Database:</b> ChromaDB (Local Persistence).</li>
  <li><b>Orchestration:</b> LangChain.</li>
  <li><b>Frontend:</b> Streamlit (with custom UI/UX injection).</li>
  <li><b>Observability:</b> LangSmith (Full trace auditing & latency tracking).</li>
  <li><b>Embeddings:</b> HuggingFace (<code>all-MiniLM-L6-v2</code>).</li>
</ul>

<hr>

<h2> Security & Local Setup</h2>
<p>This project follows industry-standard security protocols. <b>Secrets are never hard-coded into the codebase.</b></p>

<h3>Installation:</h3>
<ol>
  <li><b>Clone the Repository:</b>
    <pre><code>git clone https://github.com/farazahmed18/gcc-logistics-copilot.git</code></pre>
  </li>
  <li><b>Environment Setup:</b>
    <ul>
      <li>Create a <code>.env</code> file in the root directory.</li>
      <li>Add your required API keys (e.g., <code>GROQ_API_KEY</code>, <code>LANGCHAIN_API_KEY</code>).</li>
    </ul>
  </li>
  <li><b>Install Dependencies:</b>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li><b>Launch the App:</b>
    <pre><code>streamlit run app.py</code></pre>
  </li>
</ol>

<hr>

<h2> Observability & Reliability</h2>
<p>To solve the "black box" problem of traditional AI, this project implements <b>LangSmith Tracing</b>:</p>
<ul>
  <li><b>Source Verification:</b> Every response is cross-referenced against the local vectorstore, and exact document names are cited in the UI.</li>
  <li><b>Traceability:</b> Developers can audit the exact document chunks retrieved and monitor retrieval latency (averaging ~0.02s) for every user query to ensure zero-hallucination.</li>
</ul>

<hr>

<div align="center">
  <p><i>Developed by Faraz Ahmed Siddiqui for the GenAI Engineering Hackathon - 2026</i></p>
</div>

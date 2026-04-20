<h1 align="center">🚢 UAE & GCC Logistics Copilot</h1>

<div align="center">
  <h3><i>Empowering Trade with AI-Driven Intelligence</i></h3>
</div>

<hr>

<h2> Project Overview</h2>
<p>
  <b>Problem:</b> Navigating UAE and GCC logistics involves manually parsing hundreds of pages of complex, region-specific customs and free zone manuals, often leading to costly compliance errors.
</p>
<p>
  <b>Value Proposition:</b> The UAE & GCC Logistics Copilot instantly retrieves and synthesizes exact fees, steps, and regulations, saving hours of research and preventing unexpected supply chain costs.
</p>
<p>
  <b>AI Architecture:</b> This project utilizes a <b>Retrieval-Augmented Generation (RAG)</b> pipeline. It leverages <b>Groq's Llama 3.3</b> and <b>ChromaDB</b> to extract, reason over, and accurately cite specific local trade documents while maintaining conversational memory.
</p>



<hr>

<h2> Tech Stack</h2>
<ul>
  <li><b>LLM:</b> Llama 3.3 (via Groq for ultra-low latency)</li>
  <li><b>Vector Database:</b> ChromaDB (Local Persistence)</li>
  <li><b>Orchestration:</b> LangChain</li>
  <li><b>Frontend:</b> Streamlit</li>
  <li><b>Observability:</b> LangSmith (Full trace auditing)</li>
  <li><b>Embeddings:</b> HuggingFace (<code>all-MiniLM-L6-v2</code>)</li>
</ul>

<hr>

<h2> Security & Setup</h2>
<p>This project follows industry-standard security protocols for API key management. <b>Secrets are never hard-coded into the codebase.</b></p>

<h3>Local Installation:</h3>
<ol>
  <li><b>Clone the Repository:</b>
    <pre><code>git clone https://github.com/farazahmed18/gcc-logistics-copilot.git</code></pre>
  </li>
  <li><b>Environment Setup:</b>
    <ul>
      <li>Create a <code>.env</code> file in the root directory.</li>
      <li>Refer to <code>.env.example</code> for required keys (<code>GROQ_API_KEY</code>, <code>LANGCHAIN_API_KEY</code>).</li>
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
  <li><b>Source Verification:</b> Every response is cross-referenced against the local vectorstore.</li>
  <li><b>Traceability:</b> Developers can audit the exact document chunks retrieved for every user query to ensure zero-hallucination.</li>
</ul>

<hr>

<p align="center">
  <i>Developed for the AI Engineering Hackathon - 2026</i>
</p>

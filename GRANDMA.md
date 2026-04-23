# So! Here is what I built:

### 1. Concept
Ummmm...Imagine you want to ship a big box of goods to Dubai. So to do it legally, you usually have to read thousands of pages of confusing government rulebooks so as you dont get fined or your stuff dont get rejected. For that I built a smart digital assistant that does the reading and all the rule checks for you. So whether if the person is an entrepreneur trying to launch his new business in JAFZA or a massive logistic company managing hundreds of daily shipments, they just need to ask a question to m app. My program gives them the exact rule instantly, saving them hours of frustration and making global trade much easier.

### 2. GenAI Part
Normally, AI can sometimes guess or hallucinate if it doesn't know the answer. So to stop that, I uploaded the official shipping rulebooks inside my program. Now, when someone asks a question, my program acts like a super-fast librarian. It finds the exact paragraph in the rulebook first, and then explains it in plain English. It never guesses, so people don't make any legal mistake. But if a user asks a rare question and the answer is not in the rulebook then my program uses its general logistic knowledge to offer safe amd standard industry advice to guide the user in right direction. 

### 3. Stack

* **The Frontend:** Ive used streamlit to deploy my app. Ive worked a lot on the UI..so instead of a basic prototype, I made a custom theme-aware UI that also has different prompt suggestions and the app also has multi chat session memory.
* **The Brain (Llama & Groq):** The super-fast engine that understands what the person is asking. It reads all the complex rules and instantly translates them into easy to read and understandable answer so the user doesnt have to wait long and understands easily
* **The VectorStore (ChromaDB):** This is the secure digital vault where I stored all the government rulebooks. First i uploaded all the PDFs in the rag_data file...then my code reads those PDFs and chops them into smaller paragraphs and uses an embedding model (HuggingFace) to translate those english paragraphs to vectors. 
* **LangChain:** This is the tool that makes the Brain and the VectorStore talk to each other smoothly.
* **LangSmith:** This is where i watch all the ongoing chats and all the activities in real-time.

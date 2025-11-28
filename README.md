# Smart JISA – Smart Jira Issue Similarity Agent

Smart JISA is a **multi‑agent, LLM‑powered assistant** that helps QA and engineering teams quickly find **similar or duplicate Jira issues** based on natural language bug reports.

Given a new issue (title + description), Smart JISA:

1. Cleans and normalizes the text  
2. Generates an embedding for semantic understanding  
3. Searches a historical Jira issue dataset for similar tickets  
4. Produces a **triage summary** with likely duplicates and recommendations  

It’s inspired by the workflows used in real QA triage and by the agent patterns from the Kaggle **“5 Days of AI”** course.

---

## ✨ Features

- 🔍 **Semantic duplicate detection** using embeddings (not just keyword search)  
- 🧠 **Multi‑agent architecture** (ingestion, similarity, reporting)  
- 📚 Works on a local **JSON dataset** of Jira issues  
- 🧾 Generates a **human‑readable triage report** for QA engineers  
- 🧱 Extensible design: easy to plug into a web UI, API, or Jira integration later  

---

## 🏗 Architecture Overview

Smart JISA is structured as a **root agent + sub‑agents + tools**:

- **Root (conceptual)** – `smart_jisa_root_agent`  
  - Orchestrates the overall flow (ingestion → similarity → reporting)

- **Sub‑Agents & Tools**
  - 🧹 `IngestionAgent`  
    - Cleans and normalizes the Jira issue title + description  
    - Uses `utils/text_cleaner.py`

  - 📐 `embeddingAgent`  
    - Generates text embeddings using Gemini (or a deterministic fallback)  
    - Used inside the similarity layer

  - 📊 `SimilarityAgent`  
    - Builds/loads a vector index over historical Jira issues  
    - Uses `utils/vector_store.py` (FAISS or numpy fallback)  
    - Returns top‑K similar issues with scores

  - 🧾 `report_agent` (LLM agent)  
    - Receives cleaned text + similar issues as JSON  
    - Produces a short triage summary:
      - Likely duplicates  
      - Related issues  
      - Recommended action (link as duplicate, investigate module, etc.)

Agents are implemented with a lightweight :

- `Gemini` – wrapper around `google-generativeai` (with a safe dummy fallback)  
- `Agent` – holds model, name, description, and instructions  
- `Runner` + `InMemorySessionService` – manage stateful interactions and history  

---

## 📁 Project Structure

```text
smart-jisa/
├─ agents/
│  ├─ llm_framework.py      # Gemini wrapper, Agent, Runner, InMemorySessionService
│  ├─ ingestion_agent.py    # IngestionAgent (text cleaning)
│  ├─ embedding_agent.py    # embeddingAgent() using Gemini or fallback
│  ├─ similarity_agent.py   # SimilarityAgent (vector search over Jira issues)
│  └─ jisa_agents.py        # Root pipeline: tools + report_agent wiring
│
├─ utils/
│  ├─ config.py             # Env/config (GOOGLE_API_KEY, app/user IDs)
│  ├─ text_cleaner.py       # Simple text normalization utilities
│  └─ vector_store.py       # VectorStore using FAISS or numpy
│
├─ data/
│  └─ jira_issues.json      # Sample Jira issues dataset (you can expand this)
│
├─ main.py                  # CLI entrypoint: run Smart JISA from the terminal
├─ requirements.txt         # Python dependencies
├─ .env                     # Local environment variables (not committed)
└─ .gitignore               # Ignore venv, .env, cache, vector index, etc.

🔧 Setup
1. Clone the project:
git clone https://github.com/<your-username>/smart-jisa.git
cd smart-jisa

2. Create a virtual environment:
 python -m venv .venv
.venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Add your Gemini API key:
Create a .env file:
GOOGLE_API_KEY=your_api_key_here

▶️ Run Smart JISA
python main.py
You’ll be asked for:
Jira issue title
Jira issue description

The system returns:
Cleaned input
Top similar tickets
A triage summary from the report agent

📌 Example Use Case
“Cart total wrong with discount”
“Promo code double‑applies, final price becomes too low.”

Smart JISA will detect similar historical tickets (e.g., cart calculation bugs) and recommend linking as duplicate.







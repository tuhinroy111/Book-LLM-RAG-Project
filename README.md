# Universal Book RAG – Project Overview

**Universal Book RAG** is a Python-based retrieval-augmented generation (RAG) application built with Streamlit. It allows users to upload a PDF (e.g. a book) and interactively ask questions about its content. The system uses a semantic search index (FAISS) and a Google LLM (Gemini) to retrieve relevant document snippets and generate answers. It also includes a “Control Panel” mode that runs batch evaluations against a golden QA dataset, scoring answers by precision, recall, faithfulness, and relevancy using an Anthropic API. This modular design (indexer, router, orchestrator, evaluator) makes the codebase clear and testable, showcasing proficiency with modern Python AI stacks (LangChain, HuggingFace embeddings, Google GenAI, etc.). RAG techniques enhance LLM accuracy by **augmenting** the model with authoritative knowledge from the uploaded document. 

# Repo Structure

```
/Book-LLM-RAG-Project/
├── .env                        # API keys (Google GenAI, Anthropic)
├── .gitignore                  # Ignored files and folders
├── requirements.txt            # Python dependencies (sentence-transformers, etc.)
├── documents/                  # (Empty directory for document assets)
├── testdata/
│   └── affirmation_doc_qa_golden_dataset.json  # Example Q&A test dataset
├── src/
│   ├── app.py                  # Streamlit UI (Chat & Control Panel)
│   ├── indexer.py              # PDF indexing into a FAISS vector store
│   ├── router.py               # Intent classification (GREETING, UNSAFE, DOCUMENT) via Gemini
│   ├── orchestrator.py         # RAG pipeline (query rewrite, retrieve, prompt LLM)
│   ├── llm_client.py           # Google GenAI client wrapper for text QA
│   ├── evaluator.py            # Scoring logic (precision, recall, relevancy)
│   ├── run_eval.py             # Batch evaluation script calling RAGEvaluator
│   ├── query.py               # Glue code: initializes components and exposes `ask_bot`
│   ├── test_available_models.py # Utility: list Anthropic models for API key verification
│   └── __pycache__/            # Compiled Python files (ignored)
└── tests/                      # (Empty placeholder for unit tests)
```

- The `.env` file (ignored by Git) stores `GOOGLE_API_KEY` and `ANTHROPIC_API_KEY` for LLM services. 
- `requirements.txt` lists core Python libraries. Additional packages (e.g. `streamlit`, `google-genai`, `anthropic`, `langchain-community`, `langchain-experimental`, `pydantic`, `python-dotenv`) are also needed.  
- The `testdata/` folder contains a sample JSON dataset for evaluation. 

# File Explanations

| File/Folder                                     | Purpose and Key Functionality                                                                                          |
|:------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| **`src/app.py`**                                | Streamlit application entrypoint. Handles the UI with two modes: **Chat** for interactive Q&A, and **Control Panel** for batch evaluation. Manages file uploads, user input, and displays answers/metrics using Streamlit widgets (columns, metrics, dataframes). |
| **`src/indexer.py`**                            | **DocumentIndexer**: Saves the uploaded PDF to a temp file, splits it into semantic chunks using `SemanticChunker`, and creates a FAISS vector index with HuggingFace embeddings (`all-MiniLM-L6-v2`). This enables fast similarity search for queries. |
| **`src/router.py`**                             | **IntentRouter**: Uses Google GenAI (Gemini 2.5) with a Pydantic schema to classify each question’s intent. It can categorize inputs as **GREETING**, **UNSAFE**, or **DOCUMENT**. If greeting or unsafe content is detected, it returns a canned `direct_response`; otherwise it routes to the RAG pipeline. This prevents inappropriate queries and handles simple interactions. |
| **`src/orchestrator.py`**                       | **RAGOrchestrator**: Orchestrates the end-to-end pipeline. It optionally rewrites follow-up questions into stand-alone queries, retrieves relevant text chunks from the FAISS index, constructs a final prompt with page citations, and calls the LLM to generate a detailed answer. It merges retrieved context and instructions for deep analysis. Returns the LLM’s answer plus the source snippets. |
| **`src/llm_client.py`**                         | **LLMClient**: Wraps the Google GenAI SDK (Vertex AI Gemini) for text generation and QA. It handles retries on API errors. Provides methods like `generate_response(query, context)` to ask questions based on provided context. (It can also embed text if needed.) |
| **`src/evaluator.py`**                          | **RAGEvaluator**: Computes evaluation metrics. Given a user query, expected behavior, retrieved snippets, and the LLM’s answer, it calculates precision/recall (matching overlapping words) and uses Claude (Anthropic) to score faithfulness and relevancy. Structured logging records each step for analysis. |
| **`src/run_eval.py`**                           | Batch evaluation harness. Loads a golden dataset JSON (`{"id","user_query","expected_answer","pass_fail_criteria",...}`), loops through each case, calls the RAG pipeline (`ask_bot`), and uses `RAGEvaluator.evaluate()` to compute metrics for each test. Returns a list of result dicts (displayed in a DataFrame in the UI). |
| **`src/query.py`**                              | Initializes and wires together singletons: `DocumentIndexer`, `LLMClient`, `IntentRouter`, and `RAGOrchestrator`. Exposes two functions: `process_uploaded_pdf(uploaded_file)` to build the vector store, and `ask_bot(question, db, history)` to query the orchestrator and get (answer, snippets). |
| **`src/test_available_models.py`**              | Utility script to verify the Anthropic API key. On running, it prints the list of accessible Claude models. Helps troubleshoot API access. |
| **`testdata/affirmation_doc_qa_golden_dataset.json`** | Example test dataset for evaluation: each entry includes `user_query`, a golden `expected_answer`, and `pass_fail_criteria`. Used in “Control Panel” mode. |
| **`.gitignore`**, **`.idea/`**, **`__pycache__/`** | Ignored files: IDE configs, caches, compiled files. |

# Architecture & Flowcharts

The system follows a modular **RAG architecture**. The key components are: a **Streamlit UI** (front-end), a **DocumentIndexer** (FAISS vector store), an **IntentRouter** (for query classification), a **RAGOrchestrator** (retrieval+generation), an **LLMClient** (Google GenAI), and an **Evaluator** (Anthropic). A high-level flow is illustrated below. 

```mermaid
flowchart LR
    U[User] --> UI[Streamlit UI]
    UI -->|Upload PDF| Indexer[DocumentIndexer (FAISS)]
    UI -->|Ask question| Q[User Question]
    Q --> Router[IntentRouter (Gemini)]
    Router -->|Intent=DOCUMENT| Orchestrator[RAG Orchestrator]
    Orchestrator --> Indexer
    Orchestrator -->|Search Query| VectorDB[FAISS Vector DB]
    Orchestrator -->|LLM Prompt| LLM[Google GenAI (Gemini)]
    LLM -->|Answer| Orchestrator
    Orchestrator --> A[Generated Answer]
    A --> UI
```

**Chat Flow:** In “Chat” mode, the user uploads a PDF, which the `DocumentIndexer` processes into a vector store. When the user asks a question, the `IntentRouter` first checks if it’s a greeting or unsafe; if it’s a valid document question, the `RAGOrchestrator` rewrites the question (if needed), retrieves top-matching text chunks from FAISS, and formulates a final prompt. This prompt is sent to the Google GenAI LLM to generate a detailed answer, which is returned and displayed. The response may cite page numbers from the document. 

**Evaluation Flow:** In “Control Panel” mode, the user also uploads a golden JSON dataset. When “Run Evaluation” is clicked, the app iterates over each test query, runs it through the same RAG pipeline (without user interactivity), and then scores the answer. The `RAGEvaluator` compares the LLM answer against the expected answer, computing precision/recall of context overlap and using Claude to judge faithfulness and relevancy. Aggregated metrics are shown on the UI.

```mermaid
flowchart LR
    User --> UI2[Streamlit UI (Control Panel)]
    UI2 -->|Upload PDF + JSON| Indexer2[DocumentIndexer]
    Indexer2 --> RAG2[RAG Orchestrator]
    RAG2 -->|Retrieve| VectorDB2[FAISS DB]
    RAG2 -->|Generate| LLM2[Google GenAI (Gemini)]
    LLM2 --> RAG2
    RAG2 --> Answer2[Answer]
    Answer2 --> Evaluator[RAG Evaluator (Anthropic)]
    Evaluator --> Metrics[Metrics (Precision, Recall, Faithfulness, Relevancy)]
    Metrics --> UI2
```

This illustrates how document content is converted into embeddings for retrieval, then combined with user queries to prompt the LLM. The design cleanly separates concerns (UI, vector search, LLM, evaluation), which aids maintainability and testing.

# Setup & Run Instructions

1. **Environment:** Ensure you have Python 3.9+ installed. It’s recommended to create a virtual environment:
   ```bash
   python -m venv rag-env
   source rag-env/bin/activate   # (Unix/Mac) or `rag-env\Scripts\activate` (Windows)
   ```

2. **Install Dependencies:**  
   Install required Python packages (in addition to `requirements.txt`):  
   ```bash
   pip install -r requirements.txt               # core dependencies
   pip install streamlit langchain langchain-community langchain-experimental google-genai anthropic pydantic python-dotenv
   ```  
   (This includes Streamlit, LangChain components, HuggingFace Embeddings, Google GenAI SDK, and the Anthropic SDK.)  
   For example, installing Streamlit as shown in [the official docs](#){data-citation="docs.streamlit"}:  
   ```
   pip install streamlit  
   ```

3. **Environment Variables:**  
   Create a `.env` file in the project root (already in `.gitignore`) with your API keys:  
   ```
   GOOGLE_API_KEY=<your Google GenAI API key>
   ANTHROPIC_API_KEY=<your Anthropic (Claude) API key>
   ```  
   It’s best practice to use `python-dotenv` as recommended by Anthropic’s docs. This file should **not** be committed to version control.

4. **Project Structure:**  
   - Place any PDF documents to test in `documents/` or upload them via the UI.  
   - The example golden JSON (`testdata/affirmation_doc_qa_golden_dataset.json`) follows the format:
     ```json
     [
       {
         "id": "case1",
         "category": "QA",
         "user_query": "What is ...?",
         "expected_answer": "Answer text ...",
         "pass_fail_criteria": "Context"
       },
       ...
     ]
     ```
   - No additional build steps are needed.

5. **Run the App:**  
   Launch the Streamlit app with:  
   ```bash
   streamlit run src/app.py
   ```  
   By default, it will open in your browser at `http://localhost:8501/`. (You can specify a port with `--server.port` if needed.)  

6. **Common Commands:**  
   - **Development:** As you update code, Streamlit will auto-reload. You can also run linters/tests if added (none by default).  
   - **Testing Models:** Use the `test_available_models.py` script to verify that the Anthropic API key is working:  
     ```bash
     python src/test_available_models.py
     ```  
   - **Batch Evaluation:** In Control Panel mode, upload the PDF and a properly formatted JSON, then click **Run Evaluation**. The metrics will display on the UI.

7. **Multiple Stack Setup:** (For reference, if this were a different stack.) Below is a table of how one might install and run the app in various technology stacks:

   | Stack                | Install Dependencies         | Run App                             | Test / Eval                        |
   |----------------------|------------------------------|-------------------------------------|------------------------------------|
   | **Node.js / React**  | `npm install`                | `npm start` or `npm run dev`        | `npm test` or custom script        |
   | **Python (Streamlit)** | `pip install -r requirements.txt` <br>`pip install streamlit google-genai anthropic langchain-community` | `streamlit run src/app.py`         | (Use `pytest` if tests were added) |
   | **Java (Spring Boot)** | `mvn clean install`         | `mvn spring-boot:run` <br> or run JAR | `mvn test`                        |
   | **Go**               | (dependencies in `go.mod`)    | `go run main.go` or `go build && ./app` | `go test ./...`                   |
   | **Rust**             | (dependencies in `Cargo.toml`)| `cargo run`                          | `cargo test`                       |

   Each environment has analogous steps: install deps, run the server, and execute tests, adjusted to the ecosystem’s tools.

# Troubleshooting

- **Missing API Keys:** If the app throws authentication errors, ensure that `.env` contains valid `GOOGLE_API_KEY` and `ANTHROPIC_API_KEY`, and that you’ve restarted the app (or reloaded environment variables). You can verify the Anthropic key with `test_available_models.py`.
- **Dependency Errors:** If Python raises `ModuleNotFoundError`, double-check that all libraries are installed. For example, if `langchain_community` or `langchain_experimental` is missing, install via `pip install langchain-community langchain-experimental`.
- **Streamlit UI Issues:**  
  - If the browser doesn’t open automatically, navigate to `http://localhost:8501/`.  
  - For port conflicts, use `streamlit run src/app.py --server.port 8502` (for example).  
- **Indexing Failures:** Very large PDFs may exceed memory. Ensure the file is a valid PDF. The `DocumentIndexer` uses a temp file; check for disk space. 
- **No Search Results:** If no answer appears, the question may be outside the document’s content. The app will notify if the PDF or dataset is missing. 
- **Evaluation JSON Format:** The control-panel JSON must be a list of objects with the expected fields (`user_query`, etc.). Malformed JSON will prevent results. Use a JSON validator if needed.
- **LLM Errors / Rate Limits:** API calls to Google or Anthropic can fail if keys are wrong or rate limits are hit. Check the console logs for error details and ensure network access.

# Notes from LLM Evaluation of the Project

- **Architecture & Design:** The project cleanly separates UI, indexing, routing, and generation. It demonstrates knowledge of RAG pipelines: indexing documents with FAISS, retrieving relevant context, and prompting an LLM. The use of structured classes (`DocumentIndexer`, `RAGOrchestrator`, etc.) shows modular design.
- **Technical Stack:** Built in Python, using Streamlit for the frontend. It leverages LangChain components for vector search and the Google GenAI SDK for the LLM. The presence of `requirements.txt` and `pip install` commands aligns with Python best practices. Environment variables are managed via `.env` (per [Anthropic’s guidance](#){data-citation="anthropic.docs"}).
- **Complexity:** The project integrates several advanced pieces (PDF parsing, semantic chunking, two LLM APIs, evaluation metrics). It implements conversational context handling (`_contextualize_query`) and intent routing, indicating familiarity with chat system nuances.
- **Testing & Quality:** There is a basic test script for model availability. However, no unit tests are provided (the `tests/` folder is empty), so typical testing practices (e.g. using `pytest`) appear incomplete. The code includes logging and data validation (Pydantic), which is good. 
- **Code Clarity:** The code is commented and uses descriptive names. The README should highlight usage instructions as above. Documentation could be improved by adding a `requirements.txt` for LLM and LangChain packages and fleshing out the `tests/` directory.
- **Overall:** This repository demonstrates a sophisticated project leveraging modern AI/ML tooling. A recruiter should note the use of API keys, Docker (if any, though none is provided), and the need for cloud LLM credentials to run. The architecture suggests capability with AI integrations and full-stack design, but adding test coverage and clarifying dependencies would strengthen it. 

**Sources:** RAG is described as “optimizing LLM output by referencing an external knowledge base”. The app uses embeddings for semantic search (embedding vectors allow dot-product similarity search). Streamlit apps install via `pip install streamlit` and keep secrets in `.env` (anthropic recommends dotenv for keys).
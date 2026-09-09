# Agentic RAG: Router-Retriever System

An agentic retrieval-augmented generation (RAG) system built with two cooperating CrewAI agents: a **Router Agent** that classifies each incoming question, and a **Retriever Agent** that fetches a grounded answer using one of three retrieval paths:

- **PDF** — searches a local PDF (via `PDFSearchTool`, backed by a Chroma vector store) for questions about document/paper content.
- **Web** — runs a live web search via Tavily for questions about current/recent information.
- **Direct** — answers straight from the LLM's own knowledge when neither retrieval path is needed.

Every step of the routing and retrieval process is recorded in an in-memory trace log (`TRACE_LOG`), giving a step-by-step view of the system's reasoning from question to final answer.

This project runs entirely on the OpenAI API and Tavily — it does not use Azure OpenAI or any Azure-specific configuration.

## Setup

1. Clone the repo:
   ```
   git clone https://github.com/alexlcortes/Agentic-RAG-Router-Retriever.git
   cd Agentic-RAG-Router-Retriever
   ```
2. Create and activate a virtual environment (Python 3.10+ required):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy the example environment file and fill in your keys:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and set:
   - `OPENAI_API_KEY` — your OpenAI API key (from platform.openai.com/api-keys)
   - `TAVILY_API_KEY` — your Tavily API key
   - `OPENAI_MODEL` and `EMBEDDING_PROVIDER` can be left at their defaults (`gpt-4o-mini` and `onnx`)
5. Open `router_retriever_rag.ipynb` and run all cells top to bottom (select the kernel pointing at your `.venv`).

The source PDF used for the PDF retrieval path lives in `data/pdfs/`.

## Coordination flow

1. A question comes into `run_agentic_rag(question)`, which clears `TRACE_LOG` and logs `question_received`.
2. A `Task` is built for `router_agent`, instructing it to classify the question using the `Route_Question` tool and return a single word: `pdf`, `web`, or `direct`. This task runs inside its own sequential `Crew`, kicked off with `kickoff_async`.
3. The router's raw output is cleaned into one of the three valid routes (defaulting to `direct` if the response is noisy or unrecognized), and logged as `route_resolved`.
4. A second `Task` is built for `retriever_agent`, passing the resolved route and original question as a JSON payload (`{"route": ..., "question": ...}`) to the `Retrieve_Answer` tool. This task also runs in its own sequential `Crew` via `kickoff_async`.
5. Internally, `Retrieve_Answer` calls `retrieve_answer(route, question)`, which dispatches to the PDF search + LLM refinement, the web search (with a direct-LLM fallback on failure), or a direct LLM call — logging each sub-step along the way.
6. The final answer's length and a preview are logged (`final_answer`), a full trace summary is printed, the final answer is printed, and it's returned to the caller.

## Challenges and trade-offs

- **Keyword-based routing is simple but can misclassify ambiguous questions.** `route_question` relies on straightforward keyword matching (e.g. "pdf", "latest", "today") rather than semantic understanding, so a question that doesn't clearly signal its intent through wording may be routed to the wrong source.
- **PDF answers depend on chunk quality from `PDFSearchTool`.** The quality of PDF-routed answers is bounded by how well the underlying chunking and embedding retrieve relevant passages — a poorly chunked or oddly formatted section of the PDF can lead to less relevant excerpts being passed to the LLM for refinement.
- **Web search has a direct-LLM fallback when Tavily is unavailable or rate-limited.** If the Tavily request fails or returns an error, the system falls back to answering from the LLM's own knowledge while explicitly telling the user that real-time verification wasn't available, rather than crashing or silently returning bad data.

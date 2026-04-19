# Architecture Deep-Dive

## Why These Technology Choices?

### LangGraph (not LangChain LCEL chains)

LangGraph was chosen over linear LCEL chains for three reasons:

1. **Cyclical execution**: The engineering self-critique loop requires the ability to re-execute a node multiple times. LCEL chains are fundamentally acyclic. LangGraph's `StateGraph` natively supports cycles via conditional edges.

2. **Explicit state**: Every intermediate result is stored in the `PipelineState` TypedDict and passed explicitly between nodes. This makes the pipeline auditable, debuggable, and re-runnable from any checkpoint. LCEL chains pass outputs implicitly.

3. **Interruption and resume**: LangGraph supports checkpointing, meaning a long pipeline can be paused and resumed. This is important for pipelines that take 5–15 minutes to complete and might be interrupted.

### CrewAI (not raw LangChain agents)

CrewAI provides role-based agent abstraction on top of LangChain agents. The key benefit for this pipeline is:

- **Goal and backstory separation**: Defining an agent's `goal` separately from its `task` description allows the same agent to be reused with different tasks while maintaining consistent persona and tool access.
- **Process control**: `Process.sequential` gives deterministic execution order without requiring complex chain construction.
- **`max_iter` guardrail**: Prevents runaway tool-calling loops, which are a real failure mode with ReAct agents on complex tasks.

The Data Analyst Agent uses ReAct (reason + act loop with tools). The PM Agent uses a Plan-and-Execute pattern implemented via the task description (the LLM is explicitly instructed to plan before executing). The Engineering Agent uses a self-critique loop implemented externally in Python.

### ChromaDB (not Pinecone or Qdrant)

ChromaDB was chosen for its zero-infrastructure local mode and straightforward Docker deployment. For production at scale (> 10M chunks), Qdrant or Pinecone would offer better performance and managed hosting. ChromaDB's Python-native API is ideal for rapid prototyping and single-server deployments.

The hybrid search implementation combines dense cosine similarity with keyword-based sparse retrieval. ChromaDB's native sparse search is less sophisticated than a dedicated BM25 implementation (e.g., Elasticsearch), but is sufficient for this use case. The `hybrid_alpha` parameter allows tuning toward dense or sparse depending on the query type.

### Neo4j (not a relational DB or pure in-memory graph)

Pain-point-to-feature mapping and co-occurrence analysis are natural graph queries. The key query pattern is:

```cypher
MATCH (p:PainPoint)-[:MAPS_TO]->(f:Feature)
RETURN f.name, COUNT(p), SUM(p.frequency)
ORDER BY SUM(p.frequency) DESC
```

This is awkward in SQL (requires a JOIN + GROUP BY) and impossible in a vector store. Neo4j's Cypher syntax expresses this pattern naturally. The APOC library adds aggregate functions and graph algorithms useful for clustering.

### BAAI/bge-m3 Embeddings

BGE-M3 (from Beijing Academy of AI) was chosen because:
- It supports dense, sparse (SPLADE-style), and multi-vector retrieval in a single model
- 8192 token context window handles long feedback documents
- Strong multilingual performance (relevant for international customer bases)
- Available under MIT license with no API cost

For environments where GPU inference is not available, `BAAI/bge-small-en-v1.5` is a faster alternative that sacrifices some quality.

---

## Data Flow in Detail

### Stage 1: Ingestion

The `DocumentLoader` handles format detection and dispatch. CSV files are treated as one Document per row (one row = one survey respondent), which preserves granularity and avoids losing signal by merging respondents.

Metadata enrichment happens at ingestion: `doc_type` is inferred from filename keywords, enabling later metadata-filtered retrieval (e.g., "search only interviews").

### Stage 2: Semantic Chunking

The chunker uses a sentence-window approach rather than fixed token splitting. The algorithm:

1. Sentence-tokenise the document.
2. Embed each sentence.
3. For each position, compute the cosine similarity between the centroid of the left window and the centroid of the right window.
4. Insert a boundary where similarity drops below `SEMANTIC_SPLIT_THRESHOLD`.

This produces chunks that correspond to topical sections rather than arbitrary token windows, which significantly improves retrieval precision.

**Overlap** is implemented at the sentence level (`chunk_overlap_sentences=1`): the last sentence of chunk N becomes the first sentence of chunk N+1. This prevents context loss at boundaries.

### Stage 3: Entity Extraction

The entity extraction step runs an LLM on each chunk to extract structured pain points and map them to feature categories. This is the most expensive step (one LLM call per chunk) and is therefore run on only the top 50 chunks by length.

The LLM prompt is carefully structured to return a strict JSON schema, and the output is parsed with `json.loads`. If parsing fails, the chunk is skipped and the error is logged.

The resulting Neo4j graph enables queries that are impossible with the vector store alone:
- "Which feature has the most distinct pain points mapped to it?"
- "Which pain points co-occur most frequently?"
- "What themes cluster together?"

### Stage 4: Analysis

The Data Analyst Agent is given access to all five search tools and instructed to populate a structured `AnalysisReport` JSON. The agent is constrained to a maximum of 12 tool calls to prevent excessive LLM spend.

The `narrative_summary` field requires the agent to synthesise its tool results into prose, which forces a coherent interpretation of the data rather than just reporting raw numbers.

### Stage 5: PRD Drafting

The PM Agent receives the full `AnalysisReport` and is given a step-by-step execution plan in its task description. This Plan-and-Execute pattern is more reliable than asking the LLM to write the entire PRD in one shot, because:
- Each section can be drafted with targeted retrieval context
- The step-by-step instruction reduces the chance of the LLM skipping sections
- The structured JSON output forces completeness

### Stage 6: Engineering Review

The Engineering Agent scores the PRD on a 5-dimension rubric (0–10 total). The self-critique loop in `engineering_agent.py` is implemented as a Python `for` loop that re-runs the CrewAI crew with accumulated feedback as additional context. This is more predictable than relying on the LLM to autonomously loop.

The gate threshold is configurable (`ENGINEERING_GATE_THRESHOLD`, default 7.0). If the gate is not passed after `MAX_CRITIQUE_ROUNDS` attempts, the pipeline proceeds with a warning rather than halting, ensuring the system is never blocked indefinitely.

---

## Extending the Pipeline

### Adding a New Data Source

1. Add a loader method to `DocumentLoader` (e.g., `_load_notion`, `_load_zendesk`).
2. Add the new extension to `SUPPORTED_EXTENSIONS`.
3. Add keyword heuristics to `_TYPE_HINTS` for `doc_type` inference.

### Adding a New Agent

1. Create `src/agents/my_new_agent.py` with a `run_my_new_agent()` function.
2. Add a corresponding node function in `src/orchestration/nodes.py`.
3. Register the node and its edges in `src/orchestration/workflow.py`.
4. Add the new state fields to `PipelineState` in `src/orchestration/state.py`.

### Replacing the LLM

All LLM usage goes through `ChatOpenAI` instances initialised with settings from `.env`. To switch to Anthropic Claude:

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=...)
```

Inject this into the agent constructors in `data_analyst_agent.py`, `pm_agent.py`, and `engineering_agent.py`.

---

## Production Considerations

| Concern | Current Implementation | Production Recommendation |
|---|---|---|
| **LLM cost control** | `max_iter` on agents, subset sampling for entity extraction | Add token budget tracking; alert on >$X per run |
| **ChromaDB scale** | Single-node Docker | Qdrant Cloud or Pinecone for > 10M vectors |
| **Neo4j scale** | Community single-node | Neo4j Aura (managed) or Enterprise cluster |
| **Job persistence** | In-memory dict in `api.py` | Redis or PostgreSQL for job store |
| **Embedding caching** | None | Cache embeddings by content hash to avoid re-computing |
| **Authentication** | None | Add JWT middleware to FastAPI; secure Neo4j with vault |
| **Observability** | Loguru file logs | LangSmith for LLM traces; OpenTelemetry for infrastructure |

# Agent Tools Reference

Memory Agent Chat uses a LangGraph agent with three tools. Each tool is visible in the **Agent tools** panel after a reply.

## save_memory

**Purpose:** Store a user-specific fact for later recall within the current session.

**When to use:** Personal preferences, names, goals, or anything the user asks you to remember.

**Storage:** Pinecone **memory index**, filtered by `user_id` and `thread_id` (session).

**Example user message:** *"Remember that my favorite language is Python."*

## recall_memory

**Purpose:** Fetch relevant memories for the current user and session.

**When to use:** Automatically preloaded at the start of each turn. The agent may also call it explicitly when personal context matters.

**Storage:** Same Pinecone memory index as `save_memory`.

**Example user message:** *"What do you remember about me?"*

## retrieve_domain

**Purpose:** Search the uploaded **knowledge base** (domain index) for factual answers.

**When to use:** Questions about indexed documents, product docs, notes, PDFs, or other uploaded content.

**Storage:** Pinecone **domain index** with multimodal embeddings.

**Returns:** Numbered source blocks `[1]`, `[2]`, … shown in the **References** panel.

**Example user message:** *"What tech stack does this project use?"*

## Tool transparency UI

After each assistant reply you can expand:

- **Agent tools** — which tools ran and short summaries (e.g. "Retrieved 3 document(s)")
- **References** — filenames, page ranges, and previews for retrieved chunks

## Demo questions

- What tools does the Memory Agent have?
- What is the difference between save_memory and retrieve_domain?
- Where are user memories stored?

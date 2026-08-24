# Understanding RAG (Retrieval-Augmented Generation)

## Why RAG exists
LLM Problems: 
 - Cut Off Knowledge: 
 - Context Limitation: can't pass context in every prompt
 - Don't know information which not avaialble on the internet (e.g company private papers)

RAG Solve this problems by providing enough context to LLM to produce better result.

*RAG Components* 
 - Chunking: Split document in smaller chunks using overlap method or fixed size chunks (there are more method available for chunking)
 - Embedding: Convert this small chunks into vectors (number repesentation)
 - Vector database: store this embedding into vector store, responsible for to retriving chunks based on user query.
 - User Query: Embed user query and pass to vector database for similarity seach and vector store returns the Top-k tokens
 - Augmented: Pass retrived chunks to LLM + user query
 - LLM: return the output based on provided chunks


```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (offline, once per document)"]
        A[Raw documents] --> B[Chunking]
        B --> C[Embedding model]
        C --> D[(Vector database)]
    end

    subgraph QueryTime["Query time (every user question)"]
        E[User query] --> F[Embedding model]
        F --> G[Similarity search]
        G --> H[Top-k relevant chunks]
        H --> I[Augmented prompt<br/>query + chunks]
        I --> J[LLM]
        J --> K[Final answer]
    end

    D -.->|searched against| G
```

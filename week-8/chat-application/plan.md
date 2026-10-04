- Chat Application: 

  - sqlite persistence
  - multiple threads
  - resume chat 
  - HITL 
  - short term memory
  - long term memory
  - tools
  - mcp
  - context window problem (trimming,summarization)
 
 
 
                         ┌─────────────────────┐
                         │    Chat UI / API    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Chat Graph        │
                         │   Orchestrator      │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼────────────────────┐
              │                     │                    │
              ▼                     ▼                    ▼
       Memory Subgraph        Tool Subgraph          Response
       (independent)          (shared state)           LLM
              │                     │
       ┌──────┴──────┐       ┌──────┴──────┐
       ▼             ▼       ▼             ▼
 Short-term     Long-term   Local Tools    MCP
 memory          memory       │             │
                              ▼             ▼
                         Calculator    MCP Server
                         Search        External tools
                         Date/Time


- Chat Application: 

  - sqlite persistence [done]
  - multiple threads [done]
  - resume chat [done]
  - short term memory [done]
  - long term memory [done]
  - tools [done]
  - mcp [done]
  - HITL 
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


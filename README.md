# Custom RAG for Cyber Threat Intel

Experimental code for building a custom RAG model for CTI. 

<!-- ![Workflow](./RAG_Diagram.png) -->

**References:**
* https://www.youtube.com/watch?v=0zgYu_9WF7A
* https://www.youtube.com/watch?v=75uBcITe0gU&t=565s


```mermaid
---
config:
  theme: neutral
---
flowchart TD
    A1["RSS Sources"] --> B["Python Jupyter Notebook"]
    A2["CSV Sources"] --> B
    A3["PDF Sources"] --> B
    B --> C{"Langchain"}
    C -- Convert to Documents --> D["Document Transformation"]
    D --> E{"ChromaDB"}
    E -- Create Vectorstore --> F["Indexed Vector Embeddings"]
    F --> G{"Ollama LLM"}
    G -- Query Vectorstore --> H["Retrieval & Generation"]
    H --> I["User Query Response"]
     A1:::sources
     B:::process
     A2:::sources
     A3:::sources
     C:::process
     D:::process
     E:::storage
     F:::storage
     G:::model
     H:::model
     I:::model
    classDef sources fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:2px
    classDef storage fill:#bfb,stroke:#333,stroke-width:2px
    classDef model fill:#fb0,stroke:#333,stroke-width:2px

```
# rag_cyber_threat_intel.py
from langchain_community.document_loaders import RSSFeedLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama.llms import OllamaLLM

import config
import os

def main():
    # Configurations
    REGENERATE_VECTOR_STORE = False  # Set to True to rebuild vector store from data sources
    vectordb_path = config.VECTORSTORE_PERSIST_DIR

    # Check if vectorstore DB exists
    # (You had logic that actually forces vectordb_exists = not REGENERATE_VECTOR_STORE)
    vectordb_exists = len(os.listdir(vectordb_path)) != 0
    vectordb_exists = not REGENERATE_VECTOR_STORE  # This overrides the actual directory check per your notebook
    
    if not vectordb_exists:
        # Load CSV docs
        csv_loader = CSVLoader(file_path='rag_documents/enterprise-attack-v16.csv')
        csv_docs = csv_loader.load()

        # Load RSS feed documents
        all_rss_urls = config.RSS_INTEL_REPORTS_URLS + config.RSS_INTEL_TOOLS_URLS
        rss_loader = RSSFeedLoader(urls=all_rss_urls)
        rss_docs = rss_loader.load()

        # Load PDF docs
        pdf_loader = PyPDFLoader(file_path='rag_documents/ATTACK_Design_and_Philosophy_March_2020.pdf')
        pdf_docs = pdf_loader.load()

        # Combine all docs
        all_docs = csv_docs + rss_docs + pdf_docs

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(all_docs)

        # Initialize embeddings model
        local_embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDINGS_MODEL)

        # Batch insert to vectorstore to avoid batch size limits
        batch_size = 5461  # max batch size per your notebook
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            vectorstore = Chroma.from_documents(
                documents=filter_complex_metadata(batch),
                embedding=local_embeddings,
                persist_directory=vectordb_path
            )
        # Persist vectorstore on disk automatically happens via Chroma

    else:
        # Load existing vectorstore
        local_embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDINGS_MODEL)
        vectorstore = Chroma(
            embedding_function=local_embeddings,
            persist_directory=vectordb_path
        )

    # Use vectorstore retriever for query
    question = "Summarize the ATT&CK Design Philosophy."
    retriever = vectorstore.as_retriever(search_type=config.VECTORSTORE_SEARCH_TYPE, search_kwargs={"k": 4})

    # Retrieve matching docs for question
    retrieved_docs = retriever.invoke(question)

    # Concatenate content as context for LLM
    context = ' '.join([doc.page_content for doc in retrieved_docs])

    # Initialize the Ollama LLM
    llm = OllamaLLM(model=config.OLLAMA_LLM_MODEL)

    # Query LLM with context and question
    prompt = f"""
    Answer the question according to the context:
        Question: {question}
        Context: {context}
    """
    response = llm.invoke(prompt)

    # Print the response (you can format or further process this as needed)
    print(response)


if __name__ == "__main__":
    main()


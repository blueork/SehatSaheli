## the purpose of this script is to read the data from the nutrition_data.csv
## convert into the documents format for embedding purpose
## load the required embedder
## create the required FAISS vector db
## save the FAISS vector db with the created embeddings

import sys
import os

## for reading the csv file and converting into docouments format
from langchain_community.document_loaders import CSVLoader
## for loading the required embedder
from langchain_huggingface import HuggingFaceEmbeddings
## for loading the FAISS vector db
from langchain_community.vectorstores import FAISS

## directory to store the FAISS DB in
FAISS_DB_DIR = "./faiss_db"

## reads the csv and returns the content as a list of documents as required for embeddding
def get_documents():
    loader = CSVLoader(
        file_path='./nutrition_data.csv',
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['id', 'title', 'description', 'keywords']
        }
    )

    documents = loader.load()

    ## print statements to help in understanding the structure of the documents
    # print(documents[1])
    # print()
    # print(documents[1].page_content)
    # print()
    # print(documents[1].metadata)

    return documents

## returns the embedder to be used for embeddings
def get_embedder(): 
    embeddings = None
    try:
        # Use a multilingual embedding model for better performance with non-English queries
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Test if the embedding model can actually embed a simple string
        test_embedding = embeddings.embed_query("test string")
        
        if not test_embedding or len(test_embedding) == 0:
            raise ValueError("Embedding model failed to produce embeddings for a test string.")
        # DEBUG: Print a sample embedding
        print(f"DEBUG: Sample embedding (first 5 values): {test_embedding[:5]}")
        print("Embedding model initialized and tested successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model: {e}. Please check your internet connection or model name.")
         # Stop execution if embedding model fails
        sys.exit(1)

## creates the required FAISS vector store
## if there pre-exists a vector store it first deletes it and returns it
def create_vector_store():

    documents = get_documents()
    embedding = get_embedder()

    vectorstore = None
    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    vectorstore = FAISS.from_documents(
        documents[1:],
        embedding,
    )
    vectorstore.save_local(FAISS_DB_DIR)

    print(f"FAISS Vector DB created. Number of items in collection: {vectorstore.index.ntotal}")
    print("FAISS vector store created and persisted.")

     # --- DEBUG: Test retriever with a known query ---
    if vectorstore:
        test_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        test_query = "lentils and rice" # A keyword directly from one of your documents
        test_retrieved_docs = test_retriever.invoke(test_query)
        print(f"DEBUG: Retriever test for '{test_query}': Found {len(test_retrieved_docs)} documents.")
        if test_retrieved_docs:
            print(f"DEBUG: Test retrieved doc content (first 50 chars): {test_retrieved_docs[0].page_content[:50]}...")
        else:
            print(f"DEBUG: Retriever test for '{test_query}' found no documents. This is a concern.")
    # --- END DEBUG: Test retriever ---

    return vectorstore

## returns the created FAISS vector store
def get_vector_store():
    embedding = get_embedder()

    vectorstore = FAISS.load_local(
        FAISS_DB_DIR, embedding, allow_dangerous_deserialization=True
    )

    ## debugging step
    print(f"FAISS Vector DB loaded. Number of items in collection: {vectorstore.index.ntotal}")
    # --- DEBUG: Test retriever with a known query ---
    if vectorstore:
        test_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        test_query = "lentils and rice" # A keyword directly from one of your documents
        test_retrieved_docs = test_retriever.invoke(test_query)
        print(f"DEBUG: Retriever test for '{test_query}': Found {len(test_retrieved_docs)} documents.")
        if test_retrieved_docs:
            print(f"DEBUG: Test retrieved doc content (first 50 chars): {test_retrieved_docs[0].page_content[:50]}...")
        else:
            print(f"DEBUG: Retriever test for '{test_query}' found no documents. This is a concern.")
    # --- END DEBUG: Test retriever ---


    return vectorstore   
    


if __name__ == "__main__":
    # create_vector_store()
    get_vector_store()




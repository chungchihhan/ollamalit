from ollama import Client
import streamlit as st
import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Ollama Create Chatbot", layout="wide")
st.title("Ollama Embeddings")

HOSTNAME = "http://localhost:11434"
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

root_path = os.getcwd()
chroma_path = os.path.join(root_path, "chroma-files")
if "chromadb_client" not in st.session_state:
    st.session_state.chromadb_client = chromadb.PersistentClient(path=chroma_path)
collection = st.session_state.chromadb_client.get_or_create_collection(name="pdf")

rag_path = os.path.join(root_path, "rag-files/TS31103-GXA-S背隙調整.pdf")
loader = PyPDFLoader(rag_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)


for i, d in enumerate(splits):
    response = st.session_state.ollama_client.embeddings(model="all-minilm", prompt=d.page_content)
    embedding = response["embedding"]
    collection.upsert(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d.page_content],
        metadatas=[d.metadata]
    )

# an example prompt
prompt = "背隙調整的步驟是什麼？"

# generate an embedding for the prompt and retrieve the most relevant doc
response = st.session_state.ollama_client.embeddings(
  prompt=prompt,
  model="all-minilm"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

output = st.session_state.ollama_client.generate(
  model="llama3",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

st.write(output['response'])

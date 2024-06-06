from ollama import Client
import streamlit as st
import chromadb
import os

st.set_page_config(page_title="Ollama Create Chatbot", layout="wide")
st.title("Ollama Embeddings")

HOSTNAME = "http://localhost:11434"
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

root_path = os.getcwd()
chroma_path = os.path.join(root_path, "chroma-files")
if "chromadb_client" not in st.session_state:
    st.session_state.chromadb_client = chromadb.PersistentClient(path=chroma_path)

collection = st.session_state.chromadb_client.get_or_create_collection(name="docs")

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = st.session_state.ollama_client.embeddings(model="all-minilm", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

# an example prompt
prompt = "What animals are llamas related to?"

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

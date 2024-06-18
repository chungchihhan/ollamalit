from ollama import Client
import streamlit as st
import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ollama RAG Chatbot", layout="wide")
st.title(":rainbow[Ollamalit] RAG Chatbot")

# Set up Ollama client
HOSTNAME = "http://localhost:11434"
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

# Initialize list of local models
local_model_list = []
local_model_dict = st.session_state.ollama_client.list()
for m in local_model_dict["models"]:
    local_model_list.append(m["name"])

with st.sidebar:
    st.write("## Model Selection")
    embed_model  = st.selectbox("Select a Embedding Model", local_model_list, index=1)
    chat_model = st.selectbox("Select a Chat Model", local_model_list)


# Get the root path for the project
root_path = os.getcwd()

# Split the file you have selected
rag_path = os.listdir(os.path.join(root_path, "rag-files"))
for i in rag_path:
    if i.endswith(".gitignore"):
        rag_path.remove(i)

selected_file = st.selectbox("Select a file", rag_path, help="Put your pdf files in the rag-files folder")
rag_file_path = os.path.join(root_path, "rag-files", selected_file)

# Set up ChromaDB client
collection_name = selected_file.replace(".pdf", "")

try: 
    chroma_path = os.path.join(root_path, "chroma-files")
    if "chromadb_client" not in st.session_state:
        st.session_state.chromadb_client = chromadb.PersistentClient(path=chroma_path)
    collection = st.session_state.chromadb_client.get_or_create_collection(name=collection_name)
except Exception as e:
    st.error(f"The pdf file name is not valid.")
    st.info(e)
    st.stop()

loader = PyPDFLoader(rag_file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Embed the splits
try:
    for i, d in enumerate(splits):
        response = st.session_state.ollama_client.embeddings(model=embed_model, prompt=d.page_content)
        embedding = response["embedding"]
        collection.upsert(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[d.page_content],
            metadatas=[d.metadata]
        )
except Exception as e:
    st.error(f"Error: {e}")
    st.info("You need a embdedding model to run this app, e.g. **all-minilm**.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_role" not in st.session_state:
    st.session_state.message_role = []

for role, message in zip(st.session_state.message_role, st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(f"**{role}**")
        st.markdown(message["display_content"])

# Generate response
def response_generator(model_name):
    stream = st.session_state.ollama_client.chat(
        model=model_name,
        messages=st.session_state.messages,
        stream=True,
    )
    # st.write(st.session_state.messages)
    for chunk in stream:
      yield chunk["message"]["content"]


st.session_state["ai_model"] = chat_model

# Accept user chat input. Save in session and show them in the chat container.
if prompt := st.chat_input("What is up?"):
    question_embeddings = st.session_state.ollama_client.embeddings(
        prompt=prompt,
        model=embed_model,
    )

    similar_search = collection.query(
        query_embeddings=[question_embeddings["embedding"]],
        n_results=5,
    )

    distances = similar_search["distances"]
    if float(distances[0][0]) < 15:
        similar_context = similar_search["documents"][0][0]
        prompt_template = f"""Based on the following context: {similar_context}, answer the question: **{prompt}**."""
    else:
        prompt_template = f"""{prompt}"""

    st.session_state.messages.append({"role": "user", "content": prompt_template, "display_content": prompt})
    st.session_state.message_role.append("You")

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown("**You**")
        st.markdown(prompt)
    # Display assistant response in chat message container

    with st.chat_message("assistant"):
        # with st.spinner("Initializing model..."):
        #     st.session_state.ollama_client.generate(
        #         model=st.session_state.ai_model, keep_alive="10m"
        #     )

        st.markdown(f"**{st.session_state.ai_model}**")
        response = st.write_stream(response_generator(chat_model))
    st.session_state.messages.append({"role": "assistant", "content": response, "display_content": response})
    st.session_state.message_role.append(st.session_state.ai_model)

from ollama import Client
import streamlit as st
import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ollama RAG Chatbot", layout="wide")
st.title(":rainbow[Ollamalit] RAG Chatbot")

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


local_model_list = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_role" not in st.session_state:
    st.session_state.message_role = []

if "add_system_message" not in st.session_state:
    st.session_state.add_system_message = False

for role, message in zip(st.session_state.message_role, st.session_state.messages):
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(f"**{role}**")
        st.markdown(message["content"])



def response_generator():
    stream = st.session_state.ollama_client.chat(
        model="llama3:8b-instruct-q4_K_M",
        messages=st.session_state.messages,
        stream=True,
    )
    # st.write(st.session_state.messages)
    for chunk in stream:
      yield chunk["message"]["content"]

st.session_state["ai_model"] = "llama3:8b-instruct-q4_K_M"

if prompt := st.chat_input("What is up?"):
    question_embeddings = st.session_state.ollama_client.embeddings(
        prompt=prompt,
        model="all-minilm"
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

    st.session_state.messages.append({"role": "user", "content": prompt＿template})
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
        response = st.write_stream(response_generator())
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.message_role.append(st.session_state.ai_model)

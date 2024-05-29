import os
import streamlit as st
from ollama import Client

st.set_page_config(page_title="Create model", layout="wide")
st.title("Ollamalit Create models")

# Initiate ollama client
HOSTNAME = "http://localhost:11434"
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

root_path = os.getcwd() # Get the current working directory
gguf_path = os.path.join(root_path, "gguf-files")

st.info(f"Please put the gguf file in the folder: ***{gguf_path}***",icon="❗")

gguf_list = os.listdir(gguf_path)
gguf_list_filter = [i for i in gguf_list if i.endswith(".gguf")]
option = st.selectbox("Select a gguf file", gguf_list_filter, help="You can download the gguf file from Huggingface.")

select_path = gguf_path + "/" + option

model_name = st.text_input("Name you model", help="This is the name of the new model that will be created")
template = st.text_area(
    label="Enter the template",
    height=200,
)
system = st.text_area(
    label="Enter the system",
    height=100,
)
parameters = st.text_area(
    label="Enter the parameters",
    height=100,
)
parameters_list = parameters.split("\n")
parameters = "\n".join([f"PARAMETER {p}" for p in parameters_list])

modelfile = f"""
From {select_path}
TEMPLATE {template}
SYSTEM {system}
{parameters}
"""

def create_model():
    st.session_state.ollama_client.create(model=model_name, modelfile=modelfile)

if "create_model_state" not in st.session_state:
    st.session_state.create_model_state = False

def change_create_model_state():
    st.session_state.create_model_state = True

with st.container(border=True):
    st.subheader("Modelfile Preivew")
    st.code(modelfile)

_, _, _, _, bt = st.columns(5)
bt.button("Create model", on_click=change_create_model_state, type="primary",use_container_width=True)

if st.session_state.create_model_state:
    with st.spinner("Creating model..."):
        create_model()
        st.toast("Model created successfully!")
        st.session_state.create_model_state = False
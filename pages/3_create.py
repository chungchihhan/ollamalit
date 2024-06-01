import os
import re
import streamlit as st
from ollama import Client

st.set_page_config(page_title="Create model", layout="wide")
st.title("Ollamalit Create models")

# Initiate ollama client
HOSTNAME = "http://localhost:11434"
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

try:
    model_list = {"model_name": []}
    for name in st.session_state.ollama_client.list()["models"]:
        model_list["model_name"].append(name["name"])
except Exception as e:
    st.error("Your ollama should be running in the backend on http://localhost:11434. Please check the connection.")
    st.stop()

root_path = os.getcwd() # Get the current working directory
gguf_path = os.path.join(root_path, "gguf-files")

st.info(f"Please put the gguf file in the folder: ***{gguf_path}***",icon="❗")

gguf_list = os.listdir(gguf_path)
gguf_list_filter = [i for i in gguf_list if i.endswith(".gguf")]
option = st.selectbox("Select a gguf file", gguf_list_filter, help="You can download the gguf file from Huggingface.")

with st.container(border=True):
    col1, col2 = st.columns(2)
    source = col1.radio(
        "**Choose the source of the model**👇", ["origin model","gguf file"]
    )
    if source == "origin model":
        col2.markdown("**Select a model to copy from:**")
        select_path = col2.selectbox(
            "*Select a model to copy from*👇", model_list["model_name"], label_visibility="collapsed"
        )
    elif source == "gguf file":
        col2.markdown("**Select a gguf file:**")
        option = col2.selectbox("Select a gguf file ", gguf_list, label_visibility="collapsed")
        select_path = gguf_path + "/" + option

model_name = st.text_input("Name you model", help="This is the name of the new model that will be created")

with st.container(border=True):
    col1, col2 = st.columns(2)
    template_toggle = col1.toggle("Import template", False)
    col1.info("Turn on to import a template from an existing model.")
    if template_toggle:
        col2.markdown("**Select a template to import**")
        import_template = col2.selectbox(
            "Select a template to import", model_list["model_name"], label_visibility="collapsed"
        )

template = st.text_area(
    label="Enter the template",
    height=200,
    value=st.session_state.ollama_client.show(import_template)["template"] if template_toggle else "",
)
try:
    system = st.text_area(
        label="Enter the system",
        height=100,
        value=st.session_state.ollama_client.show(import_template)["system"] if template_toggle else "",
    )
except:
    system = st.text_area(
        label="Enter the system",
        height=100,
    )
parameters = st.text_area(
    label="Enter the parameters",
    height=150,
    value=(
        re.sub(r"[ ]+", " ", st.session_state.ollama_client.show(import_template)["parameters"])
        if template_toggle
        else ""
    ),
)
parameters_list = parameters.split("\n")
parameters = "\n".join([f"PARAMETER {p}" for p in parameters_list])

modelfile = f'''
FROM {select_path}
TEMPLATE """{template}"""
SYSTEM """{system}"""
{parameters}
'''

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
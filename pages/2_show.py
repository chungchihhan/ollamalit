import os
import platform
import streamlit as st
from ollama import Client
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from pypdf import PdfReader, PdfWriter
import re

st.set_page_config(page_title="Show Chatbot", layout="wide")
st.title("Ollama Show models")

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

options = st.selectbox("Select a model", model_list["model_name"])

st.header("Modelfile")
with st.container(border=True, height=500):
    st.code(st.session_state.ollama_client.show(options)["modelfile"])

st.subheader("Model template")
st.code(st.session_state.ollama_client.show(options)["template"])

if "system" in st.session_state.ollama_client.show(options):
    st.subheader("Model system")
    st.code(st.session_state.ollama_client.show(options)["system"])

st.subheader("Model parameters")
st.code(re.sub(r"[ ]+", " ", st.session_state.ollama_client.show(options)["parameters"]))

def delete(options):
    st.session_state.ollama_client.delete(options)
    st.write("Model deleted successfully!")

with st.popover("Delete this Model"):
    st.markdown("**Are you sure you want to delete?**")
    st.button("Delete", on_click=delete, args=(options,), type="primary")
import streamlit as st
import requests

st.set_page_config(
    page_title="Ollamalit",
    page_icon="👋",
    layout="wide",
)

st.balloons()
st.title("Welcome to :rainbow[Ollamalit]! 👋")
st.caption("A frontend interface for you interact with your ollama models.")

st.divider()

col1, col2 = st.columns(2)

col1.header("Why did we build this?")
col1.markdown(
    """
    We built this interface to make it easier for you to interact with your models.

    Originally, interacting with models requires using the terminal or writing code, which can be inconvenient or challenging for many users. Our goal is to allow users to use and manage their models through a simple and intuitive front-end interface, keeping them away from complex terminal operations.

    Ollamalit provides a *user-friendly interface* that enables you to easily chat with your models, create a new model, and make adjustments without needing to delve into the technical backend details. We hope this design enhances your experience, making it more convenient for everyone to enjoy the powerful capabilities of machine learning and artificial intelligence.
    """
)

col2.header("References")
col2.markdown(
    """
    Check out the following resources for more information:
    - [Streamlit](https://streamlit.io/)
    - [Ollama](https://ollama.com/)
    - [Ollama-python](https://github.com/ollama/ollama-python)
    """
)
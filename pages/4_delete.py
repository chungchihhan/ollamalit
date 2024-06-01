import pandas as pd
import streamlit as st
from ollama import Client

st.set_page_config(page_title="Delete model", layout="wide")
st.title(":rainbow[Ollamalit] Delete models")

# Initiate ollama client
HOSTNAME = "http://localhost:11434"
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

def load_model_data():
    model_list = {"model_name": []}
    for name in st.session_state.ollama_client.list()["models"]:
        model_list["model_name"].append(name["name"])
    details_list = []
    for model_name in model_list["model_name"]:
        try:
            details = st.session_state.ollama_client.show(model_name)["details"]
            details_list.append({
                "Model Name": model_name,
                "Parent Model": details.get("parent_model", ""),
                "Format": details.get("format", ""),
                "Family": details.get("family", ""),
                "Families": details.get("families", []),
                "Parameter Size": details.get("parameter_size", ""),
                "Quantization Level": details.get("quantization_level", "")
            })
        except Exception as e:
            st.warning(f"Failed to load details for model: {model_name}. Error: {e}")
            continue

    df = pd.DataFrame(details_list)
    df["Delete"] = False
    df = df[["Delete"] + [col for col in df.columns if col != "Delete"]]
    return df

if "model_df" not in st.session_state:
    st.session_state.model_df = load_model_data()

edited_df = st.data_editor(st.session_state.model_df, hide_index=True)

deleted_models = edited_df[edited_df["Delete"] == True]

with st.popover("Delete"):
    st.markdown("**Are you sure you want to delete these models?**")
    if deleted_models["Model Name"].any():
        for model_name in deleted_models["Model Name"]:
            st.text(f"- {model_name}")
        if st.button("Delete models"):
            for model_name in deleted_models["Model Name"]:
                st.session_state.ollama_client.delete(model_name)
                st.toast(f"Model {model_name} deleted successfully!")
            st.session_state.model_df = load_model_data()
            st.rerun()


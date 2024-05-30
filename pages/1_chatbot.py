import streamlit as st
from ollama import Client

st.set_page_config(page_title="Ollama Chat", layout="wide")
st.title("Chat with Ollama Model")

# Initiate ollama client
HOSTNAME = "http://localhost:11434"

if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = Client(host=HOSTNAME)

# Initialise list of local models
local_model_list = []

# Initialize chat history
if "messages" not in st.session_state:
    # st.session_state.messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    # st.session_state.messages.append({"role": "system", "content": SYSTEM_MESSAGE})
    st.session_state.messages = []
if "message_role" not in st.session_state:
    st.session_state.message_role = []

if "add_system_message" not in st.session_state:
    st.session_state.add_system_message = False


# Display chat messages from history on app rerun
for role, message in zip(st.session_state.message_role, st.session_state.messages):
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(f"**{role}**")
        st.markdown(message["content"])


def response_generator():
    stream = st.session_state.ollama_client.chat(
        model=st.session_state.ai_model,
        messages=st.session_state.messages,
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]



try:
    with st.sidebar:
        local_model_dict = st.session_state.ollama_client.list()
        for m in local_model_dict["models"]:
            local_model_list.append(m["name"])

        option = st.selectbox("Select a model", local_model_list)
        st.session_state["ai_model"] = option

        if st.toggle("System Message or not"):
            SYSTEM_MESSAGE = st.text_input(label="Enter a system message")
            if SYSTEM_MESSAGE and not st.session_state.add_system_message:
                st.session_state.messages.append(
                    {"role": "system", "content": SYSTEM_MESSAGE}
                )
                st.session_state.message_role.append("system")
                st.session_state.add_system_message = True
            else:
                pass

    if option in local_model_list:
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # TOBEDONE: RAG
            # Add rag result to prompt template
            rag = ""
            PROMPT_TEMPLATE = f"""{rag} {prompt}"""

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.message_role.append("You")
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown("**You**")
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(f"**{st.session_state.ai_model}**")
                with st.spinner("Initializing model..."):
                    st.session_state.ollama_client.generate(
                        model=st.session_state.ai_model, keep_alive="10m"
                    )
                response = st.write_stream(response_generator())
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.message_role.append(st.session_state.ai_model)
except Exception as e:
    st.error("Your ollama should be running in the backend on http://localhost:11434. Please check the connection.")
    st.stop()
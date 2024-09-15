import requests
import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, bytes):
            st.audio(content, format="audio/wav")
        else:
            st.markdown(content)

if prompt := st.chat_input("Write your prompt here"):
    with st.chat_message("assistant"):
        response = requests.get(
            f"http://localhost:8000/generate/audio?prompt={prompt}",
        ).content
        st.text("Here is the audio file")
        st.audio(response)
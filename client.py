import requests
import streamlit as st


st.title("Sentiment Analysis")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Write your prompt here"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.text(prompt)
    
    with st.chat_message("assistant"):
        response = requests.get("http://localhost:8000/generate/text?prompt={prompt}"
        ).text

        st.text(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
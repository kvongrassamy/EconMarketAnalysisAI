import streamlit as st
from agent import graph
import os

def generate_response(input_text):
    msgs = ['']
    for s in graph.stream({"messages": [("user",input_text)]}, {"recursion_limit": 100}, subgraphs=True):
        st.empty()
        if list(s[1])[0] == 'supervisor':
            print(s)
            if list(s[1].values())[0]['next'] != '__end__':
                st.write(f"""SUPERVISOR: Reviewing prompt with AI Agents...""".format())
        elif list(s[1])[0] in ["economist_agent", "evaluator_agent"]:
            ai_msg = list(s[1].values())[0]['messages'][0].content
            ai_name = list(s[1])[0].replace("_", " ")
            st.write(f"""{ai_name.upper()}:\n {ai_msg}""")
        else:
            message_info = list(s[1].values())[0]['messages'][0]
            print(message_info.content)
            print("-----"*20)

def init_chat():
    """Initialize chat session state with specific agent ID"""
    # Always reinitialize messages when switching agents
    st.session_state.messages = []

def show_chat(prompt_placeholder: str = "Ask me to pull an article!", extra_info: str = ""):
    st.markdown("---")
    st.subheader("Ask whats currently going on in the economy!")


    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].split("*&()")[0])
    
    # Chat input
    if prompt := st.chat_input(prompt_placeholder):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt + f"*&() Use this extra information to provide better context and details: {extra_info}"})
        with st.chat_message("user"):
            st.markdown(prompt)

        #st.st

        with st.chat_message("my_form"):
            #submitted = st.form_submit_button("Submit")
            st.session_state.messages.append({"role": "assistant", "content": prompt})
            generate_response(prompt)
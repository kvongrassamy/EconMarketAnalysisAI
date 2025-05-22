import streamlit as st
from chat import init_chat, show_chat
from agent import __init_marketresearch__


def init_session_state():
    """Initialize session state variables"""
    # Initialize chat with home page agent
    init_chat()

def main():
    #Turn off if you need to save on tavily tokens
    
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        __init_marketresearch__()
        st.session_state.initialized = True  # Mark initialization as done to avoid re-running
    else:
        st.write("Market Research complete!")
    with st.sidebar:
        st.title("Promt Current Econ Questions")
        st.write("Welcome to your team of AI Agents to do market research and analysis")
        # Initialize session state
        init_session_state()
        
        # Show features
        st.subheader("Prompt Help: Ask questions on the following categories")
        st.write("""
        - 💊 Healthcare
        """)
        st.write("""
        - 💸 Investments
        """)
        st.write("""
        - 🏛️ Finance
        """)
        st.write("""
        - 🏗️ Construction 
        """)
        st.write("""
        - 💻 Tech
        """)
        st.write("""
        - 🏠 Real Estate
        """)
        st.write("""
        - 📈 Economics
        """)
    
    # Show chat interface
    
    show_chat("Ask your econ question!  Your team will begin their market analysis")

if __name__ == "__main__":
    main()
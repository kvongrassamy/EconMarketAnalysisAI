import streamlit as st
from chat import init_chat, show_chat
from agent import __init_marketresearch__


def init_session_state():
    """Initialize session state variables"""
    # Initialize chat with home page agent
    init_chat()

def main():

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

        st.write("""
                Press to get updated market information
        """)

        if st.button("Initialize Market Research", use_container_width=True):
            st.write("Collecting current market news...")
            __init_marketresearch__()
            st.write("Market research is complete!")
    
    # Show chat interface
    
    show_chat("Ask your econ question!  Your team will begin their market analysis")

if __name__ == "__main__":
    main()
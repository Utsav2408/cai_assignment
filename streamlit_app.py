import streamlit as st

# --- Title and Description ---
st.set_page_config(page_title="ChatGPT Streamlit Interface", page_icon="💬", layout="wide")
st.title("💬 ChatGPT Chat Interface")
st.markdown("Ask me anything! This is a simple chat interface built with Streamlit.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input and Response Logic ---
def get_response(user_input):
    # Here, you'd replace this function to integrate any backend, AI model, or API.
    # Example: Call OpenAI API, or a local LLM, or custom logic.
    return f"You said: '{user_input}' — I am a simple bot for now! 🚀"

# --- User Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI/Bot response
    response = get_response(user_input)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)

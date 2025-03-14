import streamlit as st
import time  # Just for simulating delay, remove in production
from advanced_rag import *

# --- Page Configuration ---
st.set_page_config(page_title="Chat Interface with RAG Options", page_icon="💬", layout="wide")
st.title("💬 Group 97's RAG-Powered Chat Interface with Immediate Response")

st.markdown("""
Welcome! Use the toggle below to switch between **Basic RAG** and **Advanced RAG** modes.  
Each mode keeps a separate chat history to help you stay organized.
""")

# --- Session State Initialization for Both Modes ---
if "basic_messages" not in st.session_state:
    st.session_state.basic_messages = [{"role": "assistant", "content": "Hi! You are now in **Basic RAG** mode. How can I help you?"}]
if "advanced_messages" not in st.session_state:
    st.session_state.advanced_messages = [{"role": "assistant", "content": "Hi! You are now in **Advanced RAG** mode. How can I help you?"}]
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Basic RAG"  # Default mode

# --- Sidebar for Controls (RAG Mode + Clear Chat) ---
with st.sidebar:
    st.header("⚙️ Settings")
    # RAG Mode toggle
    rag_mode = st.radio(
        "Select RAG Mode:",
        ("Basic RAG", "Advanced RAG"),
        horizontal=False,
        index=0 if st.session_state.rag_mode == "Basic RAG" else 1
    )

    # Clear Chat button for current mode
    if st.button("🗑️ Clear Current Chat"):
        if rag_mode == "Basic RAG":
            st.session_state.basic_messages = [{"role": "assistant", "content": "Chat cleared. You are now in **Basic RAG** mode. How can I assist you?"}]
        else:
            st.session_state.advanced_messages = [{"role": "assistant", "content": "Chat cleared. You are now in **Advanced RAG** mode. How can I assist you?"}]
        st.session_state.rag_mode = rag_mode  # Sync mode
        st.rerun()

# --- Keep chat history on RAG Mode change ---
if st.session_state.rag_mode != rag_mode:
    st.session_state.rag_mode = rag_mode  # Update mode without clearing
    st.rerun()

# --- RAG Placeholder Functions (simulate delay) ---
def basic_rag(query):
    time.sleep(2)  # Simulate processing delay
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

def advanced_rag(query):
    result = generate_financial_response_sync(query)
    print(result)
    if isinstance(result, str):
        return result
    elif isinstance(result, dict):
        response = f"**Answer:** {result['answer']}\n\n"
        response += f"**Confidence Score:** {result['confidence_score']:.2f} ({result['confidence_band']})"
        return response
    else:
        return "I'm sorry, I couldn't process this query."

# --- Select Current Mode's Chat History ---
current_chat = st.session_state.basic_messages if st.session_state.rag_mode == "Basic RAG" else st.session_state.advanced_messages

# --- Input Field ---
user_input = st.chat_input(f"Type your message here for {st.session_state.rag_mode}...")

# --- Handle New User Input and "Typing" Simulation ---
if user_input:
    # Add user message
    current_chat.append({"role": "user", "content": user_input})

    # Add temporary loading message for bot
    current_chat.append({"role": "assistant", "content": "Typing...", "status": "loading"})  # Placeholder

    # Rerun to show user message and typing
    st.rerun()

# --- Render Chat with "Typing..." Support ---
for message in current_chat:
    with st.chat_message(message["role"]):
        if message.get("status") == "loading":
            # Render typing animation or dots (for now static text, can be animated with JS/CSS in advanced cases)
            st.markdown("Assistant is typing... ⏳")
        else:
            st.markdown(message["content"])

# --- Generate and Update Response if "loading" placeholder exists ---
# This part runs AFTER rerun and shows the final answer
if current_chat and current_chat[-1].get("status") == "loading":
    user_msg = current_chat[-2]["content"]  # Last user message
    # Get response based on current mode
    if st.session_state.rag_mode == "Basic RAG":
        response = basic_rag(user_msg)
    else:
        response = advanced_rag(user_msg)
    response = response if response else "I'm sorry, I couldn't find an answer. Could you rephrase your question?"

    # Update the loading placeholder with real response
    current_chat[-1] = {"role": "assistant", "content": response}  # Replace loading with actual content

    # Rerun to show final bot answer
    st.rerun()
